# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 320.0  # Safety margin before 330s kill

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _morse_energy(x, rho):
    """Compute Morse energy for a set of positions (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return 0.0
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    er = rho * (1.0 - dist)
    er = np.clip(er, -500, 500)
    exp_er = np.exp(er)
    energy = np.sum(exp_er * (exp_er - 2.0)) / 2.0
    return float(energy)

def _min_distance(x):
    """Compute minimum distance between any pair of atoms (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return np.inf
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return float(np.min(dist))

def _is_valid(x):
    """Check if the configuration is valid (atoms at least 0.5 apart)."""
    return _min_distance(x) >= 0.5

def _generate_shell(n, seed=42, spacing=1.1):
    """Generate a compact cluster by placing atoms in concentric shells."""
    rng = np.random.default_rng(seed)
    positions = [np.array([0.0, 0.0, 0.0])]
    remaining = n - 1
    shell_index = 1
    while remaining > 0:
        shell_radius = shell_index * spacing
        # Heuristic capacity: ~4*pi*r^2 / (d^2) but simplified to ensure packing
        max_in_shell = min(remaining, int(4 * shell_index * (shell_index + 1)))
        if max_in_shell <= 0:
            max_in_shell = 1
        
        for _ in range(max_in_shell):
            u = rng.uniform(-1, 1)
            theta = rng.uniform(0, 2 * np.pi)
            r = shell_radius
            x = r * np.sqrt(1 - u**2) * np.cos(theta)
            y = r * np.sqrt(1 - u**2) * np.sin(theta)
            z = r * u
            positions.append(np.array([x, y, z]))
        
        remaining -= max_in_shell
        shell_index += 1
    return np.array(positions[:n])

def _generate_fcc_cluster(n, seed=42):
    """Generate an FCC-like cluster."""
    rng = np.random.default_rng(seed)
    radius = 1.5 * (n ** (1.0/3.0))
    points = []
    basis = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    max_coord = int(radius) + 2
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in basis:
                    p = np.array([i + dx, j + dy, k + dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    points.sort(key=lambda p: np.linalg.norm(p))
    if len(points) >= n:
        # Small jitter to break symmetry
        jitter = rng.uniform(-0.01, 0.01, (n, 3))
        return np.array(points[:n]) + jitter
    else:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                points.append(p)
        return np.array(points)

def _generate_bcc_cluster(n, seed=42):
    """Generate a BCC-like cluster."""
    rng = np.random.default_rng(seed)
    radius = 1.5 * (n ** (1.0/3.0))
    points = []
    basis = [(0, 0, 0), (0.5, 0.5, 0.5)]
    max_coord = int(radius) + 2
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in basis:
                    p = np.array([i + dx, j + dy, k + dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    points.sort(key=lambda p: np.linalg.norm(p))
    if len(points) >= n:
        jitter = rng.uniform(-0.01, 0.01, (n, 3))
        return np.array(points[:n]) + jitter
    else:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                points.append(p)
        return np.array(points)

def _generate_random_cluster(n, seed=42, min_dist=0.6):
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = 2.0 * (n ** (1.0/3.0))
    points = []
    attempts = 0
    max_attempts = 50000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            # Optimization: Check only against a subset if n is large? No, n<=80 is small enough for O(N^2)
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)

def _relax_and_validate(positions, potential, rho, fast=False):
    """
    Relax positions and check validity.
    If fast=True, uses Nelder-Mead or limited L-BFGS for speed.
    """
    n = positions.shape[0]
    try:
        if fast:
            # Use a faster method for basin hopping steps
            # Nelder-Mead is derivative-free and can be robust, but L-BFGS with few iters is often better
            # Let's use L-BFGS-B with very few iterations for the "kick"
            x_flat = positions.flatten()
            
            def objective(x):
                pos = x.reshape(-1, 3)
                if _min_distance(pos) < 0.5:
                    return 1e10
                return _morse_energy(pos, rho)
            
            def gradient(x):
                pos = x.reshape(-1, 3)
                if _min_distance(pos) < 0.5:
                    return np.zeros_like(x)
                return morse_energy_gradient(pos, rho).flatten()
            
            res = minimize(objective, x_flat, jac=gradient, method='L-BFGS-B',
                           options={'maxiter': 20, 'ftol': 1e-8})
            
            if res.fun < 1e10:
                pos_res = res.x.reshape(-1, 3)
                if _min_distance(pos_res) >= 0.5:
                    return float(res.fun), pos_res
            # Fallback to provided relax if fast fails
            energy, relaxed = relax(positions, potential, rho)
            if _min_distance(relaxed) >= 0.5:
                return float(energy), relaxed
            return float('inf'), None
        else:
            # Standard high-quality relaxation
            energy, relaxed = relax(positions, potential, rho)
            if _min_distance(relaxed) >= 0.5:
                return float(energy), relaxed
            return float('inf'), None
    except Exception:
        return float('inf'), None

def _polish(positions, rho, max_iter=500):
    """
    Perform a high-precision local minimization using L-BFGS-B.
    """
    n = positions.shape[0]
    x_flat = positions.flatten()
    
    def objective(x):
        pos = x.reshape(-1, 3)
        if _min_distance(pos) < 0.5:
            return 1e10
        return _morse_energy(pos, rho)
    
    def gradient(x):
        pos = x.reshape(-1, 3)
        if _min_distance(pos) < 0.5:
            return np.zeros_like(x)
        return morse_energy_gradient(pos, rho).flatten()
    
    try:
        res = minimize(objective, x_flat, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success or res.fun < 1e10:
            pos_res = res.x.reshape(-1, 3)
            if _min_distance(pos_res) >= 0.5:
                return float(res.fun), pos_res
    except Exception:
        pass
    
    return _morse_energy(positions, rho), positions

def _basin_hop(start_positions, potential, rho, max_iterations=100, initial_step=0.5, initial_temp=2.0):
    """
    Perform basin hopping optimization with fast relaxation steps.
    """
    n = start_positions.shape[0]
    current_energy, current_positions = _relax_and_validate(start_positions, potential, rho, fast=False)
    
    if current_positions is None:
        perturbed = start_positions + np.random.normal(0, 0.1, start_positions.shape)
        current_energy, current_positions = _relax_and_validate(perturbed, potential, rho, fast=False)
        if current_positions is None:
            return float('inf'), None
    
    best_energy = current_energy
    best_positions = current_positions.copy()
    
    step_size = initial_step
    temp = initial_temp
    rng = np.random.default_rng(int(time.time())) # Random seed for reproducibility within run? No, just use global
    
    for it in range(max_iterations):
        if _get_remaining_time() < 5.0:
            break
        
        # Mix of Gaussian noise and single-atom kicks
        if rng.random() < 0.2:
            # Kick one atom
            idx = rng.integers(0, n)
            kick = rng.normal(0, step_size * 2, 3)
            new_positions = current_positions.copy()
            new_positions[idx] += kick
        else:
            noise = rng.normal(0, step_size, current_positions.shape)
            new_positions = current_positions + noise
        
        # Fast relaxation for the candidate
        new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho, fast=True)
        
        if new_relaxed is None:
            # Try a smaller step
            noise = rng.normal(0, step_size * 0.5, current_positions.shape)
            new_positions = current_positions + noise
            new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho, fast=True)
        
        if new_relaxed is None:
            continue
        
        delta_e = new_energy - current_energy
        # Metropolis acceptance
        if delta_e < 0 or (temp > 1e-6 and rng.random() < math.exp(-delta_e / temp)):
            current_energy = new_energy
            current_positions = new_relaxed
            
            if current_energy < best_energy:
                best_energy = current_energy
                # High precision polish for the new best
                polished_e, polished_p = _polish(current_positions, rho, max_iter=100)
                if polished_p is not None and polished_e < best_energy:
                    best_energy = polished_e
                    best_positions = polished_p
                else:
                    best_positions = current_positions.copy()
        
        # Annealing
        temp *= 0.98
        step_size *= 0.99
        if it > 0 and it % 20 == 0:
            step_size *= 1.2

    return best_energy, best_positions

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    if rho is None:
        rho = 6
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate diverse initial guesses
    initial_guesses = []
    
    # 1. Shell structures
    for seed in [42, 100, 200]:
        for spacing in [1.0, 1.1, 1.2]:
            try:
                initial_guesses.append(_generate_shell(n, seed=seed, spacing=spacing))
            except Exception:
                pass
    
    # 2. FCC chunks
    try:
        initial_guesses.append(_generate_fcc_cluster(n, seed=42))
        initial_guesses.append(_generate_fcc_cluster(n, seed=100))
    except Exception:
        pass
    
    # 3. BCC chunks
    try:
        initial_guesses.append(_generate_bcc_cluster(n, seed=42))
        initial_guesses.append(_generate_bcc_cluster(n, seed=100))
    except Exception:
        pass
    
    # 4. Random clusters
    for seed in [42, 100, 200]:
        try:
            initial_guesses.append(_generate_random_cluster(n, seed=seed, min_dist=0.6))
        except Exception:
                pass

    # Relax all initial guesses in parallel
    relaxed_results = []
    
    # Limit the number of guesses to relax to save time if we have too many
    max_guesses = 15
    if len(initial_guesses) > max_guesses:
        initial_guesses = initial_guesses[:max_guesses]
        
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_guess = {
            executor.submit(_relax_and_validate, guess, potential, rho, fast=False): guess 
            for guess in initial_guesses
        }
        
        for future in as_completed(future_to_guess):
            if _get_remaining_time() < 20.0:
                # If time is running out, stop accepting new results
                break
            try:
                energy, positions = future.result(timeout=10)
                if positions is not None and energy < float('inf'):
                    relaxed_results.append((energy, positions))
            except Exception:
                pass
    
    relaxed_results.sort(key=lambda x: x[0])
    
    if relaxed_results:
        best_energy = relaxed_results[0][0]
        best_positions = relaxed_results[0][1].copy()
    
    # Perform basin hopping from the best few relaxed results
    if best_positions is not None and _get_remaining_time() > 30.0:
        num_bh_starts = min(len(relaxed_results), 3)
        
        for idx in range(num_bh_starts):
            if _get_remaining_time() < 20.0:
                break
            
            start_energy, start_pos = relaxed_results[idx]
            
            # Adjust parameters based on rho and n
            if rho <= 3:
                initial_step = 0.8
                initial_temp = 5.0
            elif rho <= 6:
                initial_step = 0.5
                initial_temp = 3.0
            else:
                initial_step = 0.3
                initial_temp = 2.0
            
            if n <= 40:
                max_iter = 150
            elif n <= 60:
                max_iter = 100
            else:
                max_iter = 60
            
            bh_energy, bh_positions = _basin_hop(
                start_pos, potential, rho,
                max_iterations=max_iter,
                initial_step=initial_step,
                initial_temp=initial_temp
            )
            
            if bh_positions is not None and bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_positions.copy()
    
    # Final polish
    if best_positions is not None and _get_remaining_time() > 5.0:
        polished_energy, polished_pos = _polish(best_positions, rho)
        if polished_pos is not None and polished_energy < best_energy:
            best_energy = polished_energy
            best_positions = polished_pos
    
    if best_positions is None:
        # Fallback: simple shell
        best_positions = _generate_shell(n, seed=0, spacing=1.1)
        if not _is_valid(best_positions):
            best_positions = best_positions * 1.5
    
    return best_positions
# EVOLVE-BLOCK-END

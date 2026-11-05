# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 325.0  # 325s wall clock limit (evaluator kills at 330s)

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
    return energy

def _min_distance(x):
    """Compute minimum distance between any pair of atoms (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return np.inf
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return np.min(dist)

def _is_valid(x):
    """Check if the configuration is valid (atoms at least 0.5 apart)."""
    return _min_distance(x) >= 0.5

def _rotation_matrix(angle, axis):
    """Create a rotation matrix."""
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    c = np.cos(angle)
    s = np.sin(angle)
    x, y, z = axis
    R = np.array([
        [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]
    ])
    return R

def _generate_icosahedron(center, radius, seed=42):
    """Generate 13-atom icosahedral cluster (1 center + 12 vertices)."""
    phi = (1 + np.sqrt(5)) / 2
    scale = radius / np.sqrt(1 + phi*phi)
    
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) * scale
    
    positions = [center.copy()]
    for v in verts:
        positions.append(center + v)
    
    return np.array(positions)

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
        jitter = rng.uniform(-0.05, 0.05, (n, 3))
        return np.array(points[:n]) + jitter
    else:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
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
        jitter = rng.uniform(-0.05, 0.05, (n, 3))
        return np.array(points[:n]) + jitter
    else:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        return np.array(points)

def _generate_shell(n, seed=42):
    """
    Generate a compact cluster by placing atoms in concentric shells.
    """
    rng = np.random.default_rng(seed)
    
    positions = [np.array([0.0, 0.0, 0.0])]
    remaining = n - 1
    
    shell_index = 1
    while remaining > 0:
        shell_radius = shell_index * 1.1
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

def _generate_superatom_cluster(n, seed=42):
    """
    Generate a cluster using super-atom approach with icosahedral sub-clusters.
    This reduces the effective dimensionality by treating groups of atoms as rigid units.
    """
    rng = np.random.default_rng(seed)
    
    # Use icosahedron (13 atoms) as building blocks
    ico_size = 13
    
    num_full = n // ico_size
    remaining = n % ico_size
    
    sub_clusters = []
    
    # Generate full icosahedra
    for i in range(num_full):
        ico = _generate_icosahedron(np.zeros(3), 1.0, seed=seed + i)
        sub_clusters.append(ico)
    
    # Handle remaining atoms
    if remaining > 0:
        if remaining <= ico_size:
            ico = _generate_icosahedron(np.zeros(3), 1.0, seed=seed + 100)
            sub_clusters.append(ico[:remaining])
    
    # Place sub-clusters in a compact arrangement
    # Use a simple grid/sphere placement
    all_positions = []
    
    if len(sub_clusters) == 0:
        # Fallback to random
        return _generate_random_cluster(n, seed=seed)
    
    # Place first at origin
    all_positions.append(sub_clusters[0])
    
    # Place others around
    for i in range(1, len(sub_clusters)):
        # Distribute on a sphere
        num = len(sub_clusters) - 1
        if num == 1:
            theta, phi = 0, 0
        else:
            theta = i * 2 * np.pi / num
            phi = np.pi / 3 * (1 + 0.5 * math.sin(i * 1.7))
        
        dist = 3.5
        offset = np.array([
            dist * np.sin(phi) * np.cos(theta),
            dist * np.sin(phi) * np.sin(theta),
            dist * np.cos(phi)
        ])
        all_positions.append(sub_clusters[i] + offset)
    
    cluster = np.vstack(all_positions)
    
    # Trim or pad to exact n
    if len(cluster) > n:
        cluster = cluster[:n]
    elif len(cluster) < n:
        radius = 3.0
        while len(cluster) < n:
            p = rng.uniform(-radius, radius, 3)
            cluster = np.vstack([cluster, [p]])
    
    return cluster

def _relax_and_validate(positions, potential, rho):
    """Relax positions and check validity."""
    n = positions.shape[0]
    try:
        energy, relaxed = relax(positions, potential, rho)
        if _min_distance(relaxed) >= 0.5:
            return energy, relaxed
        else:
            return float('inf'), None
    except Exception:
        return float('inf'), None

def _polish(positions, rho, max_iter=300):
    """
    Perform a high-precision local minimization using L-BFGS-B.
    Tighter tolerance and more iterations for final polish.
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
        grad_3d = morse_energy_gradient(pos, rho)
        return grad_3d.flatten()
    
    try:
        res = minimize(objective, x_flat, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.success or res.fun < 1e10:
            pos_res = res.x.reshape(-1, 3)
            if _min_distance(pos_res) >= 0.5:
                return res.fun, pos_res
    except Exception:
        pass
    
    return _morse_energy(positions, rho), positions

def _simulated_annealing(start_positions, potential, rho, max_iterations=150, 
                         initial_temp=5.0, cooling_rate=0.97, step_size=0.5):
    """
    Perform simulated annealing optimization with L-BFGS local minimization.
    More principled acceptance criterion than basin hopping.
    """
    n = start_positions.shape[0]
    current_energy, current_positions = _relax_and_validate(start_positions, potential, rho)
    
    if current_positions is None:
        perturbed = start_positions + np.random.normal(0, 0.1, start_positions.shape)
        current_energy, current_positions = _relax_and_validate(perturbed, potential, rho)
        if current_positions is None:
            return float('inf'), None
    
    best_energy = current_energy
    best_positions = current_positions.copy()
    
    temp = initial_temp
    
    for it in range(max_iterations):
        if _get_remaining_time() < 5.0:
            break
        
        noise = np.random.normal(0, step_size, current_positions.shape)
        new_positions = current_positions + noise
        
        new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            noise = np.random.normal(0, step_size * 0.5, current_positions.shape)
            new_positions = current_positions + noise
            new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            continue
        
        delta_e = new_energy - current_energy
        
        # Metropolis acceptance criterion
        if delta_e < 0 or (temp > 1e-8 and np.random.rand() < math.exp(-delta_e / temp)):
            current_energy = new_energy
            current_positions = new_relaxed
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
        
        temp *= cooling_rate
        
        # Occasionally increase step size to escape local minima
        if it > 0 and it % 25 == 0:
            step_size *= 1.3
    
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
    
    # Generate diverse initial guesses including super-atom approach
    initial_guesses = []
    
    try:
        initial_guesses.append(_generate_superatom_cluster(n, seed=42))
    except Exception:
        pass
    
    try:
        initial_guesses.append(_generate_shell(n, seed=42))
    except Exception:
        pass
    
    try:
        initial_guesses.append(_generate_fcc_cluster(n, seed=42))
    except Exception:
        pass
    
    try:
        initial_guesses.append(_generate_bcc_cluster(n, seed=42))
    except Exception:
        pass
    
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        try:
            initial_guesses.append(_generate_random_cluster(n, seed=seed))
        except Exception:
            pass
    
    # Relax all initial guesses
    relaxed_results = []
    for guess in initial_guesses:
        if _get_remaining_time() < 10.0:
            break
        
        energy, positions = _relax_and_validate(guess, potential, rho)
        if positions is not None and energy < float('inf'):
            relaxed_results.append((energy, positions))
    
    relaxed_results.sort(key=lambda x: x[0])
    
    if relaxed_results:
        best_energy = relaxed_results[0][0]
        best_positions = relaxed_results[0][1].copy()
    
    # Perform simulated annealing from multiple starts
    num_sa_starts = min(len(relaxed_results), 5)
    
    for idx in range(num_sa_starts):
        if _get_remaining_time() < 15.0:
            break
        
        start_energy, start_pos = relaxed_results[idx]
        
        # Tune parameters based on rho and n
        if rho <= 3:
            initial_temp = 8.0
            step_size = 0.8
        elif rho <= 6:
            initial_temp = 5.0
            step_size = 0.5
        else:
            initial_temp = 3.0
            step_size = 0.3
        
        if n <= 40:
            max_iter = 150
        elif n <= 60:
            max_iter = 100
        else:
            max_iter = 60
        
        sa_energy, sa_positions = _simulated_annealing(
            start_pos, potential, rho,
            max_iterations=max_iter,
            initial_temp=initial_temp,
            cooling_rate=0.97,
            step_size=step_size
        )
        
        if sa_positions is not None and sa_energy < best_energy:
            best_energy = sa_energy
            best_positions = sa_positions.copy()
        
        # Try a second pass with different parameters
        if _get_remaining_time() > 20.0 and idx < 3:
            sa_energy2, sa_positions2 = _simulated_annealing(
                start_pos, potential, rho,
                max_iterations=max_iter // 2,
                initial_temp=initial_temp * 1.5,
                cooling_rate=0.95,
                step_size=step_size * 1.5
            )
            if sa_positions2 is not None and sa_energy2 < best_energy:
                best_energy = sa_energy2
                best_positions = sa_positions2.copy()
    
    # Final high-precision polish
    if best_positions is not None and _get_remaining_time() > 5.0:
        polished_energy, polished_pos = _polish(best_positions, rho, max_iter=300)
        if polished_pos is not None and polished_energy < best_energy:
            best_energy = polished_energy
            best_positions = polished_pos
    
    if best_positions is None:
        best_positions = _generate_shell(n, seed=0)
        if not _is_valid(best_positions):
            best_positions = best_positions * 1.5
            if not _is_valid(best_positions):
                best_positions = _generate_random_cluster(n, seed=99)
    
    return best_positions
# EVOLVE-BLOCK-END

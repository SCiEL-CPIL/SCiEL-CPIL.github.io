# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 320.0  # 320s wall clock limit (evaluator kills at 330s)

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _elapsed_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return time.time() - _GLOBAL_START_TIME

def _morse_energy(x, rho):
    """Compute Morse energy for a set of positions (vectorized, upper triangle only)."""
    n = x.shape[0]
    if n < 2:
        return 0.0
    # Compute pairwise distances using upper triangle only
    i, j = np.triu_indices(n, k=1)
    diff = x[i] - x[j]
    dist = np.sqrt(np.sum(diff * diff, axis=1))
    er = rho * (1.0 - dist)
    er = np.clip(er, -500, 500)
    exp_er = np.exp(er)
    energy = np.sum(exp_er * (exp_er - 2.0))
    return energy

def _min_distance(x):
    """Compute minimum distance between any pair of atoms (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return np.inf
    i, j = np.triu_indices(n, k=1)
    diff = x[i] - x[j]
    dist = np.sqrt(np.sum(diff * diff, axis=1))
    return np.min(dist)

def _is_valid(x):
    """Check if the configuration is valid (atoms at least 0.5 apart)."""
    return _min_distance(x) >= 0.5

def _rotation_matrix(angle, axis):
    """Create a rotation matrix."""
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
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

def _generate_fibonacci_sphere(n, radius=1.0, seed=42):
    """Generate points on a sphere using Fibonacci spiral for even distribution."""
    rng = np.random.default_rng(seed)
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Fibonacci spiral on sphere
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    positions = []
    for i in range(n):
        t = i / max(1, n - 1)
        z = 1.0 - 2.0 * t  # z goes from 1 to -1
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * i
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        positions.append(np.array([x, y, z]))
    
    positions = np.array(positions) * radius
    # Center the cluster
    positions -= positions.mean(axis=0)
    return positions

def _generate_compact_random(n, seed=42, density=1.0):
    """Generate a compact random cluster with controlled density."""
    rng = np.random.default_rng(seed)
    # Estimate radius for n atoms at given density
    # For close packing, density ~ 0.74, so volume ~ n / 0.74
    vol = n / density
    radius = (3.0 * vol / (4.0 * math.pi)) ** (1.0/3.0)
    radius = max(1.5, radius * 1.2)  # add margin
    
    positions = []
    attempts = 0
    max_attempts = n * 200
    while len(positions) < n and attempts < max_attempts:
        attempts += 1
        # Sample from ball
        u = rng.uniform(-1, 1, 3)
        r = rng.uniform(0, 1) ** (1.0/3.0)
        p = u * r * radius
        # Check minimum distance
        if positions:
            dists = np.linalg.norm(np.array(positions) - p, axis=1)
            if np.min(dists) >= 0.5:
                positions.append(p)
        else:
            positions.append(p)
    
    # If we couldn't place all atoms, add remaining at larger radius
    while len(positions) < n:
        theta = rng.uniform(0, 2 * math.pi)
        phi = rng.uniform(0, math.pi)
        r = radius + 0.3 * rng.uniform(0, 1)
        p = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])
        positions.append(p)
    
    return np.array(positions)

def _generate_icosahedral_shell(n, seed=42):
    """Generate cluster using icosahedral symmetry shells."""
    rng = np.random.default_rng(seed)
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (12)
    ico_verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    positions = [np.zeros(3)]  # Center atom
    
    # Add shells
    shell_radii = [1.0, 1.8, 2.5, 3.2, 3.8, 4.5]
    shell_counts = [12, 24, 42, 60, 80, 110]
    
    remaining = n - 1
    for radius, max_count in zip(shell_radii, shell_counts):
        if remaining <= 0:
            break
        count = min(remaining, max_count)
        # Use icosahedron vertices scaled, plus rotations
        for i in range(count):
            v = ico_verts[i % 12].copy()
            # Apply random rotation for diversity
            if i >= 12:
                angle = rng.uniform(0, 2 * math.pi)
                R = _rotation_matrix(angle, [0, 0, 1])
                v = R @ v
            positions.append(v * radius)
        remaining -= count
    
    # If still need more, add random points on outer shell
    while len(positions) < n:
        theta = rng.uniform(0, 2 * math.pi)
        phi_angle = rng.uniform(0, math.pi)
        r = shell_radii[-1] + 0.5
        p = np.array([
            r * np.sin(phi_angle) * np.cos(theta),
            r * np.sin(phi_angle) * np.sin(theta),
            r * np.cos(phi_angle)
        ])
        positions.append(p)
    
    return np.array(positions[:n])

def _generate_fcc_chunk(n, seed=42):
    """Generate FCC-like chunk."""
    rng = np.random.default_rng(seed)
    basis = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    
    # Estimate radius needed for n atoms in FCC
    # FCC density: ~1 atom per unit cell of volume 1, so radius ~ (n/4)^(1/3)
    radius = max(2.0, (n / 4.0) ** (1.0/3.0) + 1.0)
    max_coord = int(radius) + 2
    
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in basis:
                    p = np.array([i + dx, j + dy, k + dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    points.sort(key=lambda p: np.linalg.norm(p))
    
    if len(points) >= n:
        selected = np.array(points[:n])
    else:
        selected = np.array(points)
        while len(selected) < n:
            p = rng.uniform(-radius, radius, 3)
            selected = np.vstack([selected, [p]])
    
    return selected[:n]

def _generate_hcp_chunk(n, seed=42):
    """Generate HCP-like chunk."""
    rng = np.random.default_rng(seed)
    a = 1.0  # lattice parameter
    c = 1.633 * a  # ideal HCP c/a ratio
    
    # HCP basis
    basis = [
        (0, 0, 0),
        (a, 0, 0),
        (a/2, a*math.sqrt(3)/2, 0),
        (a/2, a*math.sqrt(3)/2, c/2),
        (a, 0, c/2),
        (0, 0, c)
    ]
    
    radius = max(2.0, (n / 2.0) ** (1.0/3.0) + 1.0)
    max_coord = int(radius / a) + 2
    
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for bx, by, bz in basis:
                    p = np.array([i*a + bx, j*a + by, k*c + bz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    points.sort(key=lambda p: np.linalg.norm(p))
    
    if len(points) >= n:
        selected = np.array(points[:n])
    else:
        selected = np.array(points)
        while len(selected) < n:
            p = rng.uniform(-radius, radius, 3)
            selected = np.vstack([selected, [p]])
    
    return selected[:n]

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
        grad_3d = morse_energy_gradient(pos, rho)
        return grad_3d.flatten()
    
    try:
        res = minimize(objective, x_flat, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        if res.success or res.fun < 1e10:
            pos_res = res.x.reshape(-1, 3)
            if _min_distance(pos_res) >= 0.5:
                return res.fun, pos_res
    except Exception:
        pass
    
    return _morse_energy(positions, rho), positions

def _basin_hop(start_positions, potential, rho, max_iterations=150, initial_step=0.6, initial_temp=3.0, seed=42):
    """
    Perform basin hopping optimization with adaptive step size.
    Uses a cooling schedule: step size and temperature decrease over iterations.
    Returns (best_energy, best_positions).
    """
    rng = np.random.default_rng(seed)
    n = start_positions.shape[0]
    current_energy, current_positions = _relax_and_validate(start_positions, potential, rho)
    
    if current_positions is None:
        # Try smaller perturbation
        for scale in [0.3, 0.2, 0.1]:
            perturbed = start_positions + rng.normal(0, scale, start_positions.shape)
            current_energy, current_positions = _relax_and_validate(perturbed, potential, rho)
            if current_positions is not None:
                break
        if current_positions is None:
            return float('inf'), None
    
    best_energy = current_energy
    best_positions = current_positions.copy()
    
    step_size = initial_step
    temp = initial_temp
    no_improve_count = 0
    min_step = initial_step * 0.1
    min_temp = 0.1
    
    for it in range(max_iterations):
        if _get_remaining_time() < 8.0:
            break
        
        # Cooling schedule: exponential decay with occasional reheating
        progress = it / max(1, max_iterations - 1)
        scheduled_step = initial_step * math.exp(-3.0 * progress)
        scheduled_temp = initial_temp * math.exp(-4.0 * progress)
        
        # Blend scheduled with adaptive
        step_size = max(min_step, 0.7 * scheduled_step + 0.3 * step_size)
        temp = max(min_temp, 0.7 * scheduled_temp + 0.3 * temp)
        
        # Perturbation: mix of small and large moves
        if rng.random() < 0.15:
            # Large jump (restart-like)
            noise = rng.normal(0, step_size * 2.0, current_positions.shape)
        else:
            noise = rng.normal(0, step_size, current_positions.shape)
        
        new_positions = current_positions + noise
        
        new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            # Try smaller perturbation
            noise = rng.normal(0, step_size * 0.5, current_positions.shape)
            new_positions = current_positions + noise
            new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            no_improve_count += 1
            continue
        
        delta_e = new_energy - current_energy
        accept = delta_e < 0 or (temp > 1e-6 and rng.random() < math.exp(-min(delta_e / temp, 50.0)))
        
        if accept:
            current_energy = new_energy
            current_positions = new_relaxed
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Adaptive adjustments
        if no_improve_count > 25:
            step_size *= 0.8
            no_improve_count = 0
    
    return best_energy, best_positions

def _parallel_relax(guesses, potential, rho, max_workers=4):
    """Relax multiple initial guesses in parallel (up to max_workers threads)."""
    results = []
    
    def _worker(guess):
        return _relax_and_validate(guess, potential, rho)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, g): i for i, g in enumerate(guesses)}
        for future in as_completed(futures):
            if _get_remaining_time() < 5.0:
                break
            try:
                energy, positions = future.result(timeout=60)
                if positions is not None and energy < float('inf'):
                    results.append((energy, positions))
            except Exception:
                pass
    
    results.sort(key=lambda x: x[0])
    return results

def _parallel_basin_hop(starts, potential, rho, max_workers=4, max_iterations=100, initial_step=0.5, initial_temp=3.0):
    """Run multiple basin hopping chains in parallel."""
    results = []
    
    def _worker(args):
        start_pos, seed = args
        return _basin_hop(start_pos, potential, rho, max_iterations=max_iterations,
                         initial_step=initial_step, initial_temp=initial_temp, seed=seed)
    
    args_list = [(s, i * 1000 + 42) for i, s in enumerate(starts)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, a): i for i, a in enumerate(args_list)}
        for future in as_completed(futures):
            if _get_remaining_time() < 10.0:
                break
            try:
                energy, positions = future.result(timeout=120)
                if positions is not None and energy < float('inf'):
                    results.append((energy, positions))
            except Exception:
                pass
    
    results.sort(key=lambda x: x[0])
    return results

def _parallel_polish(positions_list, rho, max_workers=4):
    """Polish multiple position sets in parallel."""
    results = []
    
    def _worker(pos):
        return _polish(pos, rho, max_iter=300)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, p): i for i, p in enumerate(positions_list)}
        for future in as_completed(futures):
            if _get_remaining_time() < 5.0:
                break
            try:
                energy, positions = future.result(timeout=60)
                if positions is not None and energy < float('inf'):
                    results.append((energy, positions))
            except Exception:
                pass
    
    results.sort(key=lambda x: x[0])
    return results

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
    
    # 1. Fibonacci sphere (good for medium rho)
    try:
        fib = _generate_fibonacci_sphere(n, radius=2.0, seed=42)
        initial_guesses.append(fib)
    except Exception:
        pass
    
    # 2. Icosahedral shells (good for low rho)
    try:
        ico = _generate_icosahedral_shell(n, seed=42)
        initial_guesses.append(ico)
    except Exception:
        pass
    
    # 3. FCC chunk (good for high rho)
    try:
        fcc = _generate_fcc_chunk(n, seed=42)
        initial_guesses.append(fcc)
    except Exception:
        pass
    
    # 4. HCP chunk (alternative close-packed)
    try:
        hcp = _generate_hcp_chunk(n, seed=42)
        initial_guesses.append(hcp)
    except Exception:
        pass
    
    # 5. Random compact cluster
    try:
        rng = np.random.default_rng(123)
        radius = 2.5
        positions = []
        for _ in range(n):
            p = rng.uniform(-radius, radius, 3)
            positions.append(p)
        initial_guesses.append(np.array(positions))
    except Exception:
        pass
    
    # Relax all initial guesses in parallel
    relaxed_results = _parallel_relax(initial_guesses, potential, rho, max_workers=3)
    
    if relaxed_results:
        best_energy = relaxed_results[0][0]
        best_positions = relaxed_results[0][1].copy()
    
    # Perform basin hopping from best relaxed results
    if best_positions is not None and _get_remaining_time() > 40.0:
        num_bh_starts = min(len(relaxed_results), 3)
        
        # Choose BH parameters based on rho and n
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
        
        for idx in range(num_bh_starts):
            if _get_remaining_time() < 25.0:
                break
            
            start_energy, start_pos = relaxed_results[idx]
            
            bh_energy, bh_positions = _basin_hop(
                start_pos, potential, rho,
                max_iterations=max_iter,
                initial_step=initial_step,
                initial_temp=initial_temp
            )
            
            if bh_positions is not None and bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_positions.copy()
    
    # Final high-precision polish
    if best_positions is not None and _get_remaining_time() > 10.0:
        polished_energy, polished_pos = _polish(best_positions, rho, max_iter=500)
        if polished_pos is not None and polished_energy < best_energy:
            best_energy = polished_energy
            best_positions = polished_pos
    
    if best_positions is None:
        # Fallback: simple shell
        rng = np.random.default_rng(0)
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
        best_positions = np.array(positions[:n])
        if not _is_valid(best_positions):
            best_positions = best_positions * 1.5
    
    return best_positions
# EVOLVE-BLOCK-END

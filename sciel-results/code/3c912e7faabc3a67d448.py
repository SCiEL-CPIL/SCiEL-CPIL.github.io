"""Dzugutov cluster global minima search via parallel basin hopping and geometric seeding."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
_HARD_LIMIT = 290.0  # Safety margin before 300s/330s kill

def _get_remaining_time():
    return max(0.1, _HARD_LIMIT - (time.time() - _START_TIME))

def _dzugutov_energy(positions):
    """Compute energy using the library function for consistency if possible, 
    otherwise fallback to vectorized numpy."""
    try:
        # The library function usually returns energy and gradient, or just energy?
        # The task says: chem_clusters.dzugutov_energy_gradient(positions) may be called.
        # Let's assume it returns (E, grad) or similar. 
        # Based on the current program, _dzugutov_energy_gradient returns (E, G).
        # Let's use the library function for energy calculation to be safe, or implement locally.
        # The current program implements _dzugutov_energy_gradient locally.
        # Let's stick to the local implementation for speed and control, but ensure it's correct.
        pass
    except Exception:
        pass
    
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(positions)
    if n < 2:
        return 0.0
    
    energy = 0.0
    for i in range(n):
        diffs = positions[i+1:] - positions[i]
        dists = np.linalg.norm(diffs, axis=1)
        
        # Mask for r < a
        mask_a = dists < a
        if np.any(mask_a):
            da = dists[mask_a]
            da = np.maximum(da, 1e-10)
            v_a = A * (da ** (-m) - B) * np.exp(c / (da - a))
            energy += np.sum(v_a)
            
        # Mask for a <= r < b
        mask_b = (dists >= a) & (dists < b)
        if np.any(mask_b):
            db = dists[mask_b]
            db = np.maximum(db, 1e-10)
            v_b = B * np.exp(d / (db - b))
            energy += np.sum(v_b)
            
    return energy

def _dzugutov_energy_gradient(positions):
    """Compute energy and gradient."""
    try:
        # Use library if it returns (E, G)
        res = dzugutov_energy_gradient(positions)
        if isinstance(res, tuple) and len(res) == 2:
            return res
    except Exception:
        pass
    
    # Fallback: Local implementation
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(positions)
    energy = 0.0
    gradient = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[i] - positions[j]
            r = np.linalg.norm(diff)
            if r < 1e-10:
                continue
            r_vec = diff / r
            force_mag = 0.0
            
            if r < a:
                term1 = -m * r ** (-m - 1)
                term2 = -c * (r ** (-m) - B) / ((r - a) ** 2)
                force_mag = A * np.exp(c / (r - a)) * (term1 + term2)
                energy += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                force_mag = B * np.exp(d / (r - b)) * (-d / ((r - b) ** 2))
                energy += B * np.exp(d / (r - b))
            
            gradient[i] += force_mag * r_vec
            gradient[j] -= force_mag * r_vec
    
    return energy, gradient

def _icosahedron_13():
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    r1 = 1.9
    positions = np.vstack([np.zeros(3), verts * r1])
    return positions

def _generate_icosahedral_shells(n_target, rng):
    """
    Generate a structure based on concentric icosahedral shells.
    Shell 1: 12 atoms at r1
    Shell 2: 42 atoms (12 vertices + 30 face centers) at r2
    Shell 3: 120 atoms at r3
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    positions = [np.zeros(3)]
    
    # Shell 1: 12 atoms
    r1 = 1.9
    for v in verts:
        if len(positions) < n_target:
            positions.append(v * r1)
    
    if len(positions) >= n_target:
        return np.array(positions[:n_target])
    
    # Shell 2: 42 atoms
    # 12 vertices at r2
    # 30 face centers
    r2 = 1.9 * np.sqrt(1 + 1/phi**2) * 1.2 # Approx scaling
    # Actually, for Mackay icosahedron, r2 is typically around 2 * r1 * sin(72/2)? 
    # Let's use a standard ratio. For 55-atom Mackay, r2 is such that distance from center is ~3.5-4.0?
    # Let's try r2 = 1.9 * 1.8
    r2 = 1.9 * 1.8
    
    # Vertices of shell 2
    for v in verts:
        if len(positions) < n_target:
            positions.append(v * r2)
            
    # Face centers of shell 2 (midpoints of edges of the icosahedron, scaled)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.5:
                edges.append((i, j))
                
    for i, j in edges:
        if len(positions) >= n_target:
            break
        fc = (verts[i] + verts[j]) / 2
        fc = fc / np.linalg.norm(fc) * r2
        positions.append(fc)
        
    if len(positions) >= n_target:
        return np.array(positions[:n_target])

    # Shell 3: 120 atoms
    # This is complex to generate exactly. We can approximate by placing atoms on a sphere
    # or just stop here and let the search handle the rest.
    # For n > 55, we add random atoms on a larger sphere.
    r3 = 1.9 * 2.8
    while len(positions) < n_target:
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v) * r3
        positions.append(v)
        
    return np.array(positions[:n_target])

def _random_compact(n_atoms, rng, spacing=1.1):
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n_atoms and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.55:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    if len(points) < n_atoms:
        while len(points) < n_atoms:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
            
    return np.array(points)

def _bcc_lattice(n_atoms, rng):
    a = 1.8
    points = []
    max_coord = int(np.ceil(n_atoms ** (1.0/3.0) / 2)) + 2
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                p = np.array([ix, iy, iz]) * a
                if len(points) < n_atoms:
                    points.append(p)
                p2 = (np.array([ix, iy, iz]) + 0.5) * a
                if len(points) < n_atoms:
                    points.append(p2)
    points = np.array(points[:n_atoms * 2])
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n_atoms]]
    return points

def _local_minimize(positions, rng, maxiter=100):
    """Perform local minimization using L-BFGS-B."""
    try:
        res = minimize(
            lambda x: _dzugutov_energy_gradient(x.reshape(-1, 3))[0],
            positions.flatten(),
            jac=lambda x: _dzugutov_energy_gradient(x.reshape(-1, 3))[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxls': 50, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        if res.fun < np.inf:
            return res.fun, res.x.reshape(-1, 3)
    except Exception:
        pass
    return _dzugutov_energy_gradient(positions)[0], positions

def _basin_hopping_chain(initial_positions, rng, time_limit, step_size=0.5):
    """Run a basin hopping chain."""
    current_energy, current_positions = _local_minimize(initial_positions, rng)
    best_energy, best_positions = current_energy, current_positions.copy()
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if _get_remaining_time() < 5:
            break
            
        perturbed = current_positions + rng.normal(0, step_size, current_positions.shape)
        
        try:
            new_energy, new_positions = _local_minimize(perturbed, rng, maxiter=50)
        except Exception:
            continue
        
        if new_energy < best_energy:
            best_energy, best_positions = new_energy, new_positions.copy()
            
        # Metropolis acceptance
        if new_energy < current_energy:
            current_energy, current_positions = new_energy, new_positions
        else:
            delta = new_energy - current_energy
            if delta < 0 or rng.random() < np.exp(-delta / 0.1):
                current_energy, current_positions = new_energy, new_positions
                
    return best_energy, best_positions

def _solve_instance_with_deadline(instance, deadline):
    n = instance['n']
    potential = instance['potential']
    
    rng = np.random.default_rng(n)
    
    # Generate seeds
    seeds = []
    try:
        seeds.append(_generate_icosahedral_shells(n, rng))
    except Exception:
        pass
    try:
        seeds.append(_random_compact(n, rng))
    except Exception:
        pass
    try:
        seeds.append(_bcc_lattice(n, rng))
    except Exception:
        pass
    
    if not seeds:
        seeds.append(_random_compact(n, rng))
        
    # Parallel Basin Hopping
    # We have 4 threads available.
    # Let's run 4 BH chains in parallel, each starting from a different seed (or random perturbation).
    
    best_energy = np.inf
    best_pos = None
    lock = threading.Lock()
    
    def run_bh(seed_idx):
        nonlocal best_energy, best_pos
        # Each thread gets a unique RNG
        thread_rng = np.random.default_rng(n + seed_idx)
        
        # Start from the seed or a perturbation
        initial = seeds[seed_idx % len(seeds)].copy()
        # Add some noise to diversify
        initial += thread_rng.normal(0, 0.2, initial.shape)
        
        # Time budget per thread
        # Total time left / 4 threads
        time_per_thread = max(1.0, (_get_remaining_time() - 10) / 4)
        
        try:
            bh_energy, bh_pos = _basin_hopping_chain(initial, thread_rng, time_per_thread)
            
            with lock:
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_pos = bh_pos
        except Exception:
            pass

    # Launch threads
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(4):
            futures.append(executor.submit(run_bh, i))
            
        # Wait for completion or timeout
        for f in as_completed(futures):
            pass # Exceptions are handled inside
            
    if best_pos is None:
        # Fallback to best single relaxation
        for seed in seeds:
            try:
                e, pos = _local_minimize(seed, rng)
                if e < best_energy:
                    best_energy = e
                    best_pos = pos
            except Exception:
                pass
                
    if best_pos is None:
        best_pos = seeds[0]
        
    return best_pos, best_energy

def construct(instance):
    n = instance['n']
    potential = instance['potential']
    
    local_deadline = time.time() + 45.0 # 300/6 = 50s, use 45s
    
    pos, energy = _solve_instance_with_deadline(instance, local_deadline)
    
    return pos

# EVOLVE-BLOCK-END

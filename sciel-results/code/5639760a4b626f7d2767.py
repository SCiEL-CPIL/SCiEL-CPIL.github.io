"""Dzugutov cluster global minima search with Mackay icosahedron seeding and parallel basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize
import autoresearch_suite.chem_clusters as chem_clusters
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants
_TOTAL_BUDGET = 310.0  # 330s limit with 20s safety margin
_HARD_LIMIT = 325.0
_START_TIME = None
_INSTANCE_ORDER = [38, 45, 50, 55, 57, 60]
_INSTANCE_RESULTS = {}

A_DZ = 5.82
B_DZ = 1.28
a_DZ = 1.87
b_DZ = 1.94
c_DZ = 1.10
d_DZ = 0.27
m_DZ = 16

def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME

def _remaining():
    return max(0.0, _TOTAL_BUDGET - _elapsed())

def _hard_remaining():
    return max(0.0, _HARD_LIMIT - _elapsed())

def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        diffs = positions[i+1:] - positions[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        mask_a = dists < a_DZ
        mask_b = (dists >= a_DZ) & (dists < b_DZ)
        
        if np.any(mask_a):
            da = dists[mask_a]
            da = np.maximum(da, 1e-10)
            v_a = A_DZ * (da ** (-m_DZ) - B_DZ) * np.exp(c_DZ / (da - a_DZ))
            total += np.sum(v_a)
        
        if np.any(mask_b):
            db = dists[mask_b]
            db = np.maximum(db, 1e-10)
            v_b = B_DZ * np.exp(d_DZ / (db - b_DZ))
            total += np.sum(v_b)
    
    return total

def _dzugutov_energy_gradient_vec(positions):
    """Compute energy and gradient using the provided API or fallback."""
    try:
        # The API might return (energy, gradient) or just gradient? 
        # The task description says: chem_clusters.dzugutov_energy_gradient(positions)
        # Usually these return gradient. Let's assume it returns gradient only or check docstring behavior.
        # Based on common patterns in these suites, it often returns just the gradient array.
        # However, the inspiration code suggested: energy, grad = chem_clusters.dzugutov_energy_gradient(positions)
        # Let's try to call it and handle both cases.
        res = dzugutov_energy_gradient(positions)
        if isinstance(res, tuple):
            return res[0], res[1]
        else:
            # If it returns only gradient, we compute energy separately
            e = _dzugutov_energy(positions)
            return e, res
    except Exception:
        # Fallback to finite difference or manual calculation if API fails
        e = _dzugutov_energy(positions)
        grad = np.zeros_like(positions)
        eps = 1e-6
        for i in range(len(positions)):
            for j in range(3):
                pos_p = positions.copy()
                pos_p[i, j] += eps
                E_p = _dzugutov_energy(pos_p)
                grad[i, j] = (E_p - e) / eps
        return e, grad

def _enforce_min_distance(positions, min_dist=0.5):
    """Separate any atoms closer than min_dist."""
    pos = positions.copy()
    n = len(pos)
    for _ in range(50):
        needs_fix = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    needs_fix = True
                    diff = pos[i] - pos[j]
                    if np.linalg.norm(diff) < 1e-8:
                        diff = np.array([1.0, 0.0, 0.0])
                    else:
                        diff = diff / np.linalg.norm(diff)
                    push = (min_dist - d) / 2.0 + 0.01
                    pos[i] += diff * push
                    pos[j] -= diff * push
        if not needs_fix:
            break
    return pos

def _generate_mackay_icosahedron(n_atoms):
    """
    Generate a Mackay icosahedron-like structure with approximately n_atoms.
    Mackay icosahedra have sizes 1, 13, 55, 147...
    For n in [38, 60], we are between 13 and 55.
    Strategy: Generate a 55-atom Mackay icosahedron and randomly remove atoms 
    to match the target size, or generate a 13-atom core and add a shell.
    Given the range 38-60, a "truncated" 55-atom icosahedron or a 
    13-atom core + 25-47 shell atoms is best.
    
    Let's construct the 55-atom Mackay icosahedron explicitly.
    Shell 1 (12 atoms) + Shell 2 (30 atoms) + Center (1 atom) + Shell 3 (12 atoms) = 55.
    Actually, Mackay 55 is:
    1 center
    12 vertices (icosahedron)
    30 edge midpoints (icosahedron edges)
    12 face centers (icosahedron faces)
    Total 55.
    
    For N < 55, we can start with the 55-atom structure and remove the outermost atoms 
    (face centers) first, then edge midpoints, preserving the core.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere for direction, then scale
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_dir = verts / norms
    
    # Determine scale. 
    # Optimal Dzugutov distance is around 1.1-1.2.
    # In a Mackay icosahedron, the distance from center to vertex is R1.
    # Let's set R1 such that the bond length is appropriate.
    # For Dzugutov, the minimum is around r=1.15.
    # Let's try scaling the 55-atom structure to have typical bond length ~1.15.
    
    # Generate all 55 points
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: Vertices (12)
    # Distance from center to vertex in standard icosahedron with edge 2 is sqrt(phi^2+1) * something?
    # Let's just use the directions and assign radii based on Mackay structure ratios.
    # In a Mackay icosahedron with edge length L:
    # Center to Vertex: R1
    # Center to Edge Midpoint: R2
    # Center to Face Center: R3
    # Ratios: R1 : R2 : R3 = 1 : 0.809 : 0.526 (approx for unit edge? No, let's compute).
    
    # Let's construct coordinates explicitly for a Mackay icosahedron with "unit" edge length 1, then scale.
    # Standard icosahedron vertices (edge 2):
    # Vertices are at distance sqrt(1+phi^2) = sqrt(1+2.618) = 1.902 from center? No.
    # Distance from origin to [0,1,phi] is sqrt(1+phi^2).
    # Edge length between [0,1,phi] and [1,phi,0] is sqrt(1 + (phi-1)^2 + phi^2) = sqrt(1 + 0.382 + 2.618) = sqrt(4) = 2.
    # So for edge length 2, R1 = sqrt(1+phi^2) = 1.902.
    # We want edge length ~1.15. Scale factor S = 1.15 / 2 = 0.575.
    # So R1_scaled = 1.902 * 0.575 = 1.094.
    
    S = 0.575
    R1 = np.sqrt(1 + phi**2) * S
    # R2: Distance to edge midpoint.
    # Midpoint of [0,1,phi] and [1,phi,0] is [0.5, (1+phi)/2, phi/2].
    # Norm: sqrt(0.25 + (1.5+phi)^2/4 + phi^2/4) = 0.5 * sqrt(1 + (1+phi)^2 + phi^2) = 0.5 * sqrt(1 + 1 + 2phi + phi^2 + phi^2) = 0.5 * sqrt(2 + 2phi + 2phi^2) = 0.5 * sqrt(2(1+phi+phi^2))
    # phi^2 = phi+1, so 1+phi+phi^2 = 1+phi+phi+1 = 2+2phi.
    # R2_unscaled = 0.5 * sqrt(2(2+2phi)) = sqrt(2+2phi) = sqrt(2*2.618) = sqrt(5.236) = 2.288? 
    # Wait, midpoint distance should be less than vertex distance.
    # Let's just calculate numerically.
    
    verts_scaled = verts * S
    points.extend([v for v in verts_scaled])
    
    # Shell 2: Edge midpoints (30)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, 2.0): # Edge length 2
                mid = (verts[i] + verts[j]) / 2.0
                edges.append(mid)
    # There should be 30 edges.
    # Scale them
    points.extend([e * S for e in edges])
    
    # Shell 3: Face centers (12)
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d_ij = np.linalg.norm(verts[i] - verts[j])
                d_jk = np.linalg.norm(verts[j] - verts[k])
                d_ik = np.linalg.norm(verts[i] - verts[k])
                if np.isclose(d_ij, 2.0) and np.isclose(d_jk, 2.0) and np.isclose(d_ik, 2.0):
                    fc = (verts[i] + verts[j] + verts[k]) / 3.0
                    faces.append(fc)
    # There should be 12 faces.
    points.extend([f * S for f in faces])
    
    all_atoms = np.array(points)
    
    # If we have more atoms than needed, remove the outermost ones (face centers, then edge midpoints)
    if len(all_atoms) > n_atoms:
        # Sort by distance from center
        dists = np.linalg.norm(all_atoms, axis=1)
        # We want to keep the core (smaller dists)
        # Sort indices by distance
        order = np.argsort(dists)
        all_atoms = all_atoms[order[:n_atoms]]
        
    elif len(all_atoms) < n_atoms:
        # If we need more than 55 (not in our range 38-60, but just in case)
        # Or if we need more than 13 but less than 55, the above logic handles it by keeping core.
        # If we need exactly 55, it's fine.
        # If we need 13, it keeps center + 12 verts.
        pass
        
    return all_atoms

def _basin_hopping(positions, rng, max_iter, step_size, time_limit):
    """
    Basin hopping with Metropolis acceptance.
    """
    start_time = time.time()
    current = positions.copy()
    current_energy = _dzugutov_energy(current)
    
    best_energy = current_energy
    best_pos = current.copy()
    
    T = 1.0
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        if _hard_remaining() < 5.0:
            break
            
        # Perturb
        perturbed = current + rng.normal(0, step_size, current.shape)
        perturbed = _enforce_min_distance(perturbed, 0.5)
        
        # Relax
        try:
            e_new, relaxed = relax(perturbed, 'dzugutov')
        except Exception:
            continue
            
        if e_new is None or not np.isfinite(e_new):
            continue
            
        # Metropolis acceptance
        delta_e = e_new - current_energy
        if delta_e < 0:
            accept = True
        else:
            accept = rng.random() < np.exp(-delta_e / T)
            
        if accept:
            current = relaxed
            current_energy = e_new
            if e_new < best_energy:
                best_energy = e_new
                best_pos = relaxed.copy()
                
        T *= 0.95
        step_size *= 0.99
        
    return best_energy, best_pos

def _worker_bh(args):
    """Worker for parallel basin hopping."""
    positions, seed, time_budget = args
    rng = np.random.default_rng(seed)
    try:
        # Initial relax
        e0, pos0 = relax(positions, 'dzugutov')
        if e0 is None:
            e0 = _dzugutov_energy(positions)
            pos0 = positions
            
        # Run BH
        best_e, best_p = _basin_hopping(pos0, rng, max_iter=100, step_size=0.4, time_limit=time_budget)
        return best_e, best_p
    except Exception:
        return np.inf, None

def _process_instance(size, potential, parameter, time_budget):
    """Process a single instance."""
    if _hard_remaining() < 10.0:
        # Not enough time, return a random valid structure
        radius = 1.1 * size ** (1.0 / 3.0)
        return np.random.uniform(-radius, radius, (size, 3))
        
    # Generate Mackay icosahedron seed
    seed_pos = _generate_mackay_icosahedron(size)
    
    # Create multiple diverse starting points by perturbing the seed
    # We will run parallel BH trajectories
    n_threads = 4
    time_per_thread = time_budget / n_threads
    
    seeds = []
    rng_master = np.random.default_rng(size)
    
    for i in range(n_threads):
        # Perturb the seed differently for each thread
        perturbation = rng_master.normal(0, 0.2, seed_pos.shape)
        perturbed_seed = seed_pos + perturbation
        perturbed_seed = _enforce_min_distance(perturbed_seed, 0.5)
        seeds.append((perturbed_seed, size * 100 + i, time_per_thread))
        
    best_energy = np.inf
    best_positions = None
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(_worker_bh, s) for s in seeds]
        for future in as_completed(futures):
            if _hard_remaining() < 5.0:
                break
            try:
                energy, positions = future.result()
                if positions is not None and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
                
    # Final Polish: L-BFGS on the best candidate
    if best_positions is not None and _hard_remaining() > 5.0:
        try:
            def objective(x):
                pos = x.reshape(-1, 3)
                e, g = _dzugutov_energy_gradient_vec(pos)
                return e, g.flatten()
                
            res = minimize(objective, best_positions.flatten(), method='L-BFGS-B', 
                           options={'maxiter': 20, 'maxfun': 50, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_energy:
                best_energy = res.fun
                best_positions = res.x.reshape(-1, 3)
                # Enforce min distance after optimization
                best_positions = _enforce_min_distance(best_positions, 0.5)
                # Re-evaluate energy after enforcing distance
                best_energy = _dzugutov_energy(best_positions)
        except Exception:
            pass
            
    if best_positions is None:
        best_positions = seed_pos
        
    return best_positions

def construct(instance):
    """Find low-energy Dzugutov cluster."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Check if we already have a result
    if size in _INSTANCE_RESULTS:
        return _INSTANCE_RESULTS[size]
        
    # Calculate time budget
    # Simple strategy: divide remaining time by number of remaining instances
    remaining_instances = [s for s in _INSTANCE_ORDER if s >= size]
    if not remaining_instances:
        remaining_instances = [size]
        
    time_per_instance = _remaining() / max(1, len(remaining_instances))
    # Cap time per instance to ensure we don't spend too long on one
    time_per_instance = min(time_per_instance, 45.0)
    
    # Process this instance
    result = _process_instance(size, potential, parameter, time_per_instance)
    
    # Cache the result
    _INSTANCE_RESULTS[size] = result
    
    return result
# EVOLVE-BLOCK-END

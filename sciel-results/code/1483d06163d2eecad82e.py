"""
Rewritten program for Dzugutov cluster global minima.
Uses a portfolio of structural heuristics (FCC, BCC, Icosahedral, Random)
relaxed in parallel with perturbation restarts and strict time management.
"""
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_HARD_LIMIT = 330.0
_SAFETY_MARGIN = 15.0  # Stop accepting new work 15s before hard limit

def _get_elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME

def _remaining_time():
    return _HARD_LIMIT - _SAFETY_MARGIN - _get_elapsed()

def _generate_fcc_chunk(n, rng):
    """Generate an FCC lattice chunk roughly spherical."""
    # Estimate radius for n atoms in FCC-like packing
    # FCC packing fraction ~ 0.74, atom diameter ~ 2.0
    # Volume per atom ~ 1/(0.74/8) ~ 10.8
    R = (3.0 * n / (4.0 * np.pi) * 10.8) ** (1.0 / 3.0) * 1.15
    a = 2.0  # Lattice constant (equilibrium distance)
    
    # Generate FCC points: (0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0) per unit cell
    points = []
    # Use a grid large enough to contain R
    extent = int(np.ceil(R / a)) + 1
    
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                # FCC basis
                basis = np.array([
                    [0, 0, 0],
                    [0, 0.5, 0.5],
                    [0.5, 0, 0.5],
                    [0.5, 0.5, 0]
                ]) * a
                for b in basis:
                    p = np.array([i * a, j * a, k * a]) + b
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    points = np.array(points)
    
    # Trim or pad to n
    if len(points) > n:
        # Sort by distance from center and take closest n
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        # Add random points near the surface
        while len(points) < n:
            p = rng.normal(0, R * 0.3, 3)
            p = p / np.linalg.norm(p) * R * 0.95
            points = np.vstack([points, p.reshape(1, 3)])
    
    points -= np.mean(points, axis=0)
    return points

def _generate_bcc_chunk(n, rng):
    """Generate a BCC lattice chunk roughly spherical."""
    R = (3.0 * n / (4.0 * np.pi) * 10.0) ** (1.0 / 3.0) * 1.15
    a = 2.0
    
    points = []
    extent = int(np.ceil(R / a)) + 1
    
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                # BCC basis: corner + body center
                p_corner = np.array([i * a, j * a, k * a])
                p_center = p_corner + np.array([0.5, 0.5, 0.5]) * a
                if np.linalg.norm(p_corner) <= R:
                    points.append(p_corner)
                if np.linalg.norm(p_center) <= R:
                    points.append(p_center)
    
    points = np.array(points)
    
    if len(points) > n:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        while len(points) < n:
            p = rng.normal(0, R * 0.3, 3)
            p = p / np.linalg.norm(p) * R * 0.95
            points = np.vstack([points, p.reshape(1, 3)])
    
    points -= np.mean(points, axis=0)
    return points

def _generate_mackay_icosahedron(n, rng):
    """
    Generate a Mackay icosahedral cluster.
    Layers: 1, 12, 42, 120, 300...
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 atoms at vertices
    r1 = 1.95
    shell1 = verts * r1
    points.extend(shell1)
    
    if n <= len(points):
        return np.array(points[:n])

    # Shell 2: 42 atoms (midpoints of edges + face centers approximation)
    # For simplicity, place atoms on a sphere of radius r2
    r2 = r1 * 1.65
    # Generate 42 points on sphere using fibonacci sphere
    for i in range(42):
        z = 1 - 2 * (i + 0.5) / 42
        r = np.sqrt(1 - z * z)
        theta = np.pi * (1 + np.sqrt(5)) * i
        p = r2 * np.array([r * np.cos(theta), r * np.sin(theta), z])
        points.append(p)
    
    if n <= len(points):
        return np.array(points[:n])

    # Shell 3: fill remaining with dense random in shell
    target_n = n
    current_n = len(points)
    
    if target_n > current_n:
        r_outer = (3.0 * target_n / (4.0 * np.pi * 9.5)) ** (1.0/3.0) * 1.3
        r_outer = max(r_outer, r2 * 1.3)
        
        attempts = 0
        max_attempts = 20000
        while len(points) < target_n and attempts < max_attempts:
            u = rng.uniform(0, 1)
            r = r2 * 1.05 + (r_outer - r2 * 1.05) * u**(1.0/3.0)
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * r
            
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) < 1.75:
                    attempts += 1
                    continue
            points.append(p)
            attempts += 1
        
        while len(points) < target_n:
            p = rng.uniform(-r_outer, r_outer, 3)
            points.append(p)
            
    points = np.array(points[:target_n])
    points -= np.mean(points, axis=0)
    return points

def _generate_dense_random(n, rng, min_dist=1.8):
    """Generate a dense random cluster using rejection sampling."""
    r_est = (3.0 * n / (4.0 * np.pi * 8.5)) ** (1.0/3.0) * 1.4
    r_est = max(r_est, 3.5)
    
    points = []
    attempts = 0
    max_attempts = 8000 * n
    
    points.append(np.array([0.0, 0.0, 0.0]))
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-r_est, r_est, 3)
        if np.linalg.norm(p) > r_est:
            attempts += 1
            continue
            
        if len(points) > 0:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) < min_dist:
                attempts += 1
                continue
                
        points.append(p)
        attempts += 1
        
    while len(points) < n:
        p = rng.uniform(-r_est, r_est, 3)
        points.append(p)
        
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _perturb(positions, rng, scale=0.15):
    """Add small random perturbation to escape local minima."""
    perturbed = positions + rng.normal(0, scale, positions.shape)
    return perturbed

def _relax_worker(args):
    """Worker for parallel relaxation."""
    positions, potential, parameter, timeout = args
    try:
        energy, relaxed_positions = relax(positions, potential, parameter)
        return energy, relaxed_positions
    except Exception:
        return float('inf'), positions

def _polish_worker(args):
    """Polish a relaxed structure with small perturbations and re-relax."""
    positions, potential, parameter, timeout, rng_seed = args
    try:
        rng = np.random.default_rng(rng_seed)
        best_e, best_pos = float('inf'), positions.copy()
        
        # Try a few perturbations
        for _ in range(3):
            if _remaining_time() < 3.0:
                break
            perturbed = _perturb(best_pos, rng, scale=0.12)
            try:
                e, pos = relax(perturbed, potential, parameter)
                if e < best_e:
                    best_e, best_pos = e, pos
            except Exception:
                pass
        
        return best_e, best_pos
    except Exception:
        return float('inf'), positions

def construct(instance):
    """Find the lowest energy structure for the given cluster size."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Check if we have time
    if _remaining_time() < 8.0:
        rng = np.random.default_rng(size)
        r = 3.5
        points = rng.uniform(-r, r, (size, 3))
        return points

    rng = np.random.default_rng(size)
    initial_clusters = []
    
    # 1. FCC
    try:
        fcc = _generate_fcc_chunk(size, rng)
        initial_clusters.append(fcc)
    except Exception:
        pass
        
    # 2. BCC
    try:
        bcc = _generate_bcc_chunk(size, rng)
        initial_clusters.append(bcc)
    except Exception:
        pass
        
    # 3. Icosahedral
    try:
        ico = _generate_mackay_icosahedron(size, rng)
        initial_clusters.append(ico)
    except Exception:
        pass
        
    # 4. Dense Random Packings (2 variants)
    for i in range(2):
        try:
            min_d = 1.75 + 0.1 * i
            rand = _generate_dense_random(size, rng, min_dist=min_d)
            initial_clusters.append(rand)
        except Exception:
            pass
            
    if not initial_clusters:
        rand = _generate_dense_random(size, rng)
        initial_clusters.append(rand)
        
    # Phase 1: Relax all initial clusters in parallel
    args_list = []
    for cluster in initial_clusters:
        if cluster.shape != (size, 3):
            continue
        args_list.append((cluster, potential, parameter, _remaining_time()))
        
    if not args_list:
        return initial_clusters[0]
        
    best_energy = float('inf')
    best_positions = initial_clusters[0]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_relax_worker, args) for args in args_list]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                energy, positions = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
            
            if _remaining_time() < 10.0:
                break
                
    # Phase 2: Polish the best structure with perturbations
    if _remaining_time() > 15.0:
        try:
            polish_args = (best_positions, potential, parameter, _remaining_time(), size * 7 + 1)
            e, pos = _polish_worker(polish_args)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except Exception:
            pass
            
    # Phase 3: If time remains, try perturbing the best and re-relaxing
    if _remaining_time() > 20.0:
        try:
            rng2 = np.random.default_rng(size * 13 + 2)
            for _ in range(2):
                if _remaining_time() < 12.0:
                    break
                perturbed = _perturb(best_positions, rng2, scale=0.2)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_relax_worker, (perturbed, potential, parameter, _remaining_time()))
                    try:
                        e, pos = fut.result(timeout=_remaining_time())
                        if e < best_energy:
                            best_energy = e
                            best_positions = pos
                    except Exception:
                        pass
        except Exception:
            pass
            
    return best_positions

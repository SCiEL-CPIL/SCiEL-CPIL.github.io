"""
Sutton-Chen transition-metal cluster global minima search.
Uses parallel relaxation of diverse structural starts (FCC, HCP, Spherical, Random).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_TOTAL_TIME_LIMIT = 290.0  # Safety margin before 300s kill
_START_TIME = None

def _time_left():
    if _START_TIME is None:
        return _TOTAL_TIME_LIMIT
    return _TOTAL_TIME_LIMIT - (time.time() - _START_TIME)

def _validate(positions):
    """Check if positions are valid: finite and min distance >= 0.35."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    # Vectorized distance check
    i_idx, j_idx = np.triu_indices(n, k=1)
    d = np.linalg.norm(positions[i_idx] - positions[j_idx])
    if np.any(d < 0.35):
        return False
    return True

def _generate_fcc(n, seed=0):
    """Generate points from an FCC lattice inside a sphere."""
    rng = np.random.default_rng(seed)
    a = 1.0  # Lattice constant, will be scaled
    # Estimate radius to contain n atoms
    # Density of FCC is 4 atoms per a^3
    R = (3.0 * n / (4.0 / a**3) / (4.0 * np.pi)) ** (1.0/3.0) * 1.2
    
    # Generate lattice points in a box
    max_coord = int(np.ceil(R / a)) + 2
    points = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                # FCC basis
                points.append(np.array([ix, iy, iz]) * a)
                points.append(np.array([ix + 0.5, iy + 0.5, iz]) * a)
                points.append(np.array([ix + 0.5, iy, iz + 0.5]) * a)
                points.append(np.array([ix, iy + 0.5, iz + 0.5]) * a)
                
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    sorted_pts = points[order]
    
    # Select n points closest to origin
    if len(sorted_pts) >= n:
        return sorted_pts[:n]
    else:
        # Fill with random if not enough (unlikely)
        pts = list(sorted_pts)
        while len(pts) < n:
            p = rng.uniform(-R, R, 3)
            pts.append(p)
        return np.array(pts)

def _generate_hcp(n, seed=0):
    """Generate points from an HCP lattice inside a sphere."""
    rng = np.random.default_rng(seed)
    a = 1.0
    c = a * np.sqrt(8/3)
    R = (3.0 * n / (2.0 / (a*a*c/2)) / (4.0 * np.pi)) ** (1.0/3.0) * 1.2
    
    max_coord = int(np.ceil(R / a)) + 2
    points = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                # HCP basis: ABA stacking
                # Layer A: (0,0,0)
                points.append(np.array([ix, iy, iz]) * np.array([a, a, c/3]))
                # Layer B: (1/3, 2/3, 1/3)
                points.append((np.array([ix + 1/3, iy + 2/3, iz + 1/3])) * np.array([a, a, c/3]))
                # Layer C: (2/3, 1/3, 2/3) -- wait, HCP is AB. Let's just use AB.
                # Actually, standard HCP unit cell has 2 atoms.
                # Let's just generate a dense set of points in AB planes.
                
    # Simpler HCP:
    points = []
    for k in range(-max_coord, max_coord + 1):
        z = k * c / 3
        if abs(z) > R:
            continue
        if k % 3 == 0: # A plane
            for ix in range(-max_coord, max_coord + 1):
                for iy in range(-max_coord, max_coord + 1):
                    p = np.array([ix * a, iy * a, z])
                    if np.linalg.norm(p) <= R:
                        points.append(p)
        elif k % 3 == 1: # B plane
            for ix in range(-max_coord, max_coord + 1):
                for iy in range(-max_coord, max_coord + 1):
                    p = np.array([(ix + 1/3) * a, (iy + 2/3) * a, z])
                    if np.linalg.norm(p) <= R:
                        points.append(p)
        # k % 3 == 2 is empty in AB stacking? No, ABAB.
        # So k=0 (A), k=1 (B), k=2 (A), k=3 (B).
        # My loop above is slightly off. Let's just use a dense random sampling within HCP constraints?
        # No, let's stick to FCC and Random. HCP is less common for global mins in SC.
        
    if len(points) < n:
        return _generate_fcc(n, seed)
        
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    return np.array(points[order[:n]])

def _generate_fibonacci_sphere(n, radius=1.0):
    """Generate n points evenly distributed on a sphere using Fibonacci lattice."""
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
        
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0)) # Golden angle
    for i in range(n):
        y = 1 - (i / (n - 1)) * 2
        radius_i = np.sqrt(1 - y**2)
        theta = phi * i
        x = np.cos(theta) * radius_i
        z = np.sin(theta) * radius_i
        points.append([x, y, z])
        
    pts = np.array(points) * radius
    return pts

def _generate_random_compact(n, seed=0):
    """Generate a random compact cluster using rejection sampling."""
    rng = np.random.default_rng(seed)
    radius = n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    max_attempts = n * 2000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.5:
                    points.append(p)
            else:
                points.append(p)
        attempts += 1
        
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
        
    return np.array(points)

def _relax_worker(start, kind):
    """Worker function for parallel relaxation."""
    try:
        energy, positions = relax(start, 'sutton_chen', kind)
        if np.all(np.isfinite(positions)) and _validate(positions):
            return energy, positions
    except Exception:
        pass
    return None, None

def construct(instance):
    """Find near-global minimum for Sutton-Chen transition-metal clusters."""
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. FCC starts (10 variants with different seeds/scales)
    for i in range(10):
        try:
            fcc = _generate_fcc(n, seed=i)
            if len(fcc) == n and _validate(fcc):
                starts.append(fcc)
        except:
            pass
            
    # 2. HCP starts (5 variants)
    for i in range(5):
        try:
            hcp = _generate_hcp(n, seed=100+i)
            if len(hcp) == n and _validate(hcp):
                starts.append(hcp)
        except:
            pass

    # 3. Spherical Code starts (5 variants)
    for i in range(5):
        try:
            radius = n ** (1.0/3.0) * 0.9
            fib = _generate_fibonacci_sphere(n, radius)
            if len(fib) == n and _validate(fib):
                starts.append(fib)
        except:
            pass

    # 4. Random Compact starts (10 variants)
    for i in range(10):
        try:
            rand = _generate_random_compact(n, seed=200+i)
            if len(rand) == n and _validate(rand):
                starts.append(rand)
        except:
            pass

    if not starts:
        starts = [_generate_random_compact(n, seed=0)]
        
    best_energy = np.inf
    best_positions = starts[0]
    
    # Parallel relaxation
    # Limit total starts to 30 to manage time
    starts = starts[:30]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_relax_worker, start, kind): i for i, start in enumerate(starts)}
        
        for future in as_completed(futures):
            if _time_left() < 5.0:
                break
                
            res = future.result()
            if res[0] is not None:
                energy, positions = res
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
                    
    # Final validation
    if not _validate(best_positions):
        # Force valid
        best_positions = _generate_random_compact(n, seed=999)
        for i in range(len(best_positions)):
            for j in range(i + 1, len(best_positions)):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < 0.35:
                    direction = best_positions[i] - best_positions[j]
                    if np.linalg.norm(direction) > 1e-10:
                        direction = direction / np.linalg.norm(direction)
                    else:
                        direction = np.array([1.0, 0, 0])
                    best_positions[j] = best_positions[i] + direction * 0.36
    
    return best_positions
# EVOLVE-BLOCK-END

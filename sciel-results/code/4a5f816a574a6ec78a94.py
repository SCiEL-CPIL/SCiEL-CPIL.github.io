"""
Sutton-Chen transition-metal cluster global minima search.
Uses structure prediction with icosahedral/FCC cores, basin-hopping optimization,
and a final high-precision polish pass.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_TOTAL_TIME_LIMIT = 295.0
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
    i_idx, j_idx = np.triu_indices(n, k=1)
    d = np.linalg.norm(positions[i_idx] - positions[j_idx])
    if np.any(d < 0.35):
        return False
    return True

def _get_icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin, normalized to unit radius."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _generate_icosahedral_cluster(n, seed=0):
    """
    Generate an icosahedral cluster with correct relative shell radii.
    Shell 0: 1 atom at origin.
    Shell 1: 12 atoms at radius R1 = 1.0.
    Shell 2: 30 atoms (edge midpoints) at radius R2.
    Shell 3: 20 atoms (face centers) at radius R3.
    
    Total atoms available: 1 + 12 + 30 + 20 = 63.
    If n > 63, we add random atoms on the surface of Shell 3.
    """
    rng = np.random.default_rng(seed)
    
    ico_verts = _get_icosahedron_vertices()
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 vertices
    R1 = 1.0
    shell1 = ico_verts * R1
    for v in shell1:
        points.append(v)
        
    # Calculate R2 (edge midpoints)
    v0 = ico_verts[0]
    v1 = ico_verts[1]
    edge_len = np.linalg.norm(v0 - v1)
    R2 = np.sqrt(R1**2 - (edge_len/2.0)**2)
    
    # Find edges
    edges = set()
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if np.isclose(d, edge_len, atol=1e-6):
                edges.add((i, j))
                
    shell2 = []
    for (i, j) in edges:
        mid = (ico_verts[i] + ico_verts[j]) / 2.0
        mid_dir = mid / np.linalg.norm(mid)
        shell2.append(mid_dir * R2)
    for v in shell2:
        points.append(v)
        
    # Calculate R3 (face centers)
    faces = set()
    for i in range(12):
        for j in range(i+1, 12):
            if (i, j) not in edges: continue
            for k in range(j+1, 12):
                if (j, k) not in edges: continue
                if (i, k) not in edges: continue
                faces.add(tuple(sorted([i, j, k])))
                
    shell3 = []
    for (i, j, k) in faces:
        center = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3.0
        center_dir = center / np.linalg.norm(center)
        R3 = np.linalg.norm(center)
        shell3.append(center_dir * R3)
    for v in shell3:
        points.append(v)
        
    points = np.array(points)
    
    if len(points) >= n:
        return points[:n]
    else:
        surface_dirs = [v / np.linalg.norm(v) for v in points if np.linalg.norm(v) > 0.5]
        remaining = n - len(points)
        for _ in range(remaining):
            if surface_dirs:
                idx = rng.integers(0, len(surface_dirs))
                dir_vec = surface_dirs[idx]
            else:
                dir_vec = rng.normal(0, 1, 3)
                dir_vec /= np.linalg.norm(dir_vec)
            
            r = R3 * (1.0 + rng.uniform(0.1, 0.3))
            new_p = dir_vec * r
            points = np.vstack([points, [new_p]])
            
        return points[:n]

def _generate_fcc(n, seed=0):
    """Generate points from an FCC lattice inside a sphere."""
    rng = np.random.default_rng(seed)
    a = 1.0
    R = (3.0 * n / (4.0 / a**3) / (4.0 * np.pi)) ** (1.0/3.0) * 1.2
    
    max_coord = int(np.ceil(R / a)) + 2
    points = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                points.append(np.array([ix, iy, iz]) * a)
                points.append(np.array([ix + 0.5, iy + 0.5, iz]) * a)
                points.append(np.array([ix + 0.5, iy, iz + 0.5]) * a)
                points.append(np.array([ix, iy + 0.5, iz + 0.5]) * a)
                
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    sorted_pts = points[order]
    
    if len(sorted_pts) >= n:
        return sorted_pts[:n]
    else:
        pts = list(sorted_pts)
        while len(pts) < n:
            p = rng.uniform(-R, R, 3)
            pts.append(p)
        return np.array(pts)

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

def _perturb(positions, magnitude=0.2, seed=0):
    """Perturb positions by adding random noise."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, magnitude, positions.shape)
    return positions + noise

def _relax_worker(start, kind, steps=1, seed=0):
    """
    Worker function: performs 'steps' of relax-perturb-relax.
    Returns best energy and positions found in the trajectory.
    """
    try:
        current_pos = start.copy()
        best_energy, best_pos = np.inf, None
        
        for step in range(steps):
            if _time_left() < 5.0:
                break
                
            energy, pos = relax(current_pos, 'sutton_chen', kind)
            if np.all(np.isfinite(pos)) and _validate(pos):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos.copy()
                
                if step < steps - 1:
                    current_pos = _perturb(pos, magnitude=0.15, seed=seed + step)
            else:
                break
                
        if best_pos is None:
            if _validate(start):
                e, _ = sutton_chen_energy_gradient(start, kind)
                return e, start
            return None, None
            
        return best_energy, best_pos
    except Exception:
        return None, None

def construct(instance):
    """Find near-global minimum for Sutton-Chen transition-metal clusters."""
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    
    starts = []
    
    for i in range(10):
        try:
            ico = _generate_icosahedral_cluster(n, seed=i)
            if len(ico) == n and _validate(ico):
                starts.append(ico)
        except:
            pass
            
    for i in range(5):
        try:
            fcc = _generate_fcc(n, seed=100+i)
            if len(fcc) == n and _validate(fcc):
                starts.append(fcc)
        except:
            pass

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
    
    starts = starts[:30]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_relax_worker, start, kind, steps=2, seed=i): i for i, start in enumerate(starts)}
        
        for future in as_completed(futures):
            if _time_left() < 10.0:
                break
                
            res = future.result()
            if res[0] is not None:
                energy, positions = res
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
                    
    if not _validate(best_positions):
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

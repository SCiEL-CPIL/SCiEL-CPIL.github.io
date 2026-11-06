"""Dzugutov cluster global minima: Precise Mackay icosahedral seeds + Parallel Basin Hopping."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import threading
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient

_START = time.time()
TIME_LIMIT = 295.0  # Safety margin under 330s kill

# Dzugutov parameters
_A, _B = 5.82, 1.28
_a, _b = 1.87, 1.94
_c, _d = 1.10, 0.27
_m = 16
_MIN_SEP = 0.5

def _remaining():
    return TIME_LIMIT - (time.time() - _START)

def _energy(pos):
    """Vectorised Dzugutov energy (flat array, n*3)."""
    p = np.asarray(pos, dtype=float)
    n = p.size // 3
    p = p.reshape(n, 3)
    diff = p[:, None, :] - p[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    r_ij = r[iu]
    
    energy = 0.0
    mask1 = r_ij < _a
    if np.any(mask1):
        r1 = r_ij[mask1]
        # Avoid division by zero or negative if r < a is very small, but min_sep is 0.5
        energy += np.sum(_A * (r1 ** (-_m) - _B) * np.exp(_c / (r1 - _a)))
    mask2 = (r_ij >= _a) & (r_ij < _b)
    if np.any(mask2):
        r2 = r_ij[mask2]
        energy += np.sum(_B * np.exp(_d / (r2 - _b)))
    return energy

def _grad_fn(pos):
    """Gradient via the provided helper, reshaped to flat."""
    p = np.asarray(pos, dtype=float)
    n = p.size // 3
    p = p.reshape(n, 3)
    _, grad = dzugutov_energy_gradient(p)
    return grad.flatten()

def _min_sep_ok(pos, min_sep=_MIN_SEP):
    """Check minimum separation constraint."""
    n = pos.shape[0]
    if n < 2:
        return True
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    return np.min(r[iu]) >= min_sep

def _enforce_min_sep(pos, min_sep=_MIN_SEP, max_iter=100):
    """Enforce minimum separation by pushing overlapping atoms apart."""
    pos = pos.copy()
    n = pos.shape[0]
    for _ in range(max_iter):
        if _min_sep_ok(pos, min_sep):
            break
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_sep:
                    if d < 1e-10:
                        direction = np.array([1.0, 0.0, 0.0])
                    else:
                        direction = (pos[i] - pos[j]) / d
                    push = (min_sep - d) / 2 + 0.01
                    pos[i] += direction * push
                    pos[j] -= direction * push
    return pos

def _get_icosahedron_data():
    """
    Generate vertices, edge midpoints, and face centers of a regular icosahedron.
    Returns:
    ico_verts: (12, 3)
    edge_mids: (30, 3)
    face_centers: (20, 3)
    """
    phi = (1 + np.sqrt(5)) / 2
    # Standard icosahedron vertices
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit radius
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    # Find edges (distance between vertices is ~ 2*sin(pi/5) = ~1.175 for unit radius? 
    # Actually, edge length for unit circumradius icosahedron is 2/sqrt(phi^2+1) * sqrt(5-1)/2? 
    # Let's just compute distances and find the unique minimum distance > 0.
    dists = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            dists.append(d)
    min_dist = np.min(dists)
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.isclose(np.linalg.norm(ico_verts[i] - ico_verts[j]), min_dist, rtol=1e-5):
                edges.append((i, j))
                
    edge_mids = np.array([(ico_verts[i] + ico_verts[j]) / 2 for i, j in edges])
    edge_mids /= np.linalg.norm(edge_mids[0])
    
    # Find faces (triangles of edges)
    faces = []
    # For each edge, find the third vertex that forms a triangle
    # An easier way: iterate all triples and check if all 3 pairwise distances are min_dist
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                dij = np.linalg.norm(ico_verts[i] - ico_verts[j])
                djk = np.linalg.norm(ico_verts[j] - ico_verts[k])
                dik = np.linalg.norm(ico_verts[i] - ico_verts[k])
                if np.isclose(dij, min_dist, rtol=1e-5) and np.isclose(djk, min_dist, rtol=1e-5) and np.isclose(dik, min_dist, rtol=1e-5):
                    faces.append((i, j, k))
                    
    face_centers = np.array([(ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3 for i, j, k in faces])
    face_centers /= np.linalg.norm(face_centers[0])
    
    return ico_verts, edge_mids, face_centers

def _generate_mackay_seed(n, rng, twist=0.0):
    """
    Generate a Mackay icosahedral cluster with n atoms.
    Shells: 
    0: 1 atom (center)
    1: 12 atoms (icosahedron vertices)
    2: 30 atoms (edge midpoints)
    3: 20 atoms (face centers)
    4: 60 atoms (next shell) - we might not need this for n<=60
    
    For n > 63, we would need more, but max n is 60.
    So we use shells 0, 1, 2, 3.
    """
    ico_verts, edge_mids, face_centers = _get_icosahedron_data()
    
    # Radii for shells. 
    # In a perfect Mackay icosahedron, the radii are proportional to:
    # Shell 0: 0
    # Shell 1: R1
    # Shell 2: R2 = R1 * (1/2) * sqrt(5+2*sqrt(5)) / (1/2 * sqrt(5-2*sqrt(5)))? 
    # Actually, for a regular icosahedron with circumradius R:
    # Edge length a = R * sqrt(10 - 2*sqrt(5)) / 2? No.
    # Let's use the known ratio. 
    # If ico_verts are unit radius, edge_mids are at radius 1/sqrt(5+2*sqrt(5)) * something?
    # Let's just normalize them to have the same norm as ico_verts? No, that's not Mackay.
    # Mackay icosahedron is a packing of 13 icosahedra? No, it's a nested icosahedron.
    # The standard Mackay icosahedron has atoms at:
    # 1 center
    # 12 vertices of icosahedron (radius R)
    # 30 midpoints of edges (radius R * (1/2) * sqrt(5+2*sqrt(5)) / (1/2 * sqrt(5-2*sqrt(5)))? 
    # Actually, the distance from center to edge midpoint is R * (1/2) * (1/sin(pi/10))? 
    # Let's just use the geometric midpoints and face centers calculated above, 
    # and scale them to match the expected Mackay radii if needed, or just use them as is 
    # and let relaxation fix the radii. 
    # Using the unit sphere points for shells 1, 2, 3 and scaling them appropriately is a good start.
    # Let's assume the "ideal" Mackay radii are:
    # r0 = 0
    # r1 = 1.0
    # r2 = 1.0 * (1/2) * (1 + 1/sqrt(5)) / (1/2 * sqrt(5-2*sqrt(5)))? 
    # A simpler approach: The Mackay icosahedron is often generated by taking the 12 vertices, 
    # 30 edge midpoints, and 20 face centers of an icosahedron and scaling them such that 
    # the nearest neighbor distances are consistent.
    
    # Let's just place them at radii 0, 1.0, 1.5, 2.0 initially and let relaxation adjust.
    # Or better, use the exact geometric positions on the sphere and scale.
    
    points = []
    
    # Shell 0
    if n > 0:
        points.append(np.array([0.0, 0.0, 0.0]))
        n -= 1
    
    # Shell 1: 12 vertices
    if n > 0:
        count = min(12, n)
        # Apply small twist to break symmetry slightly for diversity? 
        # No, keep it symmetric for the seed.
        for i in range(count):
            p = ico_verts[i] * 1.2 # Scale factor
            points.append(p)
        n -= count
    
    # Shell 2: 30 edge midpoints
    if n > 0:
        count = min(30, n)
        for i in range(count):
            p = edge_mids[i] * 1.9 # Scale factor
            points.append(p)
        n -= count
        
    # Shell 3: 20 face centers
    if n > 0:
        count = min(20, n)
        for i in range(count):
            p = face_centers[i] * 2.5 # Scale factor
            points.append(p)
        n -= count
        
    # If n is still > 0, we need a 4th shell. 
    # For n <= 60, we only need up to shell 3 (1+12+30+20 = 63).
    # So for n=60, we take 1+12+30+17 from shell 3.
    
    # If we ran out of shell 3, we would need to generate shell 4.
    # But max n is 60, so we are safe.
    
    # Add small random perturbation to break exact symmetry and help relaxation
    if len(points) > 0:
        points = np.array(points)
        points += rng.normal(0, 0.05, points.shape)
        
    return _enforce_min_sep(points)

def _relax_lbfgsb(pos_flat, maxiter=200):
    """Relax using L-BFGS-B."""
    try:
        result = minimize(
            _energy,
            pos_flat,
            jac=_grad_fn,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50, 'disp': False}
        )
        return result
    except Exception:
        return None

def _basin_hopping_worker(seed, n, rng, lock, best_shared):
    """Worker thread for basin hopping."""
    current_pos = seed.copy()
    current_e = _energy(current_pos.flatten())
    best_pos = current_pos.copy()
    best_e = current_e
    
    kick_scale = 0.2
    T = 0.3 # Temperature
    accept_count = 0
    total_count = 0
    
    while _remaining() > 2.0:
        # Kick
        kick = rng.normal(0, kick_scale, current_pos.shape)
        kicked = current_pos + kick
        kicked = _enforce_min_sep(kicked)
        
        # Relax
        result = _relax_lbfgsb(kicked.flatten(), maxiter=100)
        if result is None:
            continue
        
        new_e = result.fun
        new_pos = result.x.reshape(n, 3)
        
        if not _min_sep_ok(new_pos):
            continue
            
        total_count += 1
        accepted = False
        
        if new_e < current_e:
            accepted = True
        else:
            delta_e = new_e - current_e
            if delta_e < 0:
                accepted = True
            else:
                # Metropolis
                prob = np.exp(-delta_e / T)
                if rng.random() < prob:
                    accepted = True
        
        if accepted:
            accept_count += 1
            current_pos = new_pos
            current_e = new_e
            
            if new_e < best_e:
                best_e = new_e
                best_pos = new_pos
                with lock:
                    if best_e < best_shared[0]:
                        best_shared[0] = best_e
                        best_shared[1] = best_pos.copy()
        
        # Adapt kick scale
        if total_count > 10:
            accept_rate = accept_count / total_count
            if accept_rate < 0.3:
                kick_scale *= 1.2
            elif accept_rate > 0.7:
                kick_scale *= 0.8
            kick_scale = np.clip(kick_scale, 0.05, 1.0)
            
            # Reset counters
            if total_count > 50:
                accept_count = 0
                total_count = 0

    # Final relaxation
    if best_e < np.inf:
        final_res = _relax_lbfgsb(best_pos.flatten(), maxiter=200)
        if final_res is not None and final_res.fun < best_e:
            best_e = final_res.fun
            best_pos = final_res.x.reshape(n, 3)
            with lock:
                if best_e < best_shared[0]:
                    best_shared[0] = best_e
                    best_shared[1] = best_pos.copy()

def construct(instance):
    """Find low-energy Dzugutov cluster using parallel basin hopping from Mackay seeds."""
    size = instance["n"]
    
    # Generate diverse Mackay seeds
    rng_main = np.random.default_rng(42 + size)
    seeds = []
    
    # 4 different seeds for 4 threads
    for i in range(4):
        # Slightly different scales or twists for each seed to explore different parts of the basin
        # We can perturb the shell radii slightly
        seed = _generate_mackay_seed(size, rng_main)
        # Apply a small random rotation to break symmetry
        # Generate random rotation matrix
        q = rng_main.normal(0, 1, 4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        seed = seed @ R.T
        seeds.append(seed)
        
    # Relax all seeds first
    relaxed_seeds = []
    for seed in seeds:
        if _remaining() < 5.0:
            relaxed_seeds.append(seed)
            continue
        res = _relax_lbfgsb(seed.flatten(), maxiter=200)
        if res is not None:
            relaxed_seeds.append(res.x.reshape(size, 3))
        else:
            relaxed_seeds.append(seed)
            
    if not relaxed_seeds:
        return seeds[0]
        
    # Setup parallel basin hopping
    n_threads = 4
    best_shared = [np.inf, None]
    lock = threading.Lock()
    threads = []
    
    for i in range(n_threads):
        seed = relaxed_seeds[i]
        thread_rng = np.random.default_rng(1000 + i + size)
        
        t = threading.Thread(
            target=_basin_hopping_worker,
            args=(seed, size, thread_rng, lock, best_shared)
        )
        threads.append(t)
        t.start()
        
    # Wait for threads to finish or timeout
    for t in threads:
        t.join()
        
    if best_shared[1] is not None:
        return best_shared[1]
    else:
        # Fallback to best relaxed seed
        return relaxed_seeds[0]

# EVOLVE-BLOCK-END

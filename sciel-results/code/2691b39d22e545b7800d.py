import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None

def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    # 330s hard kill. Use 295s as our working deadline to be safe.
    return max(0.0, 295.0 - (time.time() - _START_TIME))

def _energy(positions):
    """
    Compute the Dzugutov energy.
    V(r) = A(r^-m - B)exp(c/(r - a)) for r < a
           B exp(d/(r - b)) for r < b
    A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16
    """
    A = 5.82
    B = 1.28
    a = 1.87
    b = 1.94
    c = 1.10
    d = 0.27
    m = 16
    
    n = len(positions)
    if n < 2:
        return 0.0
    
    # Compute all pairwise distances
    # positions shape (n, 3)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    dist = np.sqrt(dist_sq)
    
    # Mask for upper triangle (i < j)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dists = dist[mask]
    
    if len(dists) == 0:
        return 0.0
    
    # Avoid division by zero or negative r-a, r-b
    # The potential is defined for r < a and r < b.
    # If r is very small, r-a is negative, exp(c/(r-a)) goes to 0.
    # If r is close to a from below, r-a -> 0-, exp -> 0.
    # If r > a, the first term is 0.
    # If r > b, the second term is 0.
    
    energy = 0.0
    
    # Term 1: r < a
    mask1 = dists < a
    if np.any(mask1):
        r1 = dists[mask1]
        # Handle r close to 0 to avoid inf in r^-m
        # r^-m becomes huge. But exp(c/(r-a)) becomes 0 fast.
        # Let's just clip r to a small positive number to avoid inf/nan
        r1_safe = np.maximum(r1, 1e-6)
        term1 = A * (r1_safe ** (-m) - B) * np.exp(c / (r1_safe - a))
        energy += np.sum(term1)
        
    # Term 2: r < b (and r >= a is handled by being 0 in term 1, but term 2 applies for all r < b)
    # Wait, the definition is:
    # V(r) = A(r^-m - B)exp(c/(r - a)) [r < a] + B exp(d/(r - b)) [r < b]
    # This implies for a <= r < b, only the second term is active.
    # For r < a, both terms are active?
    # "V(r) = A(...) [r < a] + B(...) [r < b]"
    # Usually this means sum of the two terms where their conditions are met.
    # So for r < a: Term1 + Term2
    # For a <= r < b: Term2 only
    # For r >= b: 0
    
    # Let's re-evaluate.
    # My code above added Term1 for r < a.
    # Now add Term2 for r < b.
    mask2 = dists < b
    if np.any(mask2):
        r2 = dists[mask2]
        r2_safe = np.maximum(r2, 1e-6)
        term2 = B * np.exp(d / (r2_safe - b))
        energy += np.sum(term2)
        
    return energy

def _generate_mackay_icosahedron(n, rng):
    """
    Generate the first n atoms of a Mackay icosahedron.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 vertices of an icosahedron
    verts = np.array([
        [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
    ], dtype=float)
    # Normalize to radius R1 = 1.0
    norm = np.linalg.norm(verts[0])
    verts = verts / norm * 1.0
    
    points = [np.zeros(3)]
    
    # Shell 1: 12 vertices
    for v in verts:
        if len(points) >= n:
            break
        points.append(v)
        
    if n <= 13:
        return np.array(points[:n])
        
    # Shell 2: 30 edge midpoints
    edge_mids = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            edge_mids.append((d, (verts[i] + verts[j]) / 2.0))
            
    edge_mids.sort(key=lambda x: x[0])
    for d, mid in edge_mids:
        if len(points) >= n:
            break
        points.append(mid)
        
    if n <= 43:
        return np.array(points[:n])
        
    # For n > 43, add random atoms on the surface
    points_arr = np.array(points[:43])
    if n > 43:
        extra_needed = n - 43
        for _ in range(extra_needed):
            theta = rng.random() * 2 * np.pi
            phi_ang = np.arccos(2 * rng.random() - 1)
            r = 1.0 + rng.random() * 0.5
            p = np.array([
                r * np.sin(phi_ang) * np.cos(theta),
                r * np.sin(phi_ang) * np.sin(theta),
                r * np.cos(phi_ang)
            ])
            points_arr = np.vstack([points_arr, p])
            
    points_arr = points_arr + rng.normal(scale=0.01, size=points_arr.shape)
    return points_arr[:n]

def _generate_polytetrahedral_seed(n, rng):
    """
    Generate a seed based on polytetrahedral packing.
    Uses Mackay icosahedron for n <= 43.
    For n > 43, uses 43-atom core + random surface.
    """
    return _generate_mackay_icosahedron(n, rng)

def _generate_bimodal(n, rng):
    """
    Generate a cluster from two icosahedra.
    """
    n1 = n // 2
    n2 = n - n1
    
    ico1 = _generate_mackay_icosahedron(n1, rng)
    ico2 = _generate_mackay_icosahedron(n2, rng)
    
    sep = 3.5
    ico1[:, 0] += sep / 2
    ico2[:, 0] -= sep / 2
    
    points = np.vstack([ico1, ico2])
    points = points + rng.normal(scale=0.02, size=points.shape)
    return points

def _generate_random(n, rng):
    """
    Generate a random compact cluster.
    """
    R = (3.0 * n / (4.0 * np.pi) * 10.0) ** (1.0/3.0) * 1.1
    points = []
    min_dist = 0.55
    max_attempts = n * 2000
    
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        r = R * (rng.random() ** (1.0/3.0))
        theta = rng.random() * 2 * np.pi
        phi_ang = np.arccos(2 * rng.random() - 1)
        p = np.array([
            r * np.sin(phi_ang) * np.cos(theta),
            r * np.sin(phi_ang) * np.sin(theta),
            r * np.cos(phi_ang)
        ])
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-R*1.5, R*1.5, 3)
        points.append(p)
        
    points = np.array(points[:n])
    points = points + rng.normal(scale=0.015, size=points.shape)
    return points

def _basin_hopping(positions, rng, deadline, potential, parameter):
    """
    Simple basin hopping: perturb, relax, accept if lower.
    """
    current_energy = _energy(positions)
    best_energy = current_energy
    best_positions = positions.copy()
    
    step_size = 0.2
    attempts = 0
    max_attempts = 50
    
    while time.time() < deadline - 2 and attempts < max_attempts:
        attempts += 1
        # Perturb
        new_pos = positions + rng.normal(scale=step_size, size=positions.shape)
        
        # Relax
        try:
            e_relaxed, relaxed_pos = relax(new_pos, potential, parameter)
        except Exception:
            continue
            
        if e_relaxed < best_energy:
            best_energy = e_relaxed
            best_positions = relaxed_pos.copy()
            positions = relaxed_pos.copy()
            # Reduce step size slightly on success? Or keep same.
            # step_size *= 0.9
        else:
            # Metropolis accept?
            # delta_e = e_relaxed - current_energy
            # if delta_e < 0 or rng.random() < np.exp(-delta_e / 0.1):
            #    positions = relaxed_pos.copy()
            #    current_energy = e_relaxed
            pass
            
    return best_positions, best_energy

def _worker(args):
    """Worker function to relax a single starting structure."""
    positions, potential, parameter, deadline = args
    try:
        if time.time() > deadline - 2:
            return (np.inf, positions)
        energy, relaxed = relax(positions, potential, parameter)
        return (energy, relaxed)
    except Exception:
        return (np.inf, positions)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance
    
    # Calculate deadline for this specific call
    # We have ~45s per instance on average.
    # Let's allocate 40s for this instance.
    time_budget = 40.0
    start_time = time.time()
    deadline = start_time + time_budget
    
    rng_seed = size
    rng = np.random.default_rng(rng_seed)
    
    candidates = []
    
    # 1. Icosahedral (Mackay)
    candidates.append(_generate_polytetrahedral_seed(size, rng))
    
    # 2. Bimodal (2x Icosahedra)
    if size > 20:
        candidates.append(_generate_bimodal(size, rng))
        
    # 3. Random
    candidates.append(_generate_random(size, rng))
    
    # 4. Another Icosahedral with different seed
    rng2 = np.random.default_rng(rng_seed + 123)
    candidates.append(_generate_polytetrahedral_seed(size, rng2))
    
    # 5. Another Random
    rng3 = np.random.default_rng(rng_seed + 456)
    candidates.append(_generate_random(size, rng3))
    
    # 6. Bimodal with different seed
    if size > 20:
        rng4 = np.random.default_rng(rng_seed + 789)
        candidates.append(_generate_bimodal(size, rng4))

    # Limit to 4-6 candidates to fit in time
    max_candidates = 6
    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]
        
    tasks = [(pos, potential, parameter, deadline) for pos in candidates]
    
    best_energy = np.inf
    best_positions = candidates[0]
    
    # Use 4 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_worker, t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            if time.time() > deadline - 1:
                break
            try:
                energy, pos = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos
            except Exception:
                pass
    
    # If we have time, do basin hopping on the best result
    remaining = deadline - time.time()
    if remaining > 10 and best_energy < np.inf:
        try:
            bh_rng = np.random.default_rng(rng_seed + 999)
            best_positions, best_energy = _basin_hopping(
                best_positions, bh_rng, deadline, potential, parameter
            )
        except Exception:
            pass
            
    return best_positions

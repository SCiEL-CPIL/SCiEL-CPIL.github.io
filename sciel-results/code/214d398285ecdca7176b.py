"""
Strategy: Sequential Basin Hopping with Swap Perturbations and Diverse Seeds
- Uses a time-budgeted Basin Hopping loop to escape local minima.
- Incorporates atom-exchange (swap) perturbations to hop between structural isomers (e.g., icosahedral <-> decahedral).
- Generates diverse initial seeds: Mackay icosahedral, decahedral, and molten/random compact.
- Sequential execution avoids GIL contention and allows precise time management.
- Deadline-guarded to ensure valid output within 330s.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_T0 = time.time()
_DEADLINE = 320.0  # Safety margin

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def _icosahedron_vertices():
    """Generate vertices of a regular icosahedron normalized to unit radius."""
    phi = (1 + 5**0.5) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _decahedron_vertices():
    """
    Generate a set of points resembling decahedral symmetry (cube + face centers).
    """
    s = 1.0
    v = []
    for x in [-s, s]:
        for y in [-s, s]:
            for z in [-s, s]:
                v.append([x, y, z])
    v.append([0, 0, s])
    v.append([0, 0, -s])
    v.append([0, s, 0])
    v.append([0, -s, 0])
    v.append([s, 0, 0])
    v.append([-s, 0, 0])
    
    v = np.array(v, dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _generate_random_compact(n, rng, seed_scale=0.8):
    """
    Generate a random compact cluster using rejection sampling.
    """
    points = []
    radius = 0.8 * n ** (1.0 / 3.0) * 1.5 
    
    max_attempts = 5000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if points:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) < 0.5:
                attempts += 1
                continue
        points.append(p)
        attempts += 1
    
    if len(points) < n:
        for i in range(len(points), n):
            p = np.array([i * 1.0, 0, 0])
            points.append(p)
            
    return np.array(points)

def _generate_icosahedral_seed(n, rng):
    """
    Generate a seed based on Mackay icosahedra.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    magic = [1, 13, 55, 147]
    base_n = 1
    for mn in magic:
        if mn <= n:
            base_n = mn
        else:
            break
            
    if base_n == 1:
        pts = [np.array([0.0, 0.0, 0.0])]
    elif base_n == 13:
        v = _icosahedron_vertices()
        pts = [np.array([0.0, 0.0, 0.0])] + [v[i] for i in range(12)]
    elif base_n == 55:
        v1 = _icosahedron_vertices()
        pts = [np.array([0.0, 0.0, 0.0])]
        pts.extend([v1[i] for i in range(12)])
        
        for _ in range(42):
            dir = rng.normal(size=3)
            dir /= np.linalg.norm(dir)
            pts.append(dir * 1.5)
    else:
        return _generate_random_compact(n, rng)
        
    pts = np.array(pts)
    
    if n > len(pts):
        current_radius = 1.5 if base_n >= 13 else 0.5
        for _ in range(n - len(pts)):
            dir = rng.normal(size=3)
            dir /= np.linalg.norm(dir)
            pts = np.vstack([pts, [dir * (current_radius + 0.2)]])
            
    return pts

def _generate_decahedral_seed(n, rng):
    """
    Generate a seed based on decahedral symmetry.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
        
    v = _decahedron_vertices()
    pts = [np.array([0.0, 0.0, 0.0])]
    pts.extend([v[i] for i in range(min(n-1, len(v)))])
    
    pts = np.array(pts)
    
    if n > len(pts):
        current_radius = 1.2
        for _ in range(n - len(pts)):
            dir = rng.normal(size=3)
            dir /= np.linalg.norm(dir)
            pts = np.vstack([pts, [dir * (current_radius + 0.2)]])
            
    return pts

def _basin_hopping(positions, potential, kind, rng, max_iter=100):
    """
    Custom Basin Hopping implementation with swap perturbations.
    """
    best_pos = positions.copy()
    best_e, _ = sutton_chen_energy_gradient(best_pos, kind)
    
    # Initial relaxation
    e, pos = relax(positions, potential, kind)
    if np.isfinite(e) and e < best_e:
        best_e = e
        best_pos = pos
    
    current_e = e
    current_pos = pos
    
    for _ in range(max_iter):
        if _time_left() < 5.0:
            break
            
        # Perturb
        new_pos = current_pos.copy()
        
        # 50% chance: Gaussian noise
        if rng.random() < 0.5:
            noise = rng.normal(0, 0.5, new_pos.shape)
            new_pos += noise
        else:
            # 50% chance: Swap two atoms
            i, j = rng.choice(len(new_pos), size=2, replace=False)
            new_pos[i], new_pos[j] = new_pos[j].copy(), new_pos[i].copy()
            # Add small noise to break symmetry
            new_pos[i] += rng.normal(0, 0.1, 3)
            new_pos[j] += rng.normal(0, 0.1, 3)
        
        # Relax
        try:
            e_new, pos_new = relax(new_pos, potential, kind)
        except Exception:
            continue
            
        if not np.isfinite(e_new):
            continue
            
        # Metropolis step
        if e_new < current_e:
            current_e = e_new
            current_pos = pos_new
            if e_new < best_e:
                best_e = e_new
                best_pos = pos_new
        else:
            # Accept with probability exp(-(E_new - E_current)/T)
            # Simple T=1.0 for simplicity, or adaptive
            T = 1.0
            if rng.random() < np.exp(-(e_new - current_e) / T):
                current_e = e_new
                current_pos = pos_new
                
    return best_e, best_pos

def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    best_e = np.inf
    best_pos = None
    
    # Generate diverse seeds
    seeds = []
    rngs = [np.random.default_rng(i) for i in range(10)]
    
    # 1. Icosahedral seeds
    for i in range(3):
        try:
            s = _generate_icosahedral_seed(n, rngs[i])
            if s is not None and len(s) == n:
                seeds.append(s)
        except:
            pass
            
    # 2. Decahedral seeds
    for i in range(3, 6):
        try:
            s = _generate_decahedral_seed(n, rngs[i])
            if s is not None and len(s) == n:
                seeds.append(s)
        except:
            pass
            
    # 3. Random Compact seeds
    for i in range(6, 10):
        try:
            s = _generate_random_compact(n, rngs[i])
            if len(s) == n:
                seeds.append(s)
        except:
            pass

    if not seeds:
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos

    # Run Basin Hopping on each seed
    for i, seed in enumerate(seeds):
        if _time_left() < 10.0:
            break
            
        try:
            e, pos = _basin_hopping(seed, potential, kind, rngs[i % len(rngs)])
            if np.isfinite(e) and e < best_e:
                best_e = e
                best_pos = pos
        except Exception:
            pass

    # Fallback
    if best_pos is None:
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos

    # Validate
    if not np.all(np.isfinite(best_pos)):
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos
        
    # Check minimum distance
    d = np.linalg.norm(best_pos[:, None, :] - best_pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.35:
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos

    return best_pos
# EVOLVE-BLOCK-END

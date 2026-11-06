"""Basin-hopping search for Morse cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_DEADLINE = time.time() + 280.0  # safety margin before 300s kill


def _fcc_points(n):
    """Generate n FCC lattice points (NN=1) sorted by radius from origin."""
    a = np.sqrt(2.0)
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], dtype=float) * a
    pts, pts_set = [], set()
    R = 0
    while len(pts) < n and R < 100:
        for ix in range(-R, R + 1):
            for iy in range(-R, R + 1):
                for iz in range(-R, R + 1):
                    for b in basis:
                        p = np.array([ix, iy, iz]) * a + b
                        key = tuple(np.round(p, 6))
                        if key not in pts_set:
                            pts_set.add(key)
                            pts.append(p)
        R += 1
    pts = np.array(pts)
    norms = np.linalg.norm(pts, axis=1)
    order = np.argsort(norms)
    return pts[order[:n]]


def _simple_cubic_points(n):
    """Generate n simple cubic lattice points (NN=1) sorted by radius."""
    pts, pts_set = [], set()
    R = 0
    while len(pts) < n and R < 100:
        for ix in range(-R, R + 1):
            for iy in range(-R, R + 1):
                for iz in range(-R, R + 1):
                    key = (ix, iy, iz)
                    if key not in pts_set:
                        pts_set.add(key)
                        pts.append(np.array([ix, iy, iz], dtype=float))
        R += 1
    pts = np.array(pts)
    norms = np.linalg.norm(pts, axis=1)
    order = np.argsort(norms)
    return pts[order[:n]]


def _bcc_points(n):
    """Generate n BCC lattice points (NN=1)."""
    a = 2.0 / np.sqrt(3.0)
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=float) * a
    pts, pts_set = [], set()
    R = 0
    while len(pts) < n and R < 100:
        for ix in range(-R, R + 1):
            for iy in range(-R, R + 1):
                for iz in range(-R, R + 1):
                    for b in basis:
                        p = np.array([ix, iy, iz]) * a + b
                        key = tuple(np.round(p, 6))
                        if key not in pts_set:
                            pts_set.add(key)
                            pts.append(p)
        R += 1
    pts = np.array(pts)
    norms = np.linalg.norm(pts, axis=1)
    order = np.argsort(norms)
    return pts[order[:n]]


def _random_compact(n, seed):
    """Random compact cluster with minimum distance ~0.55."""
    rng = np.random.default_rng(seed)
    pts = []
    radius = n ** (1.0 / 3.0) * 0.9 + 2.0
    attempts = 0
    while len(pts) < n and attempts < 100000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        ok = True
        for q in pts:
            if np.linalg.norm(p - q) < 0.55:
                ok = False
                break
        if ok:
            pts.append(p)
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts)


def _enforce_min_dist(pts, min_d=0.501):
    """Vectorized enforcement of minimum pairwise distance."""
    n = len(pts)
    for _ in range(50):
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Find pairs violating min distance
        mask = (dists < min_d) & (np.arange(n)[:, None] < np.arange(n)[None, :])
        if not np.any(mask):
            break
        # Push apart violating pairs
        for i, j in zip(*np.where(mask)):
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.random.randn(3)
                dist = np.linalg.norm(d)
            d = d / dist * (min_d - dist) * 0.5
            pts[i] += d
            pts[j] -= d
    return pts


def _basin_hopping(pts, potential, rho, rng, max_iter=50, kick_scale=0.3):
    """Basin-hopping: perturb + relax iteratively."""
    best_e, best_p = np.inf, pts.copy()
    cur_e, cur_p = np.inf, pts.copy()
    
    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break
        # Relax current
        try:
            e, p = relax(cur_p, potential, rho)
            cur_e, cur_p = e, p
        except Exception:
            break
        
        if cur_e < best_e:
            best_e, best_p = cur_e, cur_p.copy()
        
        # Perturb
        kick = rng.normal(0, kick_scale, (len(cur_p), 3))
        new_p = cur_p + kick
        new_p = _enforce_min_dist(new_p)
        
        # Adaptive kick scale
        kick_scale *= 0.95
        
        cur_p = new_p
    
    return best_e, best_p


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    t_start = time.time()
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial seeds
    seeds = []
    try:
        seeds.append(_simple_cubic_points(size))
    except:
        pass
    try:
        seeds.append(_fcc_points(size))
    except:
        pass
    try:
        seeds.append(_bcc_points(size))
    except:
        pass
    # Multiple random seeds
    for s in [size, size + 42, size + 123, size + 777]:
        try:
            seeds.append(_random_compact(size, s))
        except:
            pass
    
    # Basin-hop from each seed
    for idx, seed_pts in enumerate(seeds):
        if time.time() > _DEADLINE:
            break
        
        rng = np.random.default_rng(idx * 1000 + size)
        
        # Run basin hopping with decreasing kick scales
        for kick_init in [0.5, 0.3, 0.15]:
            if time.time() > _DEADLINE:
                break
            
            e, p = _basin_hopping(seed_pts, potential, parameter, rng, 
                                   max_iter=30, kick_scale=kick_init)
            
            if e < best_energy:
                best_energy = e
                best_positions = p
    
    # If no valid result, use first seed
    if best_positions is None:
        best_positions = seeds[0].copy()
    
    # Final min-distance enforcement
    best_positions = _enforce_min_dist(best_positions)
    
    return best_positions
# EVOLVE-BLOCK-END

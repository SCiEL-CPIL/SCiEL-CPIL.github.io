"""Improved: Simulated Annealing with diverse initialization and local polishing."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

def _get_kind_params(kind):
    """Map kind string to (n_exp, m_exp, c)."""
    if kind == '12-6':
        return 12, 6, 144.41
    elif kind == '9-6':
        return 9, 6, 39.432
    elif kind == '10-8':
        return 10, 8, 34.408
    else:
        return 12, 6, 144.41

def _compute_energy(positions, n_exp, m_exp, c):
    """Compute Sutton-Chen energy."""
    n = len(positions)
    if n == 0:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    inv_r_n = np.power(dists, -n_exp)
    inv_r_m = np.power(dists, -m_exp)
    
    sum_inv_r_n = np.sum(inv_r_n, axis=1)
    sum_inv_r_m = np.sum(inv_r_m, axis=1)
    
    e_rep = 0.5 * np.sum(sum_inv_r_n)
    sum_inv_r_m_no_self = sum_inv_r_m - 1.0
    sum_inv_r_m_no_self = np.maximum(sum_inv_r_m_no_self, 0.0)
    e_att = -c * np.sum(np.sqrt(sum_inv_r_m_no_self))
    
    return e_rep + e_att

def _sa_optimize(positions, n_exp, m_exp, c, T0=1.0, Tend=0.01, steps=5000, alpha=0.999, seed=42, min_dist=0.35):
    """
    Vectorized Simulated Annealing with adaptive step size.
    Returns (best_energy, best_positions).
    """
    n = len(positions)
    rng = np.random.default_rng(seed)
    
    E = _compute_energy(positions, n_exp, m_exp, c)
    if not np.isfinite(E):
        return np.inf, positions
    
    best_E = E
    best_P = positions.copy()
    
    step_size = 0.1
    accept_count = 0
    total_count = 0
    window_size = 100
    window_accepts = []
    
    for i in range(steps):
        T = T0 * (alpha ** i)
        if T < Tend:
            break
        
        # Generate random move: pick one atom and displace it
        idx = rng.integers(0, n)
        displacement = rng.normal(0, step_size, 3)
        
        old_pos = positions[idx].copy()
        positions[idx] += displacement
        
        # Quick min distance check
        dists_to_others = np.linalg.norm(positions - positions[idx], axis=1)
        dists_to_others[idx] = 1.0
        if np.any(dists_to_others < min_dist):
            positions[idx] = old_pos
            total_count += 1
            window_accepts.append(0)
            if len(window_accepts) > window_size:
                window_accepts.pop(0)
            continue
        
        new_E = _compute_energy(positions, n_exp, m_exp, c)
        
        if not np.isfinite(new_E):
            positions[idx] = old_pos
            total_count += 1
            window_accepts.append(0)
            if len(window_accepts) > window_size:
                window_accepts.pop(0)
            continue
            
        delta_E = new_E - E
        total_count += 1
        
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            E = new_E
            accept_count += 1
            window_accepts.append(1)
            if E < best_E:
                best_E = E
                best_P = positions.copy()
        else:
            positions[idx] = old_pos
            window_accepts.append(0)
            
        if len(window_accepts) > window_size:
            window_accepts.pop(0)
        
        # Adaptive step size based on acceptance rate
        if len(window_accepts) >= window_size:
            accept_rate = np.mean(window_accepts)
            if accept_rate > 0.5:
                step_size *= 1.1
            elif accept_rate < 0.1:
                step_size *= 0.9
            step_size = np.clip(step_size, 0.01, 0.5)
    
    return best_E, best_P

def _polish(positions, potential, kind):
    """Local optimization using L-BFGS-B."""
    from autoresearch_suite.chem_clusters import relax
    try:
        e, p = relax(positions, potential, kind)
        if np.isfinite(e):
            return e, p
    except Exception:
        pass
    return np.inf, positions

def _fcc_fragment(n, spacing=1.0):
    """Generate n points from an FCC lattice."""
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 2
    fcc_pts = []
    for n1 in range(-max_coord, max_coord + 1):
        for n2 in range(-max_coord, max_coord + 1):
            for n3 in range(-max_coord, max_coord + 1):
                x = (n1 + n2) * 0.5
                y = (n2 + n3) * 0.5
                z = (n3 + n1) * 0.5
                fcc_pts.append((x, y, z))
    fcc_pts = np.array(fcc_pts) * spacing
    fcc_pts -= np.mean(fcc_pts, axis=0)
    radii = np.linalg.norm(fcc_pts, axis=1)
    idx = np.argsort(radii)
    return fcc_pts[idx[:n]]

def _random_compact(n, spacing=0.8, seed=42):
    """Random points with minimum distance."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0)
    points = []
    min_dist = 0.5 * spacing
    attempts = 0
    while len(points) < n and attempts < n * 500:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    if len(points) < n:
        min_dist = 0.35
        points = []
        attempts = 0
        while len(points) < n and attempts < n * 2000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= min_dist for q in points):
                    points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)

def _snowball_growth(n, spacing=0.8, seed=42, kind='12-6'):
    """Grow cluster by adding atoms one by one at lowest energy position."""
    rng = np.random.default_rng(seed)
    n_exp, m_exp, c = _get_kind_params(kind)
    
    positions = [np.zeros(3)]
    
    for i in range(1, n):
        best_pos = None
        best_e = np.inf
        for _ in range(30):
            u = rng.normal(size=3)
            u = u / np.linalg.norm(u)
            r = spacing * (i ** (1/3))
            pos = u * r + rng.normal(0, 0.1, 3)
            
            if all(np.linalg.norm(pos - q) >= 0.35 for q in positions):
                new_positions = np.array(positions + [pos])
                e = _compute_energy(new_positions, n_exp, m_exp, c)
                if e < best_e:
                    best_e = e
                    best_pos = pos
                    
        if best_pos is None:
            best_pos = rng.uniform(-spacing, spacing, 3)
            
        positions.append(best_pos)
        
    return np.array(positions)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    n_exp, m_exp, c = _get_kind_params(kind)
    
    start_time = time.time()
    time_limit = 250.0
    
    starts = []
    
    # 1. Random compact (multiple seeds)
    for seed in [42, 123, 456, 789, 1000, 2000, 3000]:
        try:
            rnd = _random_compact(size, spacing=0.8, seed=seed)
            starts.append(rnd)
        except Exception:
            pass
            
    # 2. FCC fragment
    try:
        fcc = _fcc_fragment(size, spacing=0.8)
        starts.append(fcc)
    except Exception:
        pass
        
    # 3. Snowball growth
    try:
        snow = _snowball_growth(size, spacing=0.8, seed=42, kind=kind)
        starts.append(snow)
    except Exception:
        pass
        
    # 4. Icosahedral
    try:
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ], dtype=float)
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        scale = 0.8 * size ** (1/3)
        all_ico = []
        for r_scale in [0.5, 0.8, 1.0]:
            all_ico.extend(verts * (scale * r_scale))
        all_ico = np.array(all_ico)
        all_ico -= np.mean(all_ico, axis=0)
        radii = np.linalg.norm(all_ico, axis=1)
        idx = np.argsort(radii)
        ico_frag = all_ico[idx[:size]]
        starts.append(ico_frag)
    except Exception:
        pass
        
    best_E = np.inf
    best_P = None
    
    # Optimize each start with SA + Polish
    for i, start in enumerate(starts):
        if time.time() - start_time > time_limit:
            break
            
        try:
            sa_E, sa_P = _sa_optimize(start, n_exp, m_exp, c, 
                                      T0=1.0, Tend=0.01, 
                                      steps=3000, alpha=0.9995, 
                                      seed=100+i)
            
            pol_E, pol_P = _polish(sa_P, potential, kind)
            
            if np.isfinite(pol_E) and pol_E < best_E:
                best_E = pol_E
                best_P = pol_P
        except Exception:
            continue
            
    # If we still have time, do more SA runs on the best solution found
    if best_P is not None and time.time() - start_time < time_limit - 20:
        for _ in range(5):
            if time.time() - start_time > time_limit:
                break
            perturbed = best_P + np.random.normal(0, 0.1, best_P.shape)
            try:
                sa_E, sa_P = _sa_optimize(perturbed, n_exp, m_exp, c, 
                                          T0=0.5, Tend=0.01, 
                                          steps=1500, alpha=0.999, 
                                          seed=1000+_)
                pol_E, pol_P = _polish(sa_P, potential, kind)
                if np.isfinite(pol_E) and pol_E < best_E:
                    best_E = pol_E
                    best_P = pol_P
            except Exception:
                pass

    if best_P is None:
        best_P = _random_compact(size, spacing=0.8, seed=42)
        best_E, best_P = _polish(best_P, potential, kind)
        
    if not np.all(np.isfinite(best_P)):
        best_P = _random_compact(size, spacing=0.8, seed=42)
        best_E, best_P = _polish(best_P, potential, kind)
        
    min_d = np.inf
    for i, j in combinations(range(len(best_P)), 2):
        d = np.linalg.norm(best_P[i] - best_P[j])
        if d < min_d:
            min_d = d
            
    if min_d < 0.35:
        for _ in range(10):
            perturbed = best_P + np.random.normal(0, 0.2, best_P.shape)
            try:
                e, p = _polish(perturbed, potential, kind)
                md = np.inf
                for i, j in combinations(range(len(p)), 2):
                    d = np.linalg.norm(p[i] - p[j])
                    if d < md:
                        md = d
                if md >= 0.35 and np.isfinite(e):
                    best_E, best_P = e, p
                    break
            except Exception:
                continue
                
    return best_P
# EVOLVE-BLOCK-END

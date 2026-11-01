import numpy as np
import scipy.spatial
import scipy.optimize
import time
import itertools

N = 14

# EVOLVE-BLOCK-START

# Pre-compute triangle indices for vectorized area calculation
def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def calc_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    try:
        if len(points) < 3:
            return 0.0
        
        # Calculate convex hull area
        hull = scipy.spatial.ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        
        if hull_area < 1e-10:
            return 0.0
            
        # Vectorized triangle area calculation
        pi = points[_I]
        pj = points[_J]
        pk = points[_K]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        min_area = 0.5 * np.abs(cross).min()
        
        return min_area / hull_area
    except Exception:
        return 0.0

def soft_min_score(points, beta=50.0):
    """
    Smooth surrogate using log-sum-exp for the min triangle area.
    Returns a smooth approximation of min(triangle_areas) / hull_area.
    Lower beta = smoother, higher beta = sharper (closer to true min).
    """
    try:
        if len(points) < 3:
            return 0.0
        
        hull = scipy.spatial.ConvexHull(points)
        hull_area = hull.volume
        
        if hull_area < 1e-10:
            return 0.0
            
        pi = points[_I]
        pj = points[_J]
        pk = points[_K]
        v1 = pj - pi
        v2 = pk - pi
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        # Soft min via log-sum-exp: soft_min(x) = -logsumexp(-beta * x) / beta
        scaled = -beta * areas
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min_val = -log_sum_exp / beta
        
        return soft_min_val / hull_area
    except Exception:
        return 0.0

def objective(points_flat):
    """
    Objective function for minimization.
    Minimizes negative score.
    """
    points = points_flat.reshape(-1, 2)
    score = calc_score(points)
    return -score

def soft_objective(points_flat, beta=50.0):
    """
    Smooth objective for gradient-based optimization.
    """
    points = points_flat.reshape(-1, 2)
    score = soft_min_score(points, beta)
    return -score

def _normalize(points):
    """Scale so convex hull area = 1."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        ha = hull.volume
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    """Regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _two_ring_config(r_outer, r_inner, rot_outer=0.0, rot_inner=0.0, n_outer=7, n_inner=7):
    """
    Two-ring configuration: n_outer points on outer ring, n_inner on inner ring.
    """
    outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + rot_outer
    inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + rot_inner
    outer = np.column_stack([r_outer * np.cos(outer_angles), r_outer * np.sin(outer_angles)])
    inner = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
    return np.vstack([outer, inner])

def _four_ring_config(r1, r2, r3, r4, rots=None, counts=(4, 4, 3, 3)):
    """
    Four-ring configuration for richer structure exploration.
    """
    if rots is None:
        rots = [0.0, 0.0, 0.0, 0.0]
    all_pts = []
    for i, (r, rot, cnt) in enumerate(zip([r1, r2, r3, r4], rots, counts)):
        angles = np.linspace(0, 2 * np.pi, cnt, endpoint=False) + rot
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        all_pts.append(pts)
    return np.vstack(all_pts)

def _dumbbell_config(centers, v, n_pairs=7):
    """
    Dumbbell topology: n_pairs pairs of points, each pair separated by vector v.
    centers: (n_pairs, 2) array of pair centers
    v: (2,) separation vector
    """
    pts = np.empty((n_pairs * 2, 2))
    for i in range(n_pairs):
        pts[2*i] = centers[i] - v / 2.0
        pts[2*i+1] = centers[i] + v / 2.0
    return pts

def _diff_evaluate_ring(params, n_outer=7, n_inner=7):
    """
    Evaluate a two-ring configuration from parameter vector.
    params: [r_outer, r_inner, rot_outer, rot_inner]
    """
    r_outer, r_inner, rot_outer, rot_inner = params
    pts = _two_ring_config(r_outer, r_inner, rot_outer, rot_inner, n_outer, n_inner)
    return calc_score(pts)

def _diff_evaluate_four_ring(params):
    """
    Evaluate a four-ring configuration from parameter vector.
    params: [r1, r2, r3, r4, rot1, rot2, rot3, rot4]
    """
    r1, r2, r3, r4 = params[0:4]
    rots = params[4:8]
    pts = _four_ring_config(r1, r2, r3, r4, rots=rots)
    return calc_score(pts)

def _diff_evaluate_dumbbell(params):
    """
    Evaluate a dumbbell configuration from parameter vector.
    params: [center_x1, center_y1, ..., center_x7, center_y7, vx, vy]
    Total: 14 + 2 = 16 params
    """
    centers = params[0:14].reshape(7, 2)
    v = params[14:16]
    pts = _dumbbell_config(centers, v, 7)
    return calc_score(pts)

def _differential_evolution_2d(func, bounds, pop_size=20, n_iter=200, F=0.8, CR=0.9, deadline=None):
    """
    Simple 2D/low-dim differential evolution for small parameter spaces.
    func: objective to maximize (returns score)
    bounds: list of (low, high) for each parameter
    Returns best_params, best_score
    """
    dim = len(bounds)
    rng = np.random.RandomState(42)
    
    # Initialize population
    pop = np.array([
        [rng.uniform(b[0], b[1]) for b in bounds]
        for _ in range(pop_size)
    ])
    
    scores = np.array([func(p) for p in pop])
    best_idx = np.argmax(scores)
    best_params = pop[best_idx].copy()
    best_score = scores[best_idx]
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        for i in range(pop_size):
            if deadline is not None and time.time() > deadline:
                break
            
            # Select three distinct random indices
            a, b, c = rng.choice(pop_size, 3, replace=False)
            
            # Mutation
            mutant = pop[a] + F * (pop[b] - pop[c])
            
            # Clip to bounds
            for d in range(dim):
                mutant[d] = np.clip(mutant[d], bounds[d][0], bounds[d][1])
            
            # Crossover
            trial = pop[i].copy()
            for d in range(dim):
                if rng.random() < CR or d == rng.randint(0, dim-1):
                    trial[d] = mutant[d]
            
            # Evaluate
            trial_score = func(trial)
            
            # Selection
            if trial_score > scores[i]:
                pop[i] = trial
                scores[i] = trial_score
                if trial_score > best_score:
                    best_score = trial_score
                    best_params = trial.copy()
    
    return best_params, best_score

def _polish_points(points, maxiter=5000, ftol=1e-12, xtol=1e-10):
    """
    High-precision polish pass using Nelder-Mead with tight tolerances.
    """
    try:
        norm_pts = _normalize(points)
        res = scipy.optimize.minimize(
            objective,
            norm_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': xtol, 'fatol': ftol, 'adaptive': True}
        )
        opt_pts = _normalize(res.x.reshape(-1, 2))
        return opt_pts
    except Exception:
        return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_points = None
    
    def _update_best(pts):
        nonlocal best_score, best_points
        norm_pts = _normalize(pts)
        sc = calc_score(norm_pts)
        if sc > best_score:
            best_score = sc
            best_points = norm_pts.copy()
        return sc
    
    def _try_optimize(init_points, method='Nelder-Mead', maxiter=2000, ftol=1e-10, xtol=1e-8):
        """Try optimizing from init_points, return (score, points) or None."""
        nonlocal best_score, best_points
        if time.time() > deadline:
            return None
        try:
            init_norm = _normalize(init_points)
            init_score = calc_score(init_norm)
            
            if init_score > best_score:
                best_score = init_score
                best_points = init_norm.copy()
            
            if method == 'Nelder-Mead':
                res = scipy.optimize.minimize(
                    objective,
                    init_norm.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': xtol, 'fatol': ftol, 'adaptive': True}
                )
            elif method == 'Powell':
                res = scipy.optimize.minimize(
                    objective,
                    init_norm.flatten(),
                    method='Powell',
                    options={'maxiter': maxiter, 'ftol': ftol, 'xtol': xtol}
                )
            else:
                return None
            
            opt_pts = _normalize(res.x.reshape(-1, 2))
            sc = calc_score(opt_pts)
            
            if sc > best_score:
                best_score = sc
                best_points = opt_pts.copy()
            
            return (sc, opt_pts)
        except Exception:
            return None
    
    # === Phase 1: Ring-based Differential Evolution ===
    # Two-ring DE: 4 parameters (r_outer, r_inner, rot_outer, rot_inner)
    if time.time() < deadline - 30:
        bounds_2ring = [(0.1, 2.0), (0.0, 1.5), (0.0, 2*np.pi), (0.0, 2*np.pi)]
        best_2ring_params, best_2ring_score = _differential_evolution_2d(
            _diff_evaluate_ring, bounds_2ring, 
            pop_size=24, n_iter=300, F=0.7, CR=0.9, deadline=deadline - 25
        )
        r_o, r_i, ro_o, ro_i = best_2ring_params
        pts_2ring = _two_ring_config(r_o, r_i, ro_o, ro_i, 7, 7)
        sc = _update_best(pts_2ring)
        
        # Polish the two-ring result
        if time.time() < deadline - 15:
            polished = _polish_points(pts_2ring, maxiter=3000, ftol=1e-12, xtol=1e-10)
            _update_best(polished)
    
    # Four-ring DE: 8 parameters
    if time.time() < deadline - 25:
        bounds_4ring = [(0.05, 2.0)] * 4 + [(0.0, 2*np.pi)] * 4
        best_4ring_params, best_4ring_score = _differential_evolution_2d(
            _diff_evaluate_four_ring, bounds_4ring,
            pop_size=20, n_iter=200, F=0.7, CR=0.9, deadline=deadline - 20
        )
        pts_4ring = _four_ring_config(
            best_4ring_params[0], best_4ring_params[1],
            best_4ring_params[2], best_4ring_params[3],
            rots=list(best_4ring_params[4:8])
        )
        sc = _update_best(pts_4ring)
        
        if time.time() < deadline - 10:
            polished = _polish_points(pts_4ring, maxiter=2000, ftol=1e-12, xtol=1e-10)
            _update_best(polished)
    
    # Dumbbell DE: 16 parameters (7 centers + separation vector)
    if time.time() < deadline - 20:
        bounds_dumb = [(0.0, 2.0)] * 14 + [(-0.5, 0.5), (-0.5, 0.5)]
        best_dumb_params, best_dumb_score = _differential_evolution_2d(
            _diff_evaluate_dumbbell, bounds_dumb,
            pop_size=16, n_iter=150, F=0.6, CR=0.85, deadline=deadline - 15
        )
        centers = best_dumb_params[0:14].reshape(7, 2)
        v = best_dumb_params[14:16]
        pts_dumb = _dumbbell_config(centers, v, 7)
        sc = _update_best(pts_dumb)
        
        if time.time() < deadline - 8:
            polished = _polish_points(pts_dumb, maxiter=2000, ftol=1e-12, xtol=1e-10)
            _update_best(polished)
    
    # === Phase 2: Soft-min guided optimization from best structure ===
    if time.time() < deadline - 15 and best_points is not None:
        # Use soft-min objective with annealed beta
        for beta in [20.0, 40.0, 80.0]:
            if time.time() > deadline - 10:
                break
            try:
                res = scipy.optimize.minimize(
                    lambda x: soft_objective(x, beta=beta),
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
                soft_pts = _normalize(res.x.reshape(-1, 2))
                _update_best(soft_pts)
            except Exception:
                pass
    
    # === Phase 3: Standard diverse initializations + local optimization ===
    inits = []
    
    # Regular polygon with various rotations
    for rot in [0.0, np.pi / 14, np.pi / 7, np.pi / 3]:
        inits.append(_regular_polygon(N, 1.0, rot))
    
    # Perturbed regular polygons
    for seed in range(8):
        for ns in [0.03, 0.08, 0.15]:
            rng = np.random.RandomState(seed + 100)
            base = _regular_polygon(N, 1.0, seed * 0.1)
            noise = rng.uniform(-ns, ns, (N, 2))
            inits.append(base + noise)
    
    # Two-ring configurations with varied parameters
    for r_i in [0.3, 0.5, 0.7]:
        for rot_off in [0.0, np.pi/7, np.pi/14]:
            inits.append(_two_ring_config(1.0, r_i, 0.0, rot_off, 7, 7))
    
    # Score and sort initializations
    scored_inits = []
    for init in inits:
        try:
            init_norm = _normalize(init)
            sc = calc_score(init_norm)
            scored_inits.append((sc, init_norm))
        except Exception:
            continue
    
    scored_inits.sort(key=lambda x: -x[0])
    
    # Update best with best initial
    if scored_inits:
        best_init_score, best_init_pts = scored_inits[0]
        if best_init_score > best_score:
            best_score = best_init_score
            best_points = best_init_pts.copy()
    
    # Optimize from top initializations
    for rank, (init_score, init_pts) in enumerate(scored_inits):
        if time.time() > deadline - 5:
            break
        
        time_left = deadline - time.time()
        if time_left > 5:
            _try_optimize(init_pts, 'Nelder-Mead', maxiter=2000, ftol=1e-11, xtol=1e-9)
        
        time_left = deadline - time.time()
        if time_left > 8 and rank < 10:
            _try_optimize(init_pts, 'Powell', maxiter=400, ftol=1e-10, xtol=1e-8)
    
    # === Phase 4: Final high-precision polish ===
    if time.time() < deadline - 3 and best_points is not None:
        polished = _polish_points(best_points, maxiter=5000, ftol=1e-13, xtol=1e-11)
        _update_best(polished)
    
    # Final fallback
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0, 0.0))
    else:
        best_points = _normalize(best_points)
    
    return best_points

# EVOLVE-BLOCK-END

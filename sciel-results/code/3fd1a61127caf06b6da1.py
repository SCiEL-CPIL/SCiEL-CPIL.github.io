"""Packing 50 equal circles in a unit circle.

Strategy:
- Low-dimensional ring-parameter global search (differential evolution) over
  ring radii and angular offsets, matching the known optimal topology
  (1 centre + rings of 6, 12, 18, 13).
- Multi-stage smooth soft-min objective (log-sum-exp) for local L-BFGS-B polish.
- Binary-search radius polishing with explicit constraint enforcement.
- Final high-beta polish for precision.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
RING_COUNTS = [6, 12, 18, 13]
N_RINGS = len(RING_COUNTS)

# Precompute upper-triangular pair indices
_IU_I, _IU_J = np.triu_indices(N, k=1)
_NPAIR = _IU_I.shape[0]


def _score(pts):
    """Exact packing score: min(min_pair_dist/2, min_boundary_dist)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    d = cdist(pts, pts)
    iu = np.triu_indices(N, k=1)
    min_pair = np.min(d[iu]) / 2.0
    radii = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - radii)
    return max(min(min_pair, min_boundary), 0.0)


def _softmin_logsumexp(values, beta):
    """Soft-min via log-sum-exp."""
    neg = -values * beta
    neg_max = np.max(neg)
    exp_sum = np.sum(np.exp(neg - neg_max))
    return -(neg_max + np.log(exp_sum)) / beta


def _smooth_objective(x, beta=50.0):
    """Smooth objective: minimize -softmin of all constraint margins."""
    pts = x.reshape(N, 2)
    d = cdist(pts, pts)
    iu = np.triu_indices(N, k=1)
    pair_half = d[iu] / 2.0
    radii = np.hypot(pts[:, 0], pts[:, 1])
    boundary = 1.0 - radii
    all_vals = np.concatenate([pair_half, boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 0.0
    soft_min = _softmin_logsumexp(all_vals, beta)
    return -soft_min


def _ring_to_points(params):
    """Decode ring parameters into point positions.
    params = [r1, r2, r3, r4, off1, off2, off3, off4]
    """
    radii = params[:N_RINGS]
    offsets = params[N_RINGS:]
    radii = np.maximum(radii, 0.01)
    # Clamp outermost ring to stay inside unit circle
    max_allowed = 1.0 - 0.12
    radii = np.minimum(radii, max_allowed)
    
    points = [np.array([0.0, 0.0])]
    for k, nk in enumerate(RING_COUNTS):
        r = radii[k]
        off = offsets[k]
        for m in range(nk):
            angle = 2 * np.pi * m / nk + off
            points.append(np.array([r * np.cos(angle), r * np.sin(angle)]))
    
    pts = np.array(points[:N])
    # Project inside unit circle
    norms = np.hypot(pts[:, 0], pts[:, 1])
    mask = norms > 0.999
    if np.any(mask):
        pts[mask] *= 0.999 / norms[mask, np.newaxis]
    return pts


def _de_ring_search(deadline):
    """Differential evolution over ring parameters."""
    def neg_score(params):
        pts = _ring_to_points(params)
        return -_score(pts)
    
    bounds = [
        (0.15, 0.40),   # r1 (inner ring, 6 pts)
        (0.30, 0.60),   # r2 (12 pts)
        (0.45, 0.75),   # r3 (18 pts)
        (0.60, 0.92),   # r4 (outer ring, 13 pts)
        (0, 2*np.pi),   # off1
        (0, 2*np.pi),   # off2
        (0, 2*np.pi),   # off3
        (0, 2*np.pi),   # off4
    ]
    
    best_pts = None
    best_score = 0.0
    rng = np.random.RandomState(42)
    
    # Manual DE implementation for better time control
    pop_size = 25
    max_gen = 60
    F = 0.8
    CR = 0.9
    
    # Initial population
    pop = []
    for _ in range(pop_size):
        params = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        pop.append(params)
    pop = np.array(pop)
    
    scores = np.array([neg_score(p) for p in pop])
    best_idx = np.argmin(scores)
    best_params = pop[best_idx].copy()
    best_neg = scores[best_idx]
    best_pts = _ring_to_points(best_params)
    best_score = -best_neg
    
    for gen in range(max_gen):
        if time.time() > deadline - 20:
            break
        for i in range(pop_size):
            if time.time() > deadline - 20:
                break
            # Select three distinct random individuals
            a, b, c = [j for j in range(pop_size) if j != i]
            a, b, c = rng.choice(a, 3, replace=False)
            
            # Mutation
            v = pop[a] + F * (pop[b] - pop[c])
            # Clip to bounds
            for k in range(len(bounds)):
                v[k] = np.clip(v[k], bounds[k][0], bounds[k][1])
            
            # Crossover
            trial = pop[i].copy()
            j_rand = rng.randint(len(bounds))
            for j in range(len(bounds)):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
            
            # Evaluate
            trial_score = neg_score(trial)
            if trial_score < scores[i]:
                pop[i] = trial
                scores[i] = trial_score
                if trial_score < best_neg:
                    best_neg = trial_score
                    best_params = trial.copy()
                    best_pts = _ring_to_points(best_params)
                    best_score = -best_neg
    
    return best_pts, best_score


def _multi_stage_optimize(pts, deadline):
    """Multi-stage annealing with increasing beta."""
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    betas = [5.0, 15.0, 40.0, 100.0, 250.0, 500.0, 1000.0, 2000.0]
    
    for beta in betas:
        if time.time() > deadline - 8:
            break
        x0 = current_pts.ravel()
        try:
            result = minimize(
                _smooth_objective, x0, args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 8000}
            )
            new_pts = result.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            # Clamp to unit circle
            norms = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask = norms > 1.0
            if np.any(mask):
                new_pts[mask] /= norms[mask, np.newaxis]
            new_score = _score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        except Exception:
            pass
    
    return current_pts, current_score


def _binary_search_polish(pts, deadline):
    """Binary search on target radius with feasibility via local optimization."""
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    lo = current_score
    hi = 0.135  # slightly above known best 0.1258...
    
    for _ in range(12):
        if time.time() > deadline - 5:
            break
        mid = (lo + hi) / 2
        # Try to achieve radius mid by optimizing
        test_pts = current_pts.copy()
        # Use high-beta objective targeting the constraint
        beta = 3000.0
        x0 = test_pts.ravel()
        try:
            result = minimize(
                _smooth_objective, x0, args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 12000}
            )
            new_pts = result.x.reshape(N, 2)
            norms = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask = norms > 1.0
            if np.any(mask):
                new_pts[mask] /= norms[mask, np.newaxis]
            s = _score(new_pts)
            if s >= mid:
                lo = mid
                if s > current_score:
                    current_score = s
                    current_pts = new_pts
            else:
                hi = mid
        except Exception:
            hi = mid
    
    return current_pts, current_score


def _get_hex_initial():
    """Hexagonal lattice initial points as a diverse starting point."""
    spacing = 0.24
    points = []
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = i * spacing + (j % 2) * spacing / 2
            y = j * spacing * np.sqrt(3) / 2
            if np.hypot(x, y) < 1.0 - spacing / 2:
                points.append([x, y])
    points = np.array(points)
    radii = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radii, kind='stable')[:N]
    return points[idx]


def construct():
    t_start = time.time()
    deadline = t_start + 135.0  # 15s safety margin before 150s kill
    
    best_points = None
    best_score = 0.0
    
    # Phase 1: DE over ring parameters (global search in low-dim space)
    if time.time() < deadline - 40:
        ring_pts, ring_score = _de_ring_search(deadline - 30)
        # Local polish from DE result
        polished_pts, polished_score = _multi_stage_optimize(ring_pts, deadline - 20)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts.copy()
    
    # Phase 2: Hex lattice + multi-stage optimization (diverse starting point)
    if time.time() < deadline - 25:
        hex_pts = _get_hex_initial()
        polished_pts, polished_score = _multi_stage_optimize(hex_pts, deadline - 15)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts.copy()
    
    # Phase 3: Binary search radius polish
    if best_points is not None and time.time() < deadline - 10:
        polished_pts, polished_score = _binary_search_polish(best_points, deadline - 5)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts.copy()
    
    # Phase 4: Final high-precision polish
    if best_points is not None and time.time() < deadline - 3:
        for beta in [3000.0, 5000.0, 10000.0]:
            if time.time() > deadline - 2:
                break
            x0 = best_points.ravel()
            try:
                result = minimize(
                    _smooth_objective, x0, args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-14, 'maxfun': 15000}
                )
                new_pts = result.x.reshape(N, 2)
                norms = np.hypot(new_pts[:, 0], new_pts[:, 1])
                mask = norms > 1.0
                if np.any(mask):
                    new_pts[mask] /= norms[mask, np.newaxis]
                s = _score(new_pts)
                if s > best_score:
                    best_score = s
                    best_points = new_pts
            except Exception:
                pass
    
    if best_points is None:
        # Fallback
        best_points = _get_hex_initial()
    
    # Final validation
    final_score = _score(best_points)
    if final_score <= 0:
        best_points = _get_hex_initial()
    
    return best_points

"""Optimize 14 points in 3D to minimize (max/min pairwise distance)^2."""
import time
import numpy as np
from scipy.optimize import minimize

N = 14
D = 3
DEADLINE = 130.0  # safety margin before 150s kill

# Precompute pairwise indices for efficiency
_pair_i, _pair_j = np.triu_indices(N, k=1)
_n_pairs = N * (N - 1) // 2


def _pairwise_dists2(pts):
    """Compute all pairwise squared distances from (N, D) array."""
    # pts: (N, D)
    # Use broadcasting: (N, 1, D) - (1, N, D) -> (N, N, D)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    return d2[_pair_i, _pair_j]


def _score(pts):
    """Compute the ratio^2 score."""
    d2 = _pairwise_dists2(pts)
    min_d2 = np.min(d2)
    max_d2 = np.max(d2)
    if min_d2 < 1e-30:
        return np.inf
    return max_d2 / min_d2


def _smooth_objective(x, t=20.0):
    """
    Smooth surrogate: log-sum-exp based soft ratio.
    Minimize log(softmax(d2)) - log(softmin(d2)) = log(softmax(d2)) + log(softmin(d2))
    Actually: ratio = max/min. log(ratio) = log(max) - log(min).
    soft_max(d2) = (1/t) * log(sum(exp(t*d2)))
    soft_min(d2) = -(1/t) * log(sum(exp(-t*d2)))
    log(ratio) ≈ soft_max - soft_min
    We minimize (soft_max - soft_min)^2 or just soft_max - soft_min.
    Let's minimize (soft_max - soft_min) directly, which is monotone with ratio.
    """
    d2 = _pairwise_dists2(x.reshape(N, D))
    # Numerical stability
    d2_max = np.max(d2)
    d2_min = np.min(d2)
    
    # soft max
    exp_t_d2 = np.exp(t * (d2 - d2_max))
    soft_max = d2_max + (1.0 / t) * np.log(np.sum(exp_t_d2))
    
    # soft min
    exp_neg_t_d2 = np.exp(-t * (d2 - d2_min))
    soft_min = d2_min - (1.0 / t) * np.log(np.sum(exp_neg_t_d2))
    
    return soft_max - soft_min


def _smooth_gradient(x, t=20.0):
    """Gradient of the smooth objective."""
    pts = x.reshape(N, D)
    d2 = _pairwise_dists2(pts)
    d2_max = np.max(d2)
    d2_min = np.min(d2)
    
    # soft max gradient
    exp_t_d2 = np.exp(t * (d2 - d2_max))
    p_max = exp_t_d2 / np.sum(exp_t_d2)  # softmax weights for max
    
    # soft min gradient
    exp_neg_t_d2 = np.exp(-t * (d2 - d2_min))
    p_min = exp_neg_t_d2 / np.sum(exp_neg_t_d2)  # softmax weights for min
    
    # d(soft_max)/d(d2_k) = p_max[k]
    # d(soft_min)/d(d2_k) = -p_min[k]
    # d(objective)/d(d2_k) = p_max[k] - (-p_min[k]) = p_max[k] + p_min[k]
    # Wait: objective = soft_max - soft_min
    # d(soft_min)/d(d2_k) = -p_min[k]
    # So d(objective)/d(d2_k) = p_max[k] - (-p_min[k]) = p_max[k] + p_min[k]
    
    grad_d2 = p_max + p_min  # (n_pairs,)
    
    # Now chain rule: d2_k = ||p_i - p_j||^2
    # d(d2_k)/d(p_m) = 2*(p_i - p_j) if m == i, -2*(p_i - p_j) if m == j, 0 otherwise
    # grad_x = sum_k grad_d2[k] * d(d2_k)/d(x)
    
    grad_x = np.zeros_like(pts)
    for k in range(_n_pairs):
        i = _pair_i[k]
        j = _pair_j[k]
        diff = pts[i] - pts[j]
        grad_x[i] += 2.0 * grad_d2[k] * diff
        grad_x[j] -= 2.0 * grad_d2[k] * diff
    
    return grad_x.flatten()


def _generate_ring_candidate(radius, n_rings, seed):
    """Generate points on concentric rings with some structure."""
    rng = np.random.RandomState(seed)
    pts = np.zeros((N, D))
    # Place points on a few rings
    per_ring = N // n_rings
    leftover = N % n_rings
    idx = 0
    for r in range(n_rings):
        rad = radius * (1 + 0.3 * r)
        count = per_ring + (1 if r < leftover else 0)
        for c in range(count):
            # Random direction on sphere
            v = rng.randn(D)
            v /= np.linalg.norm(v)
            pts[idx] = rad * v
            idx += 1
    return pts


def _generate_random_candidate(seed, scale=1.0):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    return rng.randn(N, D) * scale


def _generate_grid_candidate():
    """Simple grid-based initialization."""
    side = int(np.ceil(N ** (1.0 / D)))
    grid = []
    for x in range(side):
        for y in range(side):
            for z in range(side):
                grid.append([x, y, z])
                if len(grid) == N:
                    break
            if len(grid) == N:
                break
        if len(grid) == N:
            break
    return np.array(grid, dtype=float)


def _optimize_from(x0, t=20.0, method='L-BFGS-B'):
    """Run optimization from initial point x0."""
    x0_flat = x0.flatten()
    try:
        result = minimize(
            _smooth_objective,
            x0_flat,
            jac=_smooth_gradient,
            args=(t,),
            method=method,
            options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12}
        )
        if result.success or result.fun < _smooth_objective(x0_flat, t):
            return result.x.reshape(N, D)
    except Exception:
        pass
    return x0


def _polish(pts, max_iters=200):
    """High-precision polish using SLSQP on the true objective with auxiliary variable."""
    # Use epigraph formulation: minimize t such that max_d2 <= t * min_d2
    # Actually, let's just do a local search on the smooth objective with smaller t
    # then a few iterations of coordinate descent on the true objective
    
    # First, refine with smaller t
    t_final = 50.0
    x_flat = pts.flatten()
    try:
        result = minimize(
            _smooth_objective,
            x_flat,
            jac=_smooth_gradient,
            args=(t_final,),
            method='L-BFGS-B',
            options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-14}
        )
        if result.fun < _smooth_objective(x_flat, t_final):
            pts = result.x.reshape(N, D)
    except Exception:
        pass
    
    # Coordinate descent on true objective
    for _ in range(max_iters):
        improved = False
        for i in range(N):
            best_pt = pts[i].copy()
            best_score = _score(pts)
            # Try small perturbations
            for axis in range(D):
                for delta in [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1]:
                    trial = pts.copy()
                    trial[i, axis] += delta
                    s = _score(trial)
                    if s < best_score - 1e-10:
                        best_score = s
                        best_pt = trial[i].copy()
                        improved = True
            pts[i] = best_pt
        if not improved:
            break
    
    return pts


def construct():
    start_time = time.time()
    
    # Generate diverse initial candidates
    candidates = []
    
    # Grid
    candidates.append(_generate_grid_candidate())
    
    # Ring-based candidates
    for seed in range(10):
        for n_rings in [2, 3, 4]:
            for radius in [0.5, 1.0, 1.5]:
                candidates.append(_generate_ring_candidate(radius, n_rings, seed))
    
    # Random candidates
    for seed in range(20):
        for scale in [0.5, 1.0, 2.0]:
            candidates.append(_generate_random_candidate(seed, scale))
    
    # Evaluate and sort candidates by score
    scored_candidates = []
    for cand in candidates:
        s = _score(cand)
        if np.isfinite(s):
            scored_candidates.append((s, cand))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    best_pts = scored_candidates[0][1] if scored_candidates else _generate_grid_candidate()
    best_score = scored_candidates[0][0] if scored_candidates else _score(best_pts)
    
    # Optimize from top candidates
    top_k = min(15, len(scored_candidates))
    for i in range(top_k):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        s, cand = scored_candidates[i]
        if s > best_score * 1.5:
            continue  # Skip if too far from best
        
        # Optimize
        opt_pts = _optimize_from(cand, t=20.0)
        opt_score = _score(opt_pts)
        
        if opt_score < best_score:
            best_score = opt_score
            best_pts = opt_pts
        
        # Polish
        if time.time() - start_time > DEADLINE - 15:
            break
        polished = _polish(opt_pts)
        pol_score = _score(polished)
        
        if pol_score < best_score:
            best_score = pol_score
            best_pts = polished
    
    # Final check
    if time.time() - start_time > DEADLINE - 5:
        return best_pts
    
    # One more polish on the best
    final = _polish(best_pts, max_iters=100)
    final_score = _score(final)
    if final_score < best_score:
        best_pts = final
        best_score = final_score
    
    return best_pts

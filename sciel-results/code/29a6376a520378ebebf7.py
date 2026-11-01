"""
Trust-region EDM formulation: optimize squared distances directly with
bisection on the ratio, using smooth log-sum-exp objectives and L-BFGS-B.
Combines the tetrahedron+midpoint initialization with a distance-space
optimization strategy.
"""
import time
import numpy as np
from scipy.optimize import minimize
from functools import partial

N = 14
D = 3
NUM_PAIRS = N * (N - 1) // 2

_DEADLINE = 145.0
_t0 = time.time()


def _elapsed():
    return time.time() - _t0


def _pairwise_dists_sq(P):
    """Return squared pairwise distances for P (n x d). Shape (n, n)."""
    diff = P[:, None, :] - P[None, :, :]
    return np.einsum('ijk,ijk->ij', diff, diff)


def _get_upper_sq_dists(P):
    """Return flat array of squared pairwise distances (upper triangle)."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]


def _score_from_P(P):
    """Compute (max/min)^2 ratio of pairwise distances."""
    d2 = _pairwise_dists_sq(P)
    mask = ~np.eye(N, dtype=bool)
    vals = d2[mask]
    mn = vals.min()
    mx = vals.max()
    if mn < 1e-12:
        return np.inf
    return (mx / mn) ** 2


def _init_tetrahedron_midpoints():
    """
    Construct 14 points: 4 vertices of a regular tetrahedron + 6 edge midpoints + 4 face centroids.
    """
    a = 1.0 / np.sqrt(2)
    verts = np.array([
        [a, a, a],
        [a, -a, -a],
        [-a, a, -a],
        [-a, -a, a],
    ])
    verts *= np.sqrt(2)  # edge length = 2

    points = [verts]
    # Edge midpoints
    for i in range(4):
        for j in range(i + 1, 4):
            mid = (verts[i] + verts[j]) / 2.0
            points.append(mid)
    # Face centroids
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    for f in faces:
        centroid = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        points.append(centroid)

    P = np.vstack(points)
    return P


def _fibonacci_sphere(n_points):
    """Generate n_points on a sphere using Fibonacci lattice."""
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_points):
        y = 1.0 - (i / (n_points - 1)) * 2.0
        radius = np.sqrt(1.0 - y**2)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)


def _edm_penalty(P, weight=1.0):
    """
    Compute EDM (Euclidean Distance Matrix) penalty.
    Double-center the distance matrix and check for negative eigenvalues.
    """
    # Compute pairwise squared distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    
    # Double centering: B = -0.5 * J * D2 * J where J = I - (1/n)11^T
    n = P.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ d2 @ J
    
    # Check PSD: penalty = sum of negative eigenvalues
    eigvals = np.linalg.eigvalsh(B)
    neg_eig = eigvals[eigvals < 0]
    if len(neg_eig) == 0:
        return 0.0
    return weight * np.sum(neg_eig**2)


def _objective_smooth(x, tau=10.0, edm_weight=0.01):
    """
    Smooth log-ratio objective using soft-max and soft-min, plus EDM penalty.
    x is flattened array of shape (N*D,)
    """
    points = x.reshape(N, D)
    sq_dists = _get_upper_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Soft max and soft min via log-sum-exp
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    exp_min_terms = np.exp(-tau * (sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    soft_min = np.maximum(soft_min, 1e-15)
    
    obj = np.log(soft_max) - np.log(soft_min)
    
    # EDM penalty
    obj += _edm_penalty(points, edm_weight)
    
    return obj


def _objective_precise(x, tau=5000.0, edm_weight=0.001):
    """High-precision smooth objective for final polish."""
    return _objective_smooth(x, tau=tau, edm_weight=edm_weight)


def _polish_with_smooth_objective(P_init, tau_schedule, maxiter_schedule, edm_weight_schedule):
    """
    Polish points using L-BFGS-B with increasing tau (sharper max/min approximation).
    """
    x_curr = P_init.flatten()
    
    for idx, tau in enumerate(tau_schedule):
        if _elapsed() > _DEADLINE - 10:
            break
        
        edm_w = edm_weight_schedule[idx] if idx < len(edm_weight_schedule) else edm_weight_schedule[-1]
        maxiter = maxiter_schedule[idx] if idx < len(maxiter_schedule) else maxiter_schedule[-1]
        
        obj = partial(_objective_smooth, tau=tau, edm_weight=edm_w)
        
        try:
            res = minimize(
                obj,
                x_curr,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 60}
            )
            if res.x is not None and np.isfinite(res.fun):
                x_curr = res.x
        except Exception:
            pass
    
    return x_curr.reshape(N, D)


def _bisection_ratio_search(P_init, ratio_lo, ratio_hi, n_bisections=20):
    """
    Bisection search on the ratio r = d_max / d_min.
    For a given ratio r, we want to find points where all distances are in [d_min, d_max]
    with d_max/d_min <= r. We reformulate as: minimize max(d_ij/d_min - r) subject to d_ij >= d_min.
    
    Simplified: for a target ratio r, scale points so min distance = 1, then check if max distance <= r.
    """
    best_P = P_init.copy()
    best_score = _score_from_P(P_init)
    
    # First, normalize P_init so min distance = 1
    sq_dists = _get_upper_sq_dists(P_init)
    min_d = np.sqrt(np.min(sq_dists))
    if min_d < 1e-12:
        return P_init, best_score
    
    P_norm = P_init / min_d
    current_ratio = np.max(sq_dists) / np.min(sq_dists)
    
    lo = max(1.0, current_ratio * 0.8)
    hi = current_ratio * 1.2
    
    for b in range(n_bisections):
        if _elapsed() > _DEADLINE - 15:
            break
        
        mid = (lo + hi) / 2.0
        
        # Try to optimize with constraint that max/min <= mid
        # Use a penalty-based approach
        def obj_constrained(x, target_ratio=mid):
            P = x.reshape(N, D)
            sq_dists = _get_upper_sq_dists(P)
            sq_dists = np.maximum(sq_dists, 1e-15)
            
            mn = np.min(sq_dists)
            mx = np.max(sq_dists)
            
            # Penalty for ratio exceeding target
            ratio = mx / mn
            if ratio > target_ratio:
                penalty = (ratio - target_ratio) ** 2 * 100.0
            else:
                penalty = 0.0
            
            # Also penalize if min distance is too small (degenerate)
            if mn < 0.01:
                penalty += (0.01 - mn) ** 2 * 1000.0
            
            return penalty + _edm_penalty(P, 0.01)
        
        x0 = P_norm.flatten()
        try:
            res = minimize(
                obj_constrained,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None and np.isfinite(res.fun):
                P_try = res.x.reshape(N, D)
                s = _score_from_P(P_try)
                if np.isfinite(s) and s < best_score:
                    best_score = s
                    best_P = P_try
                
                # Check if ratio constraint is satisfied
                sq_dists_try = _get_upper_sq_dists(P_try)
                actual_ratio = np.max(sq_dists_try) / np.min(sq_dists_try)
                
                if actual_ratio <= mid * 1.01:
                    hi = mid
                else:
                    lo = mid
            else:
                lo = mid
        except Exception:
            lo = mid
    
    return best_P, best_score


def _random_restart_polish(P_init, n_restarts=5):
    """Try multiple random perturbations and polish each."""
    best_P = P_init.copy()
    best_score = _score_from_P(P_init)
    
    rng = np.random.RandomState(42)
    
    tau_schedule = [5.0, 20.0, 100.0, 500.0, 2000.0, 5000.0]
    maxiter_schedule = [200, 300, 400, 500, 600, 800]
    edm_weight_schedule = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
    
    for r in range(n_restarts):
        if _elapsed() > _DEADLINE - 15:
            break
        # Random perturbation
        scale = 0.1 * (r + 1) / n_restarts
        P_try = P_init + rng.randn(N, D) * scale
        P_polished = _polish_with_smooth_objective(P_try, tau_schedule, maxiter_schedule, edm_weight_schedule)
        s = _score_from_P(P_polished)
        if np.isfinite(s) and s < best_score:
            best_score = s
            best_P = P_polished
    
    return best_P, best_score


def construct():
    global _t0
    _t0 = time.time()
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Tetrahedron + midpoints + face centroids
    P_tetra = _init_tetrahedron_midpoints()
    candidates.append(P_tetra)
    
    # 2. Fibonacci sphere
    P_fib = _fibonacci_sphere(N)
    candidates.append(P_fib)
    
    # 3. Random points on sphere with different seeds
    rng = np.random.default_rng(42)
    for seed in range(10):
        r = np.random.default_rng(seed)
        pts = r.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * 2.0
        candidates.append(pts)
        
        if _elapsed() > _DEADLINE - 40:
            break
    
    # 4. Random points in ball
    for seed in range(8):
        r = np.random.default_rng(100 + seed)
        pts = r.normal(size=(N, D)) * 2.0
        candidates.append(pts)
        
        if _elapsed() > _DEADLINE - 40:
            break
    
    # 5. Perturbed Fibonacci
    for _ in range(4):
        perturbed = P_fib + rng.normal(size=(N, D)) * 0.2
        candidates.append(perturbed)
    
    # 6. Perturbed tetrahedron
    for _ in range(4):
        perturbed = P_tetra + rng.normal(size=(N, D)) * 0.1
        candidates.append(perturbed)
    
    # Evaluate all candidates
    best_score = np.inf
    best_P = None
    
    for P_cand in candidates:
        s = _score_from_P(P_cand)
        if np.isfinite(s) and s < best_score:
            best_score = s
            best_P = P_cand.copy()
    
    if best_P is None:
        best_P = P_tetra
        best_score = _score_from_P(P_tetra)
    
    # Main optimization: smooth objective with increasing tau
    tau_schedule = [5.0, 20.0, 100.0, 500.0, 2000.0, 5000.0]
    maxiter_schedule = [300, 400, 500, 600, 800, 1000]
    edm_weight_schedule = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
    
    if _elapsed() < _DEADLINE - 30:
        P_polished = _polish_with_smooth_objective(best_P, tau_schedule, maxiter_schedule, edm_weight_schedule)
        s_polished = _score_from_P(P_polished)
        if np.isfinite(s_polished) and s_polished < best_score:
            best_score = s_polished
            best_P = P_polished
    
    # Bisection ratio search
    if _elapsed() < _DEADLINE - 25:
        P_bisect, s_bisect = _bisection_ratio_search(best_P, 1.0, best_score**0.5, n_bisections=15)
        if np.isfinite(s_bisect) and s_bisect < best_score:
            best_score = s_bisect
            best_P = P_bisect
    
    # Random restart polish
    if _elapsed() < _DEADLINE - 20:
        P_restart, s_restart = _random_restart_polish(best_P, n_restarts=4)
        if np.isfinite(s_restart) and s_restart < best_score:
            best_score = s_restart
            best_P = P_restart
    
    # Final high-precision polish
    if _elapsed() < _DEADLINE - 10:
        tau_final = [5000.0, 10000.0]
        maxiter_final = [500, 600]
        edm_final = [0.001, 0.0005]
        P_final = _polish_with_smooth_objective(best_P, tau_final, maxiter_final, edm_final)
        s_final = _score_from_P(P_final)
        if np.isfinite(s_final) and s_final < best_score:
            best_score = s_final
            best_P = P_final
    
    # Verify distinctness
    d2 = _pairwise_dists_sq(best_P)
    mask = ~np.eye(N, dtype=bool)
    if d2[mask].min() < 1e-10:
        # Points too close; add small jitter
        best_P += np.random.RandomState(0).randn(N, D) * 1e-6
    
    return best_P


if __name__ == '__main__':
    P = construct()
    print("Score:", _score_from_P(P))

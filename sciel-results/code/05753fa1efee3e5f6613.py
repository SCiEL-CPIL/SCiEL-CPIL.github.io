"""Optimized placement of 16 points in 2D to minimize (max/min pairwise distance)^2."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 16
D = 2


def _score(pts):
    """Compute (max pairwise dist / min pairwise dist)^2."""
    n = len(pts)
    # Compute all pairwise distances
    dx = pts[:, 0, None] - pts[:, None, 0]
    dy = pts[:, 1, None] - pts[:, None, 1]
    dists = np.sqrt(dx**2 + dy**2)
    # Exclude self-distances (diagonal)
    mask = ~np.eye(n, dtype=bool)
    d_flat = dists[mask]
    dmin = d_flat.min()
    dmax = d_flat.max()
    if dmin < 1e-15:
        return float('inf')
    return (dmax / dmin) ** 2


def _soft_ratio(pts, beta=50.0):
    """
    Smooth surrogate for (max/min)^2 using log-sum-exp.
    Returns a differentiable approximation.
    """
    n = len(pts)
    dx = pts[:, 0, None] - pts[:, None, 0]
    dy = pts[:, 1, None] - pts[:, None, 1]
    d2 = dx**2 + dy**2
    mask = ~np.eye(n, dtype=bool)
    d2_flat = d2[mask]
    
    # Soft max of d2 (squared distances) via log-sum-exp
    m = d2_flat.max()
    soft_max_d2 = m + np.log(np.sum(np.exp(d2_flat - m)))
    
    # Soft min of d2: use -log-sum-exp(-d2)
    mn = d2_flat.min()
    soft_min_d2 = mn - np.log(np.sum(np.exp(-(d2_flat - mn))))
    
    if soft_min_d2 < 1e-15:
        return 1e10
    return soft_max_d2 / soft_min_d2


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Initial configurations to try
    def make_grid():
        side = 4
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i, j])
        return np.array(pts, dtype=float)
    
    def make_hexagonal():
        """Hexagonal-like packing for 16 points."""
        pts = []
        # Row 0: 4 points
        for i in range(4):
            pts.append([i * 1.0, 0.0])
        # Row 1: 4 points, offset
        for i in range(4):
            pts.append([i * 1.0 + 0.5, np.sqrt(3) / 2])
        # Row 2: 4 points
        for i in range(4):
            pts.append([i * 1.0, np.sqrt(3)])
        # Row 3: 4 points, offset
        for i in range(4):
            pts.append([i * 1.0 + 0.5, 3 * np.sqrt(3) / 2])
        return np.array(pts, dtype=float)
    
    def make_circle():
        """Points on a circle."""
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    def make_concentric():
        """Two rings: 6 inner, 10 outer."""
        pts = []
        # Inner ring: 6 points
        for i in range(6):
            a = 2 * np.pi * i / 6
            pts.append([0.5 * np.cos(a), 0.5 * np.sin(a)])
        # Outer ring: 10 points
        for i in range(10):
            a = 2 * np.pi * i / 10 + np.pi / 10
            pts.append([1.5 * np.cos(a), 1.5 * np.sin(a)])
        return np.array(pts, dtype=float)
    
    def make_lattice_scaled():
        """4x4 grid with small perturbation."""
        pts = make_grid()
        rng = np.random.RandomState(42)
        pts = pts + rng.normal(0, 0.1, pts.shape)
        return pts
    
    initial_candidates = [
        make_grid(),
        make_hexagonal(),
        make_circle(),
        make_concentric(),
        make_lattice_scaled(),
    ]
    
    best_pts = None
    best_score = float('inf')
    
    def try_optimize(pts0, max_iter=500, tol=1e-8):
        """Optimize from initial point pts0 using L-BFGS-B on the smooth surrogate."""
        nonlocal best_pts, best_score
        
        def objective(x):
            pts = x.reshape(N, 2)
            return _soft_ratio(pts, beta=50.0)
        
        def gradient(x):
            pts = x.reshape(N, 2)
            n = N
            dx = pts[:, 0, None] - pts[:, None, 0]
            dy = pts[:, 1, None] - pts[:, None, 1]
            d2 = dx**2 + dy**2
            mask = ~np.eye(n, dtype=bool)
            
            # Soft max
            m = d2[mask].max()
            exp_vals = np.exp(d2 - m)
            exp_vals[~mask] = 0  # exclude diagonal
            sum_exp = exp_vals.sum()
            softmax_weights = exp_vals / sum_exp
            
            # Soft min
            mn = d2[mask].min()
            exp_neg = np.exp(-(d2 - mn))
            exp_neg[~mask] = 0
            sum_exp_neg = exp_neg.sum()
            softmin_weights = exp_neg / sum_exp_neg
            
            # Gradient of soft_max_d2 / soft_min_d2
            # d(soft_max)/d(x_ij) = 2 * sum_k softmax_weights[i,k] * (x_i - x_k)
            # d(soft_min)/d(x_ij) = -2 * sum_k softmin_weights[i,k] * (x_i - x_k)
            
            grad = np.zeros((n, 2))
            
            # For soft_max: sum over k of w_ik * 2*(x_i - x_k)
            # = 2 * (x_i * sum_k w_ik - sum_k w_ik * x_k)
            w_max_sum = softmax_weights.sum(axis=1)  # sum over k for each i
            wx_max = np.zeros((n, 2))
            for i in range(n):
                for k in range(n):
                    if i != k:
                        wx_max[i, 0] += softmax_weights[i, k] * pts[k, 0]
                        wx_max[i, 1] += softmax_weights[i, k] * pts[k, 1]
            
            d_softmax_dx = 2 * (pts * w_max_sum[:, None] - wx_max)
            
            w_min_sum = softmin_weights.sum(axis=1)
            wx_min = np.zeros((n, 2))
            for i in range(n):
                for k in range(n):
                    if i != k:
                        wx_min[i, 0] += softmin_weights[i, k] * pts[k, 0]
                        wx_min[i, 1] += softmin_weights[i, k] * pts[k, 1]
            
            d_softmin_dx = -2 * (pts * w_min_sum[:, None] - wx_min)
            
            soft_max_val = m + np.log(sum_exp)
            soft_min_val = mn - np.log(sum_exp_neg)
            
            ratio = soft_max_val / soft_min_val
            
            # d(ratio)/d(x) = (soft_min * d(soft_max) - soft_max * d(soft_min)) / soft_min^2
            grad = (soft_min_val * d_softmax_dx - soft_max_val * d_softmin_dx) / (soft_min_val ** 2)
            
            return grad.reshape(-1)
        
        x0 = pts0.flatten()
        
        try:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50}
            )
            
            pts_opt = result.x.reshape(N, 2)
            sc = _score(pts_opt)
            
            if sc < best_score:
                best_score = sc
                best_pts = pts_opt.copy()
        except Exception:
            pass
        
        return best_pts, best_score
    
    # Try each initial candidate
    for idx, pts0 in enumerate(initial_candidates):
        if time.time() > deadline - 30:
            break
        try_optimize(pts0)
    
    # If we have time, do a polish pass with tighter tolerance
    if best_pts is not None and time.time() < deadline - 20:
        x0 = best_pts.flatten()
        
        def objective_final(x):
            pts = x.reshape(N, 2)
            return _soft_ratio(pts, beta=100.0)
        
        def gradient_final(x):
            pts = x.reshape(N, 2)
            n = N
            dx = pts[:, 0, None] - pts[:, None, 0]
            dy = pts[:, 1, None] - pts[:, None, 1]
            d2 = dx**2 + dy**2
            mask = ~np.eye(n, dtype=bool)
            
            m = d2[mask].max()
            exp_vals = np.exp(d2 - m)
            exp_vals[~mask] = 0
            sum_exp = exp_vals.sum()
            softmax_weights = exp_vals / sum_exp
            
            mn = d2[mask].min()
            exp_neg = np.exp(-(d2 - mn))
            exp_neg[~mask] = 0
            sum_exp_neg = exp_neg.sum()
            softmin_weights = exp_neg / sum_exp_neg
            
            w_max_sum = softmax_weights.sum(axis=1)
            wx_max = np.zeros((n, 2))
            for i in range(n):
                for k in range(n):
                    if i != k:
                        wx_max[i, 0] += softmax_weights[i, k] * pts[k, 0]
                        wx_max[i, 1] += softmax_weights[i, k] * pts[k, 1]
            
            d_softmax_dx = 2 * (pts * w_max_sum[:, None] - wx_max)
            
            w_min_sum = softmin_weights.sum(axis=1)
            wx_min = np.zeros((n, 2))
            for i in range(n):
                for k in range(n):
                    if i != k:
                        wx_min[i, 0] += softmin_weights[i, k] * pts[k, 0]
                        wx_min[i, 1] += softmin_weights[i, k] * pts[k, 1]
            
            d_softmin_dx = -2 * (pts * w_min_sum[:, None] - wx_min)
            
            soft_max_val = m + np.log(sum_exp)
            soft_min_val = mn - np.log(sum_exp_neg)
            
            grad = (soft_min_val * d_softmax_dx - soft_max_val * d_softmin_dx) / (soft_min_val ** 2)
            
            return grad.reshape(-1)
        
        try:
            result = minimize(
                objective_final,
                x0,
                jac=gradient_final,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 100}
            )
            pts_polish = result.x.reshape(N, 2)
            sc = _score(pts_polish)
            if sc < best_score:
                best_score = sc
                best_pts = pts_polish.copy()
        except Exception:
            pass
    
    # Final fallback
    if best_pts is None:
        best_pts = make_grid()
    
    # Ensure all points are distinct
    # Remove any duplicates if they crept in
    seen = set()
    unique_pts = []
    for p in best_pts:
        key = (round(p[0], 10), round(p[1], 10))
        if key not in seen:
            seen.add(key)
            unique_pts.append(p)
    
    if len(unique_pts) < N:
        # Fallback to grid if duplicates exist
        best_pts = make_grid()
    else:
        best_pts = np.array(unique_pts[:N], dtype=float)
    
    return best_pts

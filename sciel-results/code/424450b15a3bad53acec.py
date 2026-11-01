"""Circle packing in unit square n=32 — smooth-penalty gradient optimization with restarts."""
import numpy as np
import time
from scipy.optimize import minimize

N = 32
TIME_LIMIT = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _objective_and_grad(params, beta):
    """
    Objective: maximize sum of radii - penalty for overlaps and boundary violations.
    params: array of shape (3*N,) = [x1,y1,r1, x2,y2,r2, ...]
    beta: penalty weight for constraint violations
    Returns (value, gradient) where value is to be MINIMIZED (negative sum r + penalties).
    """
    p = params.reshape(N, 3)
    xs, ys, rs = p[:, 0], p[:, 1], p[:, 2]
    
    # Sum of radii (we minimize negative)
    sum_r = np.sum(rs)
    
    # Boundary penalties: circles must be inside [0,1]^2
    # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    bx = np.maximum(0, rs - xs) + np.maximum(0, rs - (1 - xs))
    by = np.maximum(0, rs - ys) + np.maximum(0, rs - (1 - ys))
    boundary_violation = np.sum(bx) + np.sum(by)
    
    # Overlap penalties: for each pair i<j, distance >= ri + rj
    # Smooth penalty using soft-plus approximation
    # violation: (ri+rj) - d_ij > 0
    penalty = 0.0
    grad = np.zeros_like(params)
    
    # Compute pairwise distances
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    d2 = dx**2 + dy**2
    d = np.sqrt(np.maximum(d2, 1e-12))
    
    rs_sum = rs[:, None] + rs[None, :]
    
    # Violation amount for each pair
    viol = rs_sum - d  # positive if overlapping
    
    # Only consider pairs i < j
    idx = np.triu_indices(N, 1)
    viol_pairs = viol[idx]
    
    # Always compute overlap gradients (even if small violations)
    eps = 1e-6
    soft_viol = np.sqrt(viol_pairs**2 + eps**2) - eps  # ~ max(0, viol) smoothed
    penalty += beta * np.sum(soft_viol)
    
    # Gradient w.r.t. positions
    dv = viol_pairs / np.sqrt(viol_pairs**2 + eps**2)
    
    # Scatter back to full matrix (symmetric)
    dv_mat = np.zeros((N, N))
    dv_mat[idx] = dv
    dv_mat = dv_mat + dv_mat.T  # symmetric
    
    # d(viol)/d(xi) = -(xi - xj)/d = (xj - xi)/d
    # d(viol)/d(ri) = 1
    # Gradient of penalty w.r.t. xi: beta * sum_j dv_ij * (xj - xi)/d_ij
    w = dv_mat / np.maximum(d, 1e-12)
    np.fill_diagonal(w, 0)
    
    # grad_x[i] = beta * sum_j w_ij * (xj - xi) = beta * (sum_j w_ij * xj - xi * sum_j w_ij)
    grad_x = beta * (w @ xs - xs * np.sum(w, axis=1))
    grad_y = beta * (w @ ys - ys * np.sum(w, axis=1))
    
    # grad_r[i] = beta * sum_{j != i} dv_ij
    grad_r = beta * np.sum(dv_mat, axis=1)
    
    # Add boundary gradients
    # For x: if xi - ri < 0, penalty = (ri - xi), grad_x += -beta, grad_r += beta
    # if 1 - xi - ri < 0, penalty = (xi + ri - 1), grad_x += beta, grad_r += beta
    # Similarly for y
    
    # Boundary x-left: violation = max(0, ri - xi)
    mask_xl = rs > xs
    grad_x[mask_xl] -= beta
    grad_r[mask_xl] += beta
    
    # Boundary x-right: violation = max(0, xi + ri - 1)
    mask_xr = (xs + rs) > 1
    grad_x[mask_xr] += beta
    grad_r[mask_xr] += beta
    
    # Boundary y-left: violation = max(0, ri - yi)
    mask_yl = rs > ys
    grad_y[mask_yl] -= beta
    grad_r[mask_yl] += beta
    
    # Boundary y-right: violation = max(0, yi + ri - 1)
    mask_yr = (ys + rs) > 1
    grad_y[mask_yr] += beta
    grad_r[mask_yr] += beta
    
    # Total value to minimize: -sum_r + penalty
    value = -sum_r + penalty
    
    # Gradient: -1 for each r (from -sum_r) + gradient from penalty
    grad[0::3] = grad_x - 1.0  # x coords
    grad[1::3] = grad_y        # y coords  
    grad[2::3] = grad_r - 1.0  # r coords
    
    return value, grad


def _is_valid(p, tol=1e-9):
    """Check if configuration is valid (no overlaps, inside square)."""
    p = p.reshape(N, 3)
    xs, ys, rs = p[:, 0], p[:, 1], p[:, 2]
    
    # Boundary check
    if np.any(xs < -tol) or np.any(xs > 1 + tol) or np.any(ys < -tol) or np.any(ys > 1 + tol):
        return False
    if np.any(rs < -tol):
        return False
    if np.any(xs + rs > 1 + tol) or np.any(1 - xs + rs > 1 + tol):
        return False
    if np.any(ys + rs > 1 + tol) or np.any(1 - ys + rs > 1 + tol):
        return False
    
    # Overlap check
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    d2 = dx**2 + dy**2
    d = np.sqrt(np.maximum(d2, 0))
    rs_sum = rs[:, None] + rs[None, :]
    
    # For i != j, need d >= rs_sum
    idx = np.triu_indices(N, 1)
    if np.any(d[idx] < rs_sum[idx] - tol):
        return False
    
    return True


def _sum_radii(p):
    """Return sum of radii if valid, else -inf."""
    p = p.reshape(N, 3)
    if not _is_valid(p):
        return -np.inf
    return np.sum(p[:, 2])


def _generate_initial(rng, method='grid'):
    """Generate initial configuration."""
    if method == 'grid':
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side + rng.uniform(-0.1, 0.1, 1)[0] / side
                y = (j + 0.5) / side + rng.uniform(-0.1, 0.1, 1)[0] / side
                r = 0.45 / side + rng.uniform(0, 0.05, 1)[0] / side
                cells.append([x, y, r])
        return np.array(cells[:N]).flatten()
    elif method == 'random':
        # Random positions with small radii
        xs = rng.uniform(0.1, 0.9, N)
        ys = rng.uniform(0.1, 0.9, N)
        rs = rng.uniform(0.02, 0.08, N)
        p = np.column_stack([xs, ys, rs])
        return p.flatten()
    elif method == 'jittered':
        # Jittered grid with varying radii
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side + rng.uniform(-0.15, 0.15, 1)[0] / side
                y = (j + 0.5) / side + rng.uniform(-0.15, 0.15, 1)[0] / side
                r = rng.uniform(0.35, 0.55) / side
                cells.append([x, y, r])
        return np.array(cells[:N]).flatten()
    elif method == 'hexagonal':
        # Hexagonal packing initial
        cells = []
        rows = 6
        cols = 6
        for i in range(rows):
            for j in range(cols):
                x = (j + 0.5 + (i % 2) * 0.5) / cols
                y = (i + 0.5) / rows
                if 0 <= x <= 1 and 0 <= y <= 1:
                    r = 0.4 / max(rows, cols)
                    cells.append([x, y, r])
        # Shuffle and take N
        if len(cells) > N:
            indices = rng.choice(len(cells), N, replace=False)
            cells = [cells[i] for i in indices]
        while len(cells) < N:
            cells.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.05])
        return np.array(cells[:N]).flatten()
    else:
        # Fallback to jittered
        return _generate_initial(rng, 'jittered')


def _optimize_single(p0, beta_start, beta_end, max_iter, time_start, deadline):
    """Run optimization from initial point p0."""
    p = p0.copy()
    beta = beta_start
    
    # Adaptive beta: start high, decrease
    n_iter = 0
    bounds = []
    for k in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0, 0.5))  # r (max 0.5 for single circle)
    
    for i in range(max_iter):
        if time.time() - time_start > deadline:
            break
        
        # Anneal beta
        progress = i / max(max_iter - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** progress
        
        try:
            value, grad = _objective_and_grad(p, beta)
            
            res = minimize(
                lambda x: _objective_and_grad(x, beta)[0],
                p,
                jac=lambda x: _objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            p = res.x
            n_iter += 1
        except Exception:
            break
    
    return p


def _polish(p, time_start, deadline, high_precision=False):
    """Final high-precision polish pass."""
    p = p.copy()
    beta = 100.0 if not high_precision else 500.0
    bounds = []
    for k in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    max_polish_iters = 15 if not high_precision else 25
    
    for _ in range(max_polish_iters):
        if time.time() - time_start > deadline:
            break
        try:
            value, grad = _objective_and_grad(p, beta)
            res = minimize(
                lambda x: _objective_and_grad(x, beta)[0],
                p,
                jac=lambda x: _objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100 if not high_precision else 200, 
                        'ftol': 1e-15 if not high_precision else 1e-18,
                        'gtol': 1e-10 if not high_precision else 1e-12}
            )
            p = res.x
        except Exception:
            break
    
    return p


def construct():
    time_start = time.time()
    deadline = time_start + TIME_LIMIT - 10.0  # 10s safety margin
    
    best_p = None
    best_score = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different methods
    methods = ['grid', 'jittered', 'random', 'grid', 'jittered', 'grid', 'jittered', 'random',
               'hexagonal', 'grid', 'jittered', 'hexagonal', 'random', 'grid', 'jittered']
    
    for restart_idx, method in enumerate(methods):
        if time.time() - time_start > deadline:
            break
        
        p0 = _generate_initial(rng, method)
        
        # Use different beta schedules for diversity
        if restart_idx % 3 == 0:
            beta_start, beta_end, max_iter = 50.0, 5.0, 20
        elif restart_idx % 3 == 1:
            beta_start, beta_end, max_iter = 100.0, 10.0, 15
        else:
            beta_start, beta_end, max_iter = 30.0, 3.0, 25
        
        p_opt = _optimize_single(p0, beta_start, beta_end, 
                                  max_iter, time_start, deadline)
        
        # Check validity and score
        score = _sum_radii(p_opt)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_p = p_opt.copy()
        
        # Also try polishing if we have time
        remaining = deadline - (time.time() - time_start)
        if remaining > 15:
            p_polished = _polish(p_opt, time_start, deadline)
            score_pol = _sum_radii(p_polished)
            if np.isfinite(score_pol) and score_pol > best_score:
                best_score = score_pol
                best_p = p_polished.copy()
    
    # Final polish on best with high precision
    if best_p is not None and time.time() - time_start < deadline - 5:
        p_final = _polish(best_p, time_start, deadline, high_precision=True)
        score_final = _sum_radii(p_final)
        if np.isfinite(score_final) and score_final > best_score:
            best_score = score_final
            best_p = p_final.copy()
    
    if best_p is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_p = np.array(cells[:N])
    
    return best_p.reshape(N, 3)
# EVOLVE-BLOCK-END

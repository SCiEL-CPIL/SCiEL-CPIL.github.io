import numpy as np
import time
import scipy.optimize as opt

# Pre-compute the index maps for projections
x_indices = np.arange(10)
y_indices = np.arange(10)
X_grid, Y_grid = np.meshgrid(x_indices, y_indices, indexing='xy')

D_vals = (X_grid - Y_grid)
H0_vals = X_grid
H1_vals = (X_grid + Y_grid)
H2_vals = (X_grid + 2 * Y_grid)

D_uniq, D_inv = np.unique(D_vals, return_inverse=True)
H0_uniq, H0_inv = np.unique(H0_vals, return_inverse=True)
H1_uniq, H1_inv = np.unique(H1_vals, return_inverse=True)
H2_uniq, H2_inv = np.unique(H2_vals, return_inverse=True)

D_inv_flat = D_inv.ravel()
H0_inv_flat = H0_inv.ravel()
H1_inv_flat = H1_inv.ravel()
H2_inv_flat = H2_inv.ravel()

D_bins = len(D_uniq)
H0_bins = len(H0_uniq)
H1_bins = len(H1_uniq)
H2_bins = len(H2_uniq)

def get_projection_entropy_and_grad(P, inv_flat, n_bins):
    q = np.bincount(inv_flat, weights=P, minlength=n_bins)
    mask = q > 0
    q_nonzero = q[mask]
    if len(q_nonzero) == 0:
        return 0.0, np.zeros_like(P)
    H = -np.sum(q_nonzero * np.log2(q_nonzero))
    grad_q = np.zeros(n_bins)
    grad_q[mask] = -np.log2(q_nonzero) - 1.0 / np.log(2.0)
    grad_P = grad_q[inv_flat]
    return H, grad_P

def objective_and_grad(z, beta=50.0):
    return _soft_max_obj(z, beta)
    P = z
    D_val, D_grad = get_projection_entropy_and_grad(P, D_inv_flat, D_bins)
    H0_val, H0_grad = get_projection_entropy_and_grad(P, H0_inv_flat, H0_bins)
    H1_val, H1_grad = get_projection_entropy_and_grad(P, H1_inv_flat, H1_bins)
    H2_val, H2_grad = get_projection_entropy_and_grad(P, H2_inv_flat, H2_bins)
    
    H012 = np.array([H0_val, H1_val, H2_val])
    grads_012 = [H0_grad, H1_grad, H2_grad]
    
    # Smooth soft-max using log-sum-exp
    h_max = np.max(H012)
    exps = np.exp(beta * (H012 - h_max))
    M = h_max + np.log(np.sum(exps)) / beta
    
    if M < 1e-9:
        return -1e9, np.zeros_like(P)
    
    ratio = D_val / M
    # Gradient of soft-max: M_grad = sum_i (exp_i / sum_exp) * grad_i
    weights = exps / np.sum(exps)
    M_grad = np.zeros_like(P)
    for i in range(3):
        M_grad += weights[i] * grads_012[i]
    
    grad_ratio = (M * D_grad - D_val * M_grad) / (M * M)
    
    obj = -ratio
    grad = -grad_ratio
    return obj, grad

def score_P(P):
    P_flat = P.ravel()
    D_val, _ = get_projection_entropy_and_grad(P_flat, D_inv_flat, D_bins)
    H0_val, _ = get_projection_entropy_and_grad(P_flat, H0_inv_flat, H0_bins)
    H1_val, _ = get_projection_entropy_and_grad(P_flat, H1_inv_flat, H1_bins)
    H2_val, _ = get_projection_entropy_and_grad(P_flat, H2_inv_flat, H2_bins)
    M = max(H0_val, H1_val, H2_val)
    if M < 1e-9:
        return 0.0
    return D_val / M

def objective_and_grad_exact(z):
    """Exact (non-smooth) objective for final polish."""
    P = z
    D_val, D_grad = get_projection_entropy_and_grad(P, D_inv_flat, D_bins)
    H0_val, H0_grad = get_projection_entropy_and_grad(P, H0_inv_flat, H0_bins)
    H1_val, H1_grad = get_projection_entropy_and_grad(P, H1_inv_flat, H1_bins)
    H2_val, H2_grad = get_projection_entropy_and_grad(P, H2_inv_flat, H2_bins)
    
    H012 = np.array([H0_val, H1_val, H2_val])
    grads_012 = [H0_grad, H1_grad, H2_grad]
    M = np.max(H012)
    
    if M < 1e-9:
        return -1e9, np.zeros_like(P)
    
    ratio = D_val / M
    idx_max = np.argmax(H012)
    M_grad = grads_012[idx_max]
    grad_ratio = (M * D_grad - D_val * M_grad) / (M * M)
    
    obj = -ratio
    grad = -grad_ratio
    return obj, grad

def _soft_max_obj(z, beta):
    """Objective with specified beta for soft-max."""
    P = z
    D_val, D_grad = get_projection_entropy_and_grad(P, D_inv_flat, D_bins)
    H0_val, H0_grad = get_projection_entropy_and_grad(P, H0_inv_flat, H0_bins)
    H1_val, H1_grad = get_projection_entropy_and_grad(P, H1_inv_flat, H1_bins)
    H2_val, H2_grad = get_projection_entropy_and_grad(P, H2_inv_flat, H2_bins)
    
    H012 = np.array([H0_val, H1_val, H2_val])
    grads_012 = [H0_grad, H1_grad, H2_grad]
    
    h_max = np.max(H012)
    exps = np.exp(beta * (H012 - h_max))
    M = h_max + np.log(np.sum(exps)) / beta
    
    if M < 1e-9:
        return -1e9, np.zeros_like(P)
    
    ratio = D_val / M
    weights = exps / np.sum(exps)
    M_grad = np.zeros_like(P)
    for i in range(3):
        M_grad += weights[i] * grads_012[i]
    
    grad_ratio = (M * D_grad - D_val * M_grad) / (M * M)
    
    obj = -ratio
    grad = -grad_ratio
    return obj, grad

def _soft_max_obj_partial(args):
    z, beta = args
    return _soft_max_obj(z, beta)

def _coordinate_descent_polish(P, max_iters=50, tol=1e-10):
    """Simple coordinate descent to polish the solution."""
    n = 10
    P = P.copy().ravel()
    current_score = score_P(P.reshape(n, n))
    
    for it in range(max_iters):
        improved = False
        for idx in range(n * n):
            old_val = P[idx]
            # Try small perturbations
            for delta in [0.01, 0.001, 0.0001, -0.0001, -0.001, -0.01]:
                new_val = old_val + delta
                if new_val < 0 or new_val > 1:
                    continue
                P[idx] = new_val
                s = P.sum()
                if s > 1e-9:
                    P_temp = P / s
                    sc = score_P(P_temp.reshape(n, n))
                    if sc > current_score + tol:
                        current_score = sc
                        improved = True
                        P[idx] = new_val
                    else:
                        P[idx] = old_val
        if not improved:
            break
    
    P = np.clip(P, 0, None)
    s = P.sum()
    if s > 0:
        P /= s
    return P.reshape(n, n)

def solve_kakeya():
    n = 10
    start_time = time.time()
    timeout = 600
    safety_margin = 30
    
    best_P = np.ones((n, n)) / (n * n)
    best_score = score_P(best_P)
    
    inits = [np.ones((n, n)) / (n * n)]
    rng = np.random.default_rng(42)
    for _ in range(10):
        p = rng.random((n, n))
        p /= p.sum()
        inits.append(p)
    
    # Heuristic inits: try to spread x-y while concentrating x
    # Concentrate on a diagonal band
    for c in range(-4, 5):
        p = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if j - i == c:
                    p[i, j] = 1.0
        if p.sum() > 0:
            p /= p.sum()
            inits.append(p)
    
    # Try 2D Gaussian-like inits
    for cx in [2, 5, 8]:
        for cy in [2, 5, 8]:
            p = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    p[i, j] = np.exp(-((i - cy)**2 + (j - cx)**2) / 4.0)
            p /= p.sum()
            inits.append(p)
    
    # Add more structured inits: anti-diagonals to maximize D
    for c in range(-4, 5):
        p = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i + j == c + 9:  # anti-diagonal
                    p[i, j] = 1.0
        if p.sum() > 0:
            p /= p.sum()
            inits.append(p)
    
    # Bimodal inits: mass on two corners
    for (x1, y1) in [(0,0), (0,9), (9,0), (9,9)]:
        for (x2, y2) in [(0,0), (0,9), (9,0), (9,9)]:
            if (x1,y1) != (x2,y2):
                p = np.zeros((n, n))
                p[y1, x1] = 0.5
                p[y2, x2] = 0.5
                inits.append(p)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                    'jac': lambda x: np.ones_like(x)}]
    bounds = [(0.0, 1.0) for _ in range(n * n)]
    
    # Phase 1: Global search with lower beta (smoother landscape)
    beta_global = 10.0
    for P_init in inits:
        if time.time() - start_time > timeout - safety_margin:
            break
        
        P_flat = P_init.ravel()
        
        # SLSQP with smooth objective (lower beta)
        try:
            result = opt.minimize(
                _soft_max_obj,
                (P_flat, beta_global),
                args=(beta_global,),
                jac=lambda z, b=beta_global: _soft_max_obj(z, b)[1],
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            P_opt = np.clip(result.x, 0, None).reshape(n, n)
            s = P_opt.sum()
            if s > 1e-9:
                P_opt /= s
                sc = score_P(P_opt)
                if sc > best_score:
                    best_score = sc
                    best_P = P_opt.copy()
        except Exception:
            pass
        
        if time.time() - start_time > timeout - safety_margin:
            break
    
    # Phase 2: Local refinement with higher beta (sharper)
    beta_local = 50.0
    P_flat = best_P.ravel()
    try:
        result = opt.minimize(
            _soft_max_obj,
            (P_flat, beta_local),
            args=(beta_local,),
            jac=lambda z, b=beta_local: _soft_max_obj(z, b)[1],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
        )
        P_opt = np.clip(result.x, 0, None).reshape(n, n)
        s = P_opt.sum()
        if s > 1e-9:
            P_opt /= s
            sc = score_P(P_opt)
            if sc > best_score:
                best_score = sc
                best_P = P_opt.copy()
    except Exception:
        pass
    
    # Phase 3: Very high beta polish
    if time.time() - start_time < timeout - safety_margin:
        beta_high = 200.0
        P_flat = best_P.ravel()
        try:
            result = opt.minimize(
                _soft_max_obj,
                (P_flat, beta_high),
                args=(beta_high,),
                jac=lambda z, b=beta_high: _soft_max_obj(z, b)[1],
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
            )
            P_opt = np.clip(result.x, 0, None).reshape(n, n)
            s = P_opt.sum()
            if s > 1e-9:
                P_opt /= s
                sc = score_P(P_opt)
                if sc > best_score:
                    best_score = sc
                    best_P = P_opt.copy()
        except Exception:
            pass
    
    # Phase 4: Exact objective polish
    if time.time() - start_time < timeout - safety_margin:
        P_flat = best_P.ravel()
        try:
            result = opt.minimize(
                objective_and_grad_exact,
                P_flat,
                jac=True,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-15, 'disp': False}
            )
            P_opt = np.clip(result.x, 0, None).reshape(n, n)
            s = P_opt.sum()
            if s > 1e-9:
                P_opt /= s
                sc = score_P(P_opt)
                if sc > best_score:
                    best_score = sc
                    best_P = P_opt.copy()
        except Exception:
            pass
    
    # Phase 5: Coordinate descent polish
    if time.time() - start_time < timeout - safety_margin:
        P_cd = _coordinate_descent_polish(best_P, max_iters=30, tol=1e-12)
        sc_cd = score_P(P_cd)
        if sc_cd > best_score:
            best_score = sc_cd
            best_P = P_cd.copy()
    
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 0:
        best_P /= s
    
    return best_P

def arithmetic_kakeya_distribution() -> np.ndarray:
    return solve_kakeya()

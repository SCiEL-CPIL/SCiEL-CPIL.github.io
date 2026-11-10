import numpy as np
import time
import scipy.optimize as opt
import itertools

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

def get_projection_entropy(P, inv_flat, n_bins):
    q = np.bincount(inv_flat, weights=P, minlength=n_bins)
    mask = q > 0
    if np.sum(mask) == 0:
        return 0.0
    q_nonzero = q[mask]
    return -np.sum(q_nonzero * np.log2(q_nonzero))

def get_projection_entropy_grad(P, inv_flat, n_bins):
    q = np.bincount(inv_flat, weights=P, minlength=n_bins)
    mask = q > 0
    if np.sum(mask) == 0:
        return 0.0, np.zeros_like(P)
    q_nonzero = q[mask]
    H = -np.sum(q_nonzero * np.log2(q_nonzero))
    grad_q = np.zeros(n_bins)
    grad_q[mask] = -np.log2(q_nonzero) - 1.0 / np.log(2.0)
    grad_P = grad_q[inv_flat]
    return H, grad_P

def score_P(P):
    P_flat = P.ravel()
    D_val = get_projection_entropy(P_flat, D_inv_flat, D_bins)
    H0_val = get_projection_entropy(P_flat, H0_inv_flat, H0_bins)
    H1_val = get_projection_entropy(P_flat, H1_inv_flat, H1_bins)
    H2_val = get_projection_entropy(P_flat, H2_inv_flat, H2_bins)
    M = max(H0_val, H1_val, H2_val)
    if M < 1e-9:
        return 0.0
    return D_val / M

def objective_and_grad(z, beta=50.0):
    """
    Objective with soft-max smoothing for the max operation.
    """
    P = z
    D_val, D_grad = get_projection_entropy_grad(P, D_inv_flat, D_bins)
    H0_val, H0_grad = get_projection_entropy_grad(P, H0_inv_flat, H0_bins)
    H1_val, H1_grad = get_projection_entropy_grad(P, H1_inv_flat, H1_bins)
    H2_val, H2_grad = get_projection_entropy_grad(P, H2_inv_flat, H2_bins)
    
    H012 = np.array([H0_val, H1_val, H2_val])
    grads_012 = [H0_grad, H1_grad, H2_grad]
    
    # Soft-max: M_soft = (1/beta) * log(sum(exp(beta * Hi)))
    scaled_H = beta * H012
    max_scaled = np.max(scaled_H)
    exp_scaled = np.exp(scaled_H - max_scaled)
    sum_exp = np.sum(exp_scaled)
    
    if sum_exp < 1e-300:
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
    
    weights = exp_scaled / sum_exp
    M = (max_scaled + np.log(sum_exp)) / beta
    
    if M < 1e-9:
        return -1e9, np.zeros_like(P)
    
    # Gradient of M_soft with respect to P
    M_grad = np.zeros_like(P)
    for i in range(3):
        M_grad += weights[i] * grads_012[i]
    
    ratio = D_val / M
    grad_ratio = (M * D_grad - D_val * M_grad) / (M * M)
    
    obj = -ratio
    grad = -grad_ratio
    return obj, grad

def gmm_to_P(params, n_components, n=10):
    """
    Convert GMM parameters to a 10x10 probability distribution.
    params structure:
    - weights: n_components
    - means: n_components * 2 (y, x)
    - covs: n_components * 3 (var_y, cov_yx, var_x) - using simplified diagonal+offdiag for 2x2
    """
    weights = params[:n_components]
    means = params[n_components : n_components + 2*n_components].reshape(n_components, 2)
    cov_params = params[n_components + 2*n_components : n_components + 3*n_components].reshape(n_components, 3)
    
    # Softmax weights to ensure they are positive and sum to 1
    exp_w = np.exp(weights - np.max(weights))
    weights = exp_w / np.sum(exp_w)
    
    # Ensure covariance is positive semi-definite
    # We parameterize as: var_y = exp(cov_params[0]), var_x = exp(cov_params[2]), rho = tanh(cov_params[1])
    # Cov matrix = [[var_y, rho*sqrt(var_y*var_x)], [rho*sqrt(var_y*var_x), var_x]]
    P = np.zeros((n, n))
    
    for k in range(n_components):
        w = weights[k]
        if w < 1e-6: continue
        
        my, mx = means[k]
        vy = np.exp(cov_params[k, 0])
        vx = np.exp(cov_params[k, 2])
        rho = np.tanh(cov_params[k, 1])
        
        # Clip mean and variance to reasonable ranges to avoid numerical issues
        my = np.clip(my, 0, 9)
        mx = np.clip(mx, 0, 9)
        vy = np.clip(vy, 0.01, 100)
        vx = np.clip(vx, 0.01, 100)
        
        # Calculate 2D Gaussian PDF
        # Note: We are working on a grid of integers 0..9
        # We can evaluate the continuous Gaussian at these integer points
        # and then normalize.
        
        # Efficient calculation for small grid
        Y, X = np.meshgrid(np.arange(n), np.arange(n), indexing='xy')
        
        dy = Y - my
        dx = X - mx
        
        # Inverse covariance determinant and matrix
        det = vy * vx * (1 - rho**2)
        if det < 1e-9:
            det = 1e-9
        inv_det = 1.0 / np.sqrt(2 * np.pi * det)
        
        # Exponent term: -0.5 * [ (vy*dx^2 - 2*rho*sqrt(vy*vx)*dx*dy + vx*dy^2) / (vy*vx*(1-rho^2)) ]
        sqrt_vy_vx = np.sqrt(vy * vx)
        numerator = vy * dx**2 - 2 * rho * sqrt_vy_vx * dx * dy + vx * dy**2
        denominator = vy * vx * (1 - rho**2)
        
        exponent = -0.5 * numerator / denominator
        pdf = inv_det * np.exp(exponent)
        
        P += w * pdf
    
    # Normalize
    s = P.sum()
    if s > 1e-9:
        P /= s
    else:
        P = np.ones((n, n)) / (n*n)
        
    return P

def solve_kakeya():
    n = 10
    start_time = time.time()
    timeout = 600
    safety_margin = 30
    
    best_P = np.ones((n, n)) / (n * n)
    best_score = score_P(best_P)
    
    inits = []
    rng = np.random.default_rng(42)
    
    # Uniform
    inits.append(np.ones((n, n)) / (n * n))
    
    # Random inits
    for _ in range(10):
        p = rng.random((n, n))
        p /= p.sum()
        inits.append(p)
    
    # Structured inits: Corners
    for w1, w2, w3, w4 in [(1,1,1,1), (1,2,2,1), (2,1,1,2), (1,3,3,1)]:
        p = np.zeros((n, n))
        p[0, 0] = w1
        p[0, 9] = w2
        p[9, 0] = w3
        p[9, 9] = w4
        p /= p.sum()
        inits.append(p)
        
    # Sparse cells
    for cells in [
        [(0,0), (9,9)],
        [(0,9), (9,0)],
        [(2,2), (7,7), (2,7), (7,2)],
    ]:
        p = np.zeros((n, n))
        for (i, j) in cells:
            p[i, j] = 1.0
        p /= p.sum()
        inits.append(p)

    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                    'jac': lambda x: np.ones_like(x)}]
    bounds = [(0.0, 1.0) for _ in range(n * n)]
    
    def try_optimize_100d(P_flat, method='SLSQP', maxiter=200, ftol=1e-12, beta=50.0):
        nonlocal best_P, best_score
        try:
            if method == 'SLSQP':
                def obj_wrapper(x):
                    return objective_and_grad(x, beta=beta)
                result = opt.minimize(
                    obj_wrapper,
                    P_flat,
                    jac=True,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
                )
            else:
                return None
            
            P_opt = np.clip(result.x, 0, None).reshape(n, n)
            s = P_opt.sum()
            if s > 1e-9:
                P_opt /= s
                sc = score_P(P_opt)
                if sc > best_score:
                    best_score = sc
                    best_P = P_opt.copy()
                return P_opt
        except Exception:
            pass
        return None

    # Phase 1: Standard Optimization on diverse inits
    for P_init in inits:
        if time.time() - start_time > timeout - safety_margin:
            break
        P_flat = P_init.ravel()
        try_optimize_100d(P_flat, beta=30.0)
        
    # Phase 2: GMM-based Search
    # We optimize the GMM parameters to find good multi-modal structures
    n_components = 4
    
    def gmm_objective(params):
        if time.time() - start_time > timeout - safety_margin:
            return 1e9
        P = gmm_to_P(params, n_components, n)
        sc = score_P(P)
        return -sc

    # Initialize GMM parameters
    # weights (4), means (8), covs (12) -> total 24 params
    for _ in range(20):
        if time.time() - start_time > timeout - safety_margin:
            break
            
        # Random GMM init
        w_init = rng.normal(0, 1, n_components)
        m_init = rng.uniform(0, 9, (n_components, 2))
        c_init = np.log(rng.uniform(0.5, 5.0, n_components)) # log var
        rho_init = rng.uniform(-0.5, 0.5, n_components)
        
        params_init = np.concatenate([w_init, m_init.ravel(), np.column_stack([c_init, rho_init, c_init]).ravel()])
        
        try:
            res = opt.minimize(
                gmm_objective,
                params_init,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            P_gmm = gmm_to_P(res.x, n_components, n)
            sc = score_P(P_gmm)
            if sc > best_score:
                best_score = sc
                best_P = P_gmm.copy()
            # Refine this GMM result with 100d SLSQP
            try_optimize_100d(P_gmm.ravel(), beta=50.0)
            
        except Exception:
            pass

    # Phase 3: Local Search with Perturbations on best_P
    if time.time() - start_time < timeout - safety_margin:
        for _ in range(30):
            if time.time() - start_time > timeout - safety_margin:
                break
            pert_size = 0.05
            perturbation = rng.random((n, n)) * pert_size
            P_pert = best_P + perturbation
            P_pert = np.clip(P_pert, 0, None)
            s = P_pert.sum()
            if s > 0:
                P_pert /= s
                try_optimize_100d(P_pert.ravel(), beta=100.0, maxiter=300, ftol=1e-13)

    # Phase 4: High-precision polish
    if time.time() - start_time < timeout - safety_margin:
        P_flat = best_P.ravel()
        try_optimize_100d(P_flat, beta=200.0, maxiter=500, ftol=1e-14)
        
        # Try trust-constr if time permits
        if time.time() - start_time < timeout - safety_margin - 30:
            from scipy.optimize import LinearConstraint
            lc = LinearConstraint(np.ones((1, 100)), 1.0, 1.0)
            try:
                def obj_wrapper(x):
                    return objective_and_grad(x, beta=200.0)
                result = opt.minimize(
                    obj_wrapper,
                    best_P.ravel(),
                    jac=True,
                    method='trust-constr',
                    bounds=bounds,
                    constraints=[lc],
                    options={'maxiter': 200, 'gtol': 1e-10, 'disp': False}
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

    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 0:
        best_P /= s
    
    return best_P

def arithmetic_kakeya_distribution() -> np.ndarray:
    return solve_kakeya()

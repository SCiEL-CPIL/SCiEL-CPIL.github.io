import numpy as np
import time
import scipy.optimize as opt
from scipy.special import logsumexp

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

def objective_and_grad(z, beta=100.0):
    """Objective with soft-max smoothing for the max operation.
    
    Uses log-sum-exp to create a smooth approximation of max(H0, H1, H2).
    M_soft = (1/beta) * log(sum(exp(beta * Hi)))
    Gradient: dM_soft/dP = sum_i (exp(beta*Hi)/sum(exp(beta*Hj))) * dHi/dP
    """
    P = z
    D_val, D_grad = get_projection_entropy_grad(P, D_inv_flat, D_bins)
    H0_val, H0_grad = get_projection_entropy_grad(P, H0_inv_flat, H0_bins)
    H1_val, H1_grad = get_projection_entropy_grad(P, H1_inv_flat, H1_bins)
    H2_val, H2_grad = get_projection_entropy_grad(P, H2_inv_flat, H2_bins)
    
    H012 = np.array([H0_val, H1_val, H2_val])
    grads_012 = [H0_grad, H1_grad, H2_grad]
    
    # Soft-max calculation
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

# --- Product-form helpers for fast pre-conditioning ---
def _build_projection_maps(proj_vals):
    maps = {}
    for y in range(10):
        for x in range(10):
            v = proj_vals[y, x]
            if v not in maps:
                maps[v] = {}
            if y not in maps[v]:
                maps[v][y] = []
            maps[v][y].append(x)
    return maps

D_map = _build_projection_maps(D_vals)
H0_map = _build_projection_maps(H0_vals)
H1_map = _build_projection_maps(H1_vals)
H2_map = _build_projection_maps(H2_vals)

def _proj_entropy_from_RC(R, C, pmap, n_bins):
    q = np.zeros(n_bins)
    for v in range(n_bins):
        if v not in pmap:
            continue
        for y, xs in pmap[v].items():
            q[v] += R[y] * np.sum(C[xs])
    mask = q > 1e-15
    if not np.any(mask):
        return 0.0
    qn = q[mask]
    H = -np.sum(qn * np.log2(qn))
    return H

def _product_score(R, C):
    R = np.clip(R, 0, None)
    C = np.clip(C, 0, None)
    Rs = R.sum()
    Cs = C.sum()
    if Rs < 1e-15 or Cs < 1e-15:
        return 0.0
    R /= Rs
    C /= Cs
    D_val = _proj_entropy_from_RC(R, C, D_map, D_bins)
    H0_val = _proj_entropy_from_RC(R, C, H0_map, H0_bins)
    H1_val = _proj_entropy_from_RC(R, C, H1_map, H1_bins)
    H2_val = _proj_entropy_from_RC(R, C, H2_map, H2_bins)
    M = max(H0_val, H1_val, H2_val)
    if M < 1e-9:
        return 0.0
    return D_val / M

def solve_kakeya():
    n = 10
    start_time = time.time()
    timeout = 630
    safety_margin = 30
    
    # Initialize best with uniform
    best_P = np.ones((n, n)) / (n * n)
    best_score = score_P(best_P)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Uniform
    inits.append(best_P.copy())
    
    # 2. Random
    for _ in range(20):
        p = rng.random((n, n))
        p /= p.sum()
        inits.append(p)
    
    # 3. Diagonal (x-y = c)
    for c in range(-9, 10):
        p = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if j - i == c:
                    p[i, j] = 1.0
        if p.sum() > 0:
            p /= p.sum()
            inits.append(p)
            
    # 4. Anti-diagonal (x+y = c)
    for c in range(0, 19):
        p = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i + j == c:
                    p[i, j] = 1.0
        if p.sum() > 0:
            p /= p.sum()
            inits.append(p)
            
    # 5. Corners and sparse sets
    for cells in [
        [(0,0), (9,9)],
        [(0,9), (9,0)],
        [(0,0), (9,0), (0,9)],
        [(0,0), (9,9), (0,9)],
        [(0,0), (9,9), (9,0)],
        [(2,2), (7,7), (2,7), (7,2)],
        [(1,1), (8,8), (1,8), (8,1)],
        [(0,0), (4,4), (8,8), (0,8), (4,0), (8,0)],
        [(0,0), (0,9), (9,0), (9,9)],
    ]:
        p = np.zeros((n, n))
        for (i, j) in cells:
            p[i, j] = 1.0
        p /= p.sum()
        inits.append(p)
        
    # 6. Product-form inits (Rank 1)
    # Try to find good rank-1 approximations quickly
    prod_inits = [np.ones(n)/n]
    # Anisotropic Gaussians
    for sigma_x in [1.0, 2.0, 3.0]:
        for sigma_y in [0.5, 1.0, 2.0]:
            R = np.exp(-0.5 * ((np.arange(n) - 4.5) / sigma_x)**2)
            C = np.exp(-0.5 * ((np.arange(n) - 4.5) / sigma_y)**2)
            R /= R.sum()
            C /= C.sum()
            prod_inits.append(R)
            prod_inits.append(C)
            
    # Optimize a few product forms to get good starting points for full space
    # This is a heuristic to find "structured" points that aren't just random
    # We'll just pick the best few product scores and convert to full P
    best_prod_P = None
    best_prod_score = -1
    
    for i in range(0, len(prod_inits), 2):
        R = prod_inits[i]
        C = prod_inits[i+1]
        P = np.outer(R, C)
        s = _product_score(R, C)
        if s > best_prod_score:
            best_prod_score = s
            best_prod_P = P
    
    if best_prod_P is not None:
        inits.append(best_prod_P)
    
    # Optimization Loop
    # We use L-BFGS-B on the full 100-dim space with the soft-max objective.
    # Constraints: sum(P) = 1, P >= 0.
    # L-BFGS-B handles bounds well. We handle sum=1 by normalizing after each step 
    # or using a penalty? No, better to use SLSQP for equality constraints or 
    # just normalize if we use a parameterization that ensures sum=1.
    # However, L-BFGS-B with bounds [0, 1] and no equality constraint might drift sum != 1.
    # The objective function assumes P is a distribution? 
    # Wait, get_projection_entropy uses weights=P. If sum(P) != 1, the entropy is scaled.
    # H(p) for p not summing to 1 is not standard. 
    # Standard entropy assumes sum p = 1.
    # The task says "P is normalized to sum 1".
    # So we must enforce sum(P)=1.
    # L-BFGS-B does not support equality constraints.
    # We can use SLSQP which supports both bounds and equality constraints.
    # Or we can parameterize P via softmax of z? 
    # If P = softmax(z), then sum(P)=1 is automatic and P>0.
    # Let's try optimizing in z-space where P = softmax(z).
    
    def softmax(z):
        z_max = np.max(z)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z)
    
    def objective_z(z, beta=100.0):
        P = softmax(z)
        val, grad_P = objective_and_grad(P, beta)
        # Chain rule: dP/dz
        # P_i = exp(z_i) / sum(exp(z))
        # dP_i/dz_j = P_i (delta_ij - P_j)
        # grad_z = grad_P @ (diag(P) - P P^T)
        # grad_z_j = sum_i grad_P_i P_i (delta_ij - P_j)
        #           = P_j (sum_i grad_P_i P_i - sum_i grad_P_i P_i P_j)
        #           = P_j (sum_i grad_P_i P_i - P_j sum_i grad_P_i P_i)
        #           = P_j (1 - P_j) sum_i grad_P_i P_i
        # Wait, that's for sum P_i grad_P_i? No.
        # Let S = sum_i grad_P_i P_i.
        # grad_z_j = P_j (S - P_j S) = P_j (1-P_j) S.
        # This is O(N^2). N=100. 10000 ops. Fine.
        
        S = np.dot(grad_P, P)
        grad_z = P * (1 - P) * S
        return val, grad_z

    bounds_z = [(-100, 100)] * (n*n)
    
    def try_optimize_full(P_init, maxiter=1000, beta=100.0):
        nonlocal best_P, best_score
        # Convert P_init to z_init
        # P = softmax(z) => z = log(P) + constant. log(P) is fine.
        # Handle zeros in P_init
        P_safe = np.clip(P_init, 1e-15, None)
        z_init = np.log(P_safe)
        
        try:
            result = opt.minimize(
                objective_z,
                z_init,
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds_z,
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False}
            )
            P_opt = softmax(result.x)
            sc = score_P(P_opt)
            if sc > best_score:
                best_score = sc
                best_P = P_opt.copy()
            return P_opt
        except Exception:
            pass
        return None

    # Run optimization from diverse inits
    for P_init in inits:
        if time.time() - start_time > timeout - safety_margin:
            break
        try_optimize_full(P_init, maxiter=500, beta=50.0)
        
    # Refine best with sharper beta
    if time.time() - start_time < timeout - safety_margin:
        try_optimize_full(best_P, maxiter=1000, beta=100.0)
        try_optimize_full(best_P, maxiter=1000, beta=200.0)
        
    # Final SLSQP polish on P directly with equality constraint
    # SLSQP can be slower but more accurate for equality constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                    'jac': lambda x: np.ones_like(x)}]
    bounds = [(0.0, 1.0) for _ in range(n * n)]
    
    def try_optimize_slsqp(P_flat, maxiter=500, beta=100.0):
        nonlocal best_P, best_score
        try:
            result = opt.minimize(
                objective_and_grad,
                P_flat,
                args=(beta,),
                jac=True,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'disp': False}
            )
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

    if time.time() - start_time < timeout - safety_margin:
        try_optimize_slsqp(best_P.ravel(), maxiter=1000, beta=100.0)
        
    # One last check
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 0:
        best_P /= s
    else:
        best_P = np.ones((n, n)) / (n * n)
        
    return best_P

def arithmetic_kakeya_distribution() -> np.ndarray:
    return solve_kakeya()

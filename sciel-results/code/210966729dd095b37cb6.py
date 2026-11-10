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
    Uses log-sum-exp to create a smooth approximation of max(H0, H1, H2).
    As beta -> infinity, this approaches the hard max.
    """
    P = z
    D_val, D_grad = get_projection_entropy_grad(P, D_inv_flat, D_bins)
    H0_val, H0_grad = get_projection_entropy_grad(P, H0_inv_flat, H0_bins)
    H1_val, H1_grad = get_projection_entropy_grad(P, H1_inv_flat, H1_bins)
    H2_val, H2_grad = get_projection_entropy_grad(P, H2_inv_flat, H2_bins)
    
    H012 = np.array([H0_val, H1_val, H2_val])
    grads_012 = [H0_grad, H1_grad, H2_grad]
    
    # Soft-max: M_soft = (1/beta) * log(sum(exp(beta * Hi)))
    # Gradient: dM_soft/dP = sum_i (exp(beta*Hi)/sum(exp(beta*Hj))) * dHi/dP
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
    for _ in range(30):
        p = rng.random((n, n))
        p /= p.sum()
        inits.append(p)
    
    # Diagonal inits (x-y = c)
    for c in range(-9, 10):
        p = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if j - i == c:
                    p[i, j] = 1.0
        if p.sum() > 0:
            p /= p.sum()
            inits.append(p)
    
    # Anti-diagonal inits (x+y = c)
    for c in range(0, 19):
        p = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i + j == c:
                    p[i, j] = 1.0
        if p.sum() > 0:
            p /= p.sum()
            inits.append(p)
    
    # Concentrate x: put mass on columns
    for cx in range(n):
        p = np.zeros((n, n))
        p[:, cx] = 1.0
        p /= p.sum()
        inits.append(p)
    
    # Concentrate y: put mass on rows
    for cy in range(n):
        p = np.zeros((n, n))
        p[cy, :] = 1.0
        p /= p.sum()
        inits.append(p)
    
    # Corner concentrations
    for cx in [0, 1, 2, 7, 8, 9]:
        for cy in [0, 1, 2, 7, 8, 9]:
            p = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    p[i, j] = np.exp(-((i - cy)**2 + (j - cx)**2) / 1.5)
            p /= p.sum()
            inits.append(p)
    
    # Spread x-y: put mass where x-y is extreme
    p = np.zeros((n, n))
    p[0, 9] = 1.0
    p[9, 0] = 1.0
    p /= p.sum()
    inits.append(p)
    
    # More spread: put mass on corners
    p = np.zeros((n, n))
    p[0, 0] = 1.0
    p[0, 9] = 1.0
    p[9, 0] = 1.0
    p[9, 9] = 1.0
    p /= p.sum()
    inits.append(p)
    
    # Try to maximize x-y entropy by spreading across diagonals
    # Put equal mass on each diagonal
    p = np.zeros((n, n))
    for c in range(-9, 10):
        for i in range(n):
            for j in range(n):
                if j - i == c:
                    p[i, j] = 1.0
    p /= p.sum()
    inits.append(p)
    
    # Put mass on 4 corners with varying weights
    for w1, w2, w3, w4 in [(1,1,1,1), (1,2,2,1), (2,1,1,2), (1,3,3,1), (3,1,1,3)]:
        p = np.zeros((n, n))
        p[0, 0] = w1
        p[0, 9] = w2
        p[9, 0] = w3
        p[9, 9] = w4
        p /= p.sum()
        inits.append(p)
    
    # Try to minimize H0, H1, H2 while maximizing D
    # Concentrate on a few cells that spread out x-y
    for cells in [
        [(0,0), (9,9)],
        [(0,9), (9,0)],
        [(0,0), (9,0), (0,9)],
        [(0,0), (9,9), (0,9)],
        [(0,0), (9,9), (9,0)],
        [(2,2), (7,7), (2,7), (7,2)],
        [(1,1), (8,8), (1,8), (8,1)],
        [(0,0), (4,4), (8,8), (0,8), (4,0), (8,0)],
    ]:
        p = np.zeros((n, n))
        for (i, j) in cells:
            p[i, j] = 1.0
        p /= p.sum()
        inits.append(p)
    
    # Gradient-based local search from promising points
    # Points that maximize D/M: we want high D, low M
    # D is max when x-y is spread, M is min when x, x+y, x+2y are concentrated
    
    # Try points where x is concentrated but x-y is spread
    for cx in range(n):
        for spread in [1, 2, 3]:
            p = np.zeros((n, n))
            for dy in range(-spread, spread+1):
                y = 0
                x = cx
                if 0 <= y < n and 0 <= x < n:
                    p[y, x] = 1.0
                y2 = 9
                x2 = cx
                if 0 <= y2 < n and 0 <= x2 < n:
                    p[y2, x2] = 1.0
            if p.sum() > 0:
                p /= p.sum()
                inits.append(p)
    
    # Additional structured inits: mass on sparse sets that spread x-y
    # These target configurations where x-y has many distinct values
    # while x, x+y, x+2y are more concentrated
    for k in [2, 3, 4, 5, 6]:
        for step in range(1, n//k + 1):
            p = np.zeros((n, n))
            for i in range(0, n, step):
                for j in range(0, n, step):
                    if i // step < k and j // step < k:
                        p[i, j] = 1.0
            if p.sum() > 0:
                p /= p.sum()
                inits.append(p)
                
    # Hyperbolic-ish curves: x*y = c
    for c in range(1, 10):
        p = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i * j == c:
                    p[i, j] = 1.0
        if p.sum() > 0:
            p /= p.sum()
            inits.append(p)
            
    # Parabolic-ish: x^2 + y^2 = c
    for c in range(0, 20):
        p = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i**2 + j**2 == c:
                    p[i, j] = 1.0
        if p.sum() > 0:
            p /= p.sum()
            inits.append(p)
    
    # Grid inits
    for size in [2, 3, 4, 5]:
        for offset in range(n - size + 1):
            p = np.zeros((n, n))
            p[offset:offset+size, offset:offset+size] = 1.0
            p /= p.sum()
            inits.append(p)
    
    # Sparse grids
    for size in [3, 4, 5]:
        for offset in range(n - size + 1):
            p = np.zeros((n, n))
            for i in range(size):
                for j in range(size):
                    if (i + j) % 2 == 0:
                        p[offset+i, offset+j] = 1.0
            if p.sum() > 0:
                p /= p.sum()
                inits.append(p)
                
    # Diagonal bands
    for c in range(-9, 10):
        for width in [1, 2, 3]:
            p = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if abs((j - i) - c) < width:
                        p[i, j] = 1.0
            if p.sum() > 0:
                p /= p.sum()
                inits.append(p)
    
    # Anti-diagonal bands
    for c in range(0, 19):
        for width in [1, 2, 3]:
            p = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if abs((i + j) - c) < width:
                        p[i, j] = 1.0
            if p.sum() > 0:
                p /= p.sum()
                inits.append(p)

    import scipy.optimize as opt
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                    'jac': lambda x: np.ones_like(x)}]
    bounds = [(0.0, 1.0) for _ in range(n * n)]
    
    def try_optimize(P_flat, method, maxiter, ftol=None, gtol=None, beta=50.0):
        nonlocal best_P, best_score
        try:
            if method == 'SLSQP':
                # Define wrapper with fixed beta
                def obj_wrapper(x):
                    return objective_and_grad(x, beta=beta)
                result = opt.minimize(
                    obj_wrapper,
                    P_flat,
                    jac=True,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': maxiter, 'ftol': ftol or 1e-12, 'disp': False}
                )
            elif method == 'trust-constr':
                from scipy.optimize import LinearConstraint
                lc = LinearConstraint(np.ones((1, 100)), 1.0, 1.0)
                def obj_wrapper(x):
                    return objective_and_grad(x, beta=beta)
                result = opt.minimize(
                    obj_wrapper,
                    P_flat,
                    jac=True,
                    method='trust-constr',
                    bounds=bounds,
                    constraints=[lc],
                    options={'maxiter': maxiter, 'gtol': gtol or 1e-10, 'disp': False}
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
    
    # First pass: quick optimization from all inits
    for P_init in inits:
        if time.time() - start_time > timeout - safety_margin:
            break
        
        P_flat = P_init.ravel()
        # Use moderate beta for exploration
        try_optimize(P_flat, 'SLSQP', 200, beta=50.0)
        
    # Second pass: refine from best and nearby points with higher beta (sharper max)
    if time.time() - start_time < timeout - safety_margin:
        P_flat = best_P.ravel()
        # Increase beta to sharpen the max approximation
        try_optimize(P_flat, 'SLSQP', 500, ftol=1e-14, beta=100.0)
        try_optimize(P_flat, 'trust-constr', 300, beta=100.0)
        
        # Even sharper
        if time.time() - start_time < timeout - safety_margin:
            try_optimize(P_flat, 'SLSQP', 500, ftol=1e-15, beta=200.0)
    
    # Local refinement: perturb best_P and re-optimize
    if time.time() - start_time < timeout - safety_margin:
        for _ in range(30):
            if time.time() - start_time > timeout - safety_margin:
                break
            perturbation = rng.random((n, n)) * 0.1
            P_pert = best_P + perturbation
            P_pert = np.clip(P_pert, 0, None)
            s = P_pert.sum()
            if s > 0:
                P_pert /= s
                P_flat = P_pert.ravel()
                # Use high beta for refinement
                try_optimize(P_flat, 'SLSQP', 300, beta=100.0)
                
    # Final polish: very high beta
    if time.time() - start_time < timeout - safety_margin:
        P_flat = best_P.ravel()
        try_optimize(P_flat, 'SLSQP', 1000, ftol=1e-16, beta=500.0)
        try_optimize(P_flat, 'trust-constr', 500, beta=500.0)

    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 0:
        best_P /= s
    
    return best_P

def arithmetic_kakeya_distribution() -> np.ndarray:
    return solve_kakeya()

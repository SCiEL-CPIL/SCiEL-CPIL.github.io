import numpy as np
import time
import itertools
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

def smooth_max_entropy(P_flat, temperature=0.1):
    """Compute smooth max of H0, H1, H2 using log-sum-exp."""
    H0_val, H0_grad = get_projection_entropy_grad(P_flat, H0_inv_flat, H0_bins)
    H1_val, H1_grad = get_projection_entropy_grad(P_flat, H1_inv_flat, H1_bins)
    H2_val, H2_grad = get_projection_entropy_grad(P_flat, H2_inv_flat, H2_bins)
    
    Hs = np.array([H0_val, H1_val, H2_val])
    grads = [H0_grad, H1_grad, H2_grad]
    
    # Log-sum-exp for smooth max
    H_max = np.max(Hs)
    if H_max < 1e-9:
        return 1e-9, np.zeros_like(P_flat)
    
    exps = np.exp((Hs - H_max) / temperature)
    exps /= exps.sum()
    M_smooth = H_max + temperature * np.log(exps.sum())
    
    M_grad = sum(exps[i] * grads[i] for i in range(3))
    return M_smooth, M_grad

def objective_and_grad_smooth(z, temperature=0.1):
    P = z
    D_val, D_grad = get_projection_entropy_grad(P, D_inv_flat, D_bins)
    M_val, M_grad = smooth_max_entropy(P, temperature)
    
    if M_val < 1e-9:
        return -1e9, np.zeros_like(P)
    
    ratio = D_val / M_val
    grad_ratio = (M_val * D_grad - D_val * M_grad) / (M_val * M_val)
    
    obj = -ratio
    grad = -grad_ratio
    return obj, grad

def objective_and_grad(z):
    P = z
    D_val, D_grad = get_projection_entropy_grad(P, D_inv_flat, D_bins)
    H0_val, H0_grad = get_projection_entropy_grad(P, H0_inv_flat, H0_bins)
    H1_val, H1_grad = get_projection_entropy_grad(P, H1_inv_flat, H1_bins)
    H2_val, H2_grad = get_projection_entropy_grad(P, H2_inv_flat, H2_bins)
    
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

def solve_kakeya():
    n = 10
    start_time = time.time()
    timeout = 600
    safety_margin = 30
    
    best_P = np.ones((n, n)) / (n * n)
    best_score = score_P(best_P)
    
    rng = np.random.default_rng(42)
    
    # === Rank-1 (product) optimization: P[y,x] = R[y]*C[x] ===
    # 20D search: R (10D) and C (10D), each normalized to sum=1
    # P = outer(R, C), sum(P) = sum(R)*sum(C) = 1
    
    def rank1_score(R, C):
        P = np.outer(R, C)
        return score_P(P)
    
    def rank1_objective(z):
        R = np.exp(z[:10])
        R /= R.sum()
        C = np.exp(z[10:20])
        C /= C.sum()
        P = np.outer(R, C)
        return -rank1_score(R, C)
    
    def rank1_objective_grad(z):
        R = np.exp(z[:10])
        R /= R.sum()
        C = np.exp(z[10:20])
        C /= C.sum()
        P = np.outer(R, C).ravel()
        
        D_val, D_grad = get_projection_entropy_grad(P, D_inv_flat, D_bins)
        H0_val, H0_grad = get_projection_entropy_grad(P, H0_inv_flat, H0_bins)
        H1_val, H1_grad = get_projection_entropy_grad(P, H1_inv_flat, H1_bins)
        H2_val, H2_grad = get_projection_entropy_grad(P, H2_inv_flat, H2_bins)
        
        H012 = np.array([H0_val, H1_val, H2_val])
        M = np.max(H012)
        if M < 1e-9:
            return -1e9, np.zeros(20)
        
        ratio = D_val / M
        idx_max = np.argmax(H012)
        M_grad = [H0_grad, H1_grad, H2_grad][idx_max]
        grad_ratio = (M * D_grad - D_val * M_grad) / (M * M)
        
        # grad_ratio is w.r.t. P[y,x]
        # P[y,x] = R[y]*C[x]
        # dP/dR[y] = C[x] for all x, but we need dP_flat/dR[y]
        # P_flat index = y*10 + x
        # dP_flat[y*10+x]/dR[y] = C[x]
        # dP_flat[y*10+x]/dC[x] = R[y]
        
        grad_R = np.zeros(10)
        grad_C = np.zeros(10)
        for y in range(10):
            for x in range(10):
                idx = y * 10 + x
                grad_R[y] += grad_ratio[idx] * C[x]
                grad_C[x] += grad_ratio[idx] * R[y]
        
        # Chain rule through softmax: z -> exp(z) -> normalize -> R
        # dR/dz = R*(1 - R) (for softmax-like, but here it's exp/sum)
        # Actually: R_i = exp(z_i) / sum(exp(z_j))
        # dR_i/dz_i = R_i * (1 - R_i)
        # dR_i/dz_j = -R_i * R_j for i != j
        # This is the Jacobian of softmax
        
        # For simplicity, use finite-difference-like scaling
        # dR/dz has entries: R_i*(delta_ij - R_j)
        # grad_z_i = sum_j (dR_j/dz_i) * grad_R_j = R_i * grad_R_i - R_i * sum_j(R_j * grad_R_j)
        # = R_i * (grad_R_i - sum_j(R_j * grad_R_j))
        
        sum_RgR = np.dot(R, grad_R)
        grad_z_R = R * (grad_R - sum_RgR)
        
        sum_CgC = np.dot(C, grad_C)
        grad_z_C = C * (grad_C - sum_CgC)
        
        grad_z = np.concatenate([grad_z_R, grad_z_C])
        grad_z = -grad_z  # we're minimizing -score
        
        return -ratio, grad_z
    
    # Optimize rank-1 structure
    rank1_inits = []
    # Uniform
    rank1_inits.append(np.zeros(20))
    # Random
    for _ in range(30):
        rank1_inits.append(rng.random(20) - 1)
    # Biased toward edges
    for bias in [0.5, 1.0, 2.0, 3.0]:
        z = np.zeros(20)
        z[0] = bias
        z[9] = bias
        z[10] = bias
        z[19] = bias
        rank1_inits.append(z)
    
    for z_init in rank1_inits:
        if time.time() - start_time > timeout - safety_margin:
            break
        try:
            result = opt.minimize(
                rank1_objective,
                z_init,
                jac=rank1_objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False}
            )
            R = np.exp(result.x[:10])
            R /= R.sum()
            C = np.exp(result.x[10:20])
            C /= C.sum()
            P_opt = np.outer(R, C)
            sc = score_P(P_opt)
            if sc > best_score:
                best_score = sc
                best_P = P_opt.copy()
        except Exception:
            pass
    
    # === Full 100D optimization ===
    inits = []
    
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
    
    # Put mass on 4 corners with varying weights
    for w1, w2, w3, w4 in [(1,1,1,1), (1,2,2,1), (2,1,1,2), (1,3,3,1), (3,1,1,3)]:
        p = np.zeros((n, n))
        p[0, 0] = w1
        p[0, 9] = w2
        p[9, 0] = w3
        p[9, 9] = w4
        p /= p.sum()
        inits.append(p)
    
    # More structured inits
    for k in range(2, 6):
        for step in range(1, n//k + 1):
            p = np.zeros((n, n))
            for i in range(0, n, step):
                for j in range(0, n, step):
                    if i // step < k and j // step < k:
                        p[i, j] = 1.0
            if p.sum() > 0:
                p /= p.sum()
                inits.append(p)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                    'jac': lambda x: np.ones_like(x)}]
    bounds = [(0.0, 1.0) for _ in range(n * n)]
    
    def try_optimize_lbfgs(P_flat, maxiter=500):
        nonlocal best_P, best_score
        try:
            result = opt.minimize(
                objective_and_grad_smooth,
                P_flat,
                args=(0.1,),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-8, 'disp': False}
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
    
    def try_optimize_slsqp(P_flat, maxiter=300, ftol=1e-12):
        nonlocal best_P, best_score
        try:
            result = opt.minimize(
                objective_and_grad,
                P_flat,
                jac=True,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
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
    
    def try_optimize_trust(P_flat, maxiter=300):
        nonlocal best_P, best_score
        try:
            from scipy.optimize import LinearConstraint
            lc = LinearConstraint(np.ones((1, 100)), 1.0, 1.0)
            result = opt.minimize(
                objective_and_grad,
                P_flat,
                jac=True,
                method='trust-constr',
                bounds=bounds,
                constraints=[lc],
                options={'maxiter': maxiter, 'gtol': 1e-10, 'disp': False}
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
    
    # First pass: quick L-BFGS-B optimization from all inits (fast)
    for P_init in inits:
        if time.time() - start_time > timeout - safety_margin:
            break
        P_flat = P_init.ravel()
        try_optimize_lbfgs(P_flat, 300)
    
    # Second pass: SLSQP refinement from top candidates
    if time.time() - start_time < timeout - safety_margin:
        P_flat = best_P.ravel()
        try_optimize_slsqp(P_flat, 500, ftol=1e-14)
        try_optimize_trust(P_flat, 300)
    
    # Local refinement: perturb best_P and re-optimize with smooth objective
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
                try_optimize_lbfgs(P_flat, 300)
    
    # Final high-precision polish with SLSQP and tight tolerance
    if time.time() - start_time < timeout - safety_margin:
        P_flat = best_P.ravel()
        try_optimize_slsqp(P_flat, 1000, ftol=1e-16)
    
    # One more trust-constr polish
    if time.time() - start_time < timeout - safety_margin:
        P_flat = best_P.ravel()
        try_optimize_trust(P_flat, 500)
    
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 0:
        best_P /= s
    
    return best_P

def arithmetic_kakeya_distribution() -> np.ndarray:
    return solve_kakeya()

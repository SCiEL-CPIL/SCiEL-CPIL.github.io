# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    
    We seek a distribution P[y, x] (10x10) that maximizes D/M where:
    D = H(x - y)
    M = max(H(x), H(x+y), H(x+2y))
    
    Returns:
        P: 10x10 array of non-negative weights summing to 1.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin for 630s limit
    
    # Precompute indices
    x_idx = np.arange(n)
    y_idx = np.arange(n)
    X, Y = np.meshgrid(x_idx, y_idx)
    
    # Values for slopes
    val_m1 = X - Y      # x - y
    val_0  = X          # x
    val_1  = X + Y      # x + y
    val_2  = X + 2*Y    # x + 2y
    
    # Unique values and their ranges
    # val_m1: -9 to 9
    # val_0: 0 to 9
    # val_1: 0 to 18
    # val_2: 0 to 27
    
    # Helper to compute entropies from P
    def compute_entropies(P):
        # P is 10x10, P[y, x]
        # We need to pool probabilities by value.
        
        # For slope r, value v = x + r*y
        # We can use np.bincount or manual aggregation.
        # Since grids are small, we can create a mapping.
        
        # D: x - y
        # Values range from -9 to 9. Shift by 9 to get 0..18
        pm1 = P.copy()
        vals_m1 = val_m1 + 9  # 0..18
        # Sum P over cells with same vals_m1
        # Using bincount on flattened array
        flat_vals_m1 = vals_m1.flatten()
        flat_P = P.flatten()
        # Note: bincount requires non-negative integers
        p_m1 = np.bincount(flat_vals_m1, weights=flat_P, minlength=19)
        
        # M components:
        # H_0: x (0..9)
        vals_0 = val_0
        flat_vals_0 = vals_0.flatten()
        p_0 = np.bincount(flat_vals_0, weights=flat_P, minlength=10)
        
        # H_1: x + y (0..18)
        vals_1 = val_1
        flat_vals_1 = vals_1.flatten()
        p_1 = np.bincount(flat_vals_1, weights=flat_P, minlength=19)
        
        # H_2: x + 2y (0..27)
        vals_2 = val_2
        flat_vals_2 = vals_2.flatten()
        p_2 = np.bincount(flat_vals_2, weights=flat_P, minlength=28)
        
        # Compute entropies (in bits)
        # H = - sum p log2 p, ignoring p=0
        
        def shannon_entropy(p):
            p = p[p > 1e-15]  # Filter tiny probabilities
            if len(p) == 0:
                return 0.0
            return -np.sum(p * np.log2(p))
        
        H_m1 = shannon_entropy(p_m1)
        H_0 = shannon_entropy(p_0)
        H_1 = shannon_entropy(p_1)
        H_2 = shannon_entropy(p_2)
        
        D = H_m1
        M = max(H_0, H_1, H_2)
        
        return D, M, H_0, H_1, H_2, p_m1, p_0, p_1, p_2
    
    def objective_and_grad(theta):
        """
        theta: 100 parameters representing log-weights or similar.
        We use a softmax parameterization to ensure P >= 0 and sum P = 1.
        P = exp(theta) / sum(exp(theta))
        """
        P = theta.reshape(10, 10)
        # Softmax
        P_max = np.max(P)
        P_exp = np.exp(P - P_max)
        P_norm = P_exp / np.sum(P_exp)
        
        D, M, H_0, H_1, H_2, p_m1, p_0, p_1, p_2 = compute_entropies(P_norm)
        
        if M < 1e-9:
            return 0.0, np.zeros_like(theta)
        
        value = D / M
        
        # Gradient is complex due to max and entropy.
        # For now, we will use a numerical gradient or a simpler optimization.
        # Since we have a deadline, let's try to use scipy.optimize.minimize
        # with BFGS or L-BFGS-B, but we need gradients.
        # Alternatively, we can use a simpler approach:
        # Try to maximize D while keeping M small? Or minimize M/D?
        
        # Actually, let's just return the negative value for minimization.
        # We will use a finite difference gradient if analytic is too hard,
        # or we can try to derive it.
        
        # Let's stick to a robust numerical approach or use the fact that
        # the landscape might be smooth enough for L-BFGS-B with numerical grad.
        
        return -value, None

    # Strategy:
    # 1. Start with uniform distribution.
    # 2. Try to optimize using scipy.
    # 3. Also try some heuristic constructions.
    
    # Initial guess: Uniform
    P_uniform = np.ones((10, 10)) / 100.0
    D_u, M_u, _, _, _, _, _, _, _ = compute_entropies(P_uniform)
    best_value = D_u / M_u if M_u > 1e-9 else 0.0
    best_P = P_uniform.copy()
    
    # Heuristic 1: Diagonal concentration
    # Maximize H(x-y) means spreading x-y values.
    # Minimize H(x), H(x+y), H(x+2y) means concentrating those.
    
    # Let's try to concentrate mass on a set of points that have distinct x-y values
    # but similar x, x+y, x+2y values? That's hard because x determines x, x+y, x+2y.
    # If we fix x, H(x) is low. But x+y and x+2y vary with y.
    
    # Let's try a grid of points where x is constant?
    # If x = c, then H(x) = 0.
    # x-y = c-y. If y is uniform 0..9, x-y is uniform -9..c-1. H(x-y) = log2(10).
    # x+y = c+y. Uniform c..c+9. H(x+y) = log2(10).
    # x+2y = c+2y. Values c, c+2, ..., c+18. H(x+2y) = log2(10).
    # M = log2(10). D = log2(10). Ratio = 1.
    # Uniform distribution also gives ratio 1?
    # Uniform:
    # H(x) = log2(10).
    # H(x+y) is convolution of two uniforms. Entropy is higher than log2(10).
    # H(x-y) is also convolution-like.
    # Let's check uniform ratio.
    
    # Let's run a local search from uniform.
    
    # We will use scipy.optimize.minimize on the log-weights.
    # Parameters: 100 values.
    # Objective: minimize -D/M
    
    # To handle the non-differentiability of max, we can use a smooth approximation
    # or just rely on the fact that the max is usually achieved by one term,
    # and the gradient of the max is the gradient of the active term.
    
    # Let's implement a simple gradient ascent on the simplex.
    # Or use scipy with numerical gradients.
    
    # Given the time limit, let's try a few random restarts with scipy.
    
    def optimize_from_initial(P_init):
        theta_init = np.log(P_init + 1e-15).flatten()
        
        def obj(theta):
            P = theta.reshape(10, 10)
            P_max = np.max(P)
            P_exp = np.exp(P - P_max)
            P_norm = P_exp / np.sum(P_exp)
            D, M, _, _, _, _, _, _, _ = compute_entropies(P_norm)
            if M < 1e-9:
                return 0.0
            return -D / M
        
        # Use Nelder-Mead or Powell which don't need gradients?
        # Or L-BFGS-B with numerical grad.
        
        res = opt.minimize(obj, theta_init, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        
        theta_opt = res.x
        P_opt = theta_opt.reshape(10, 10)
        P_opt_max = np.max(P_opt)
        P_opt = np.exp(P_opt - P_opt_max)
        P_opt = P_opt / np.sum(P_opt)
        
        D_opt, M_opt, _, _, _, _, _, _, _ = compute_entropies(P_opt)
        val_opt = D_opt / M_opt if M_opt > 1e-9 else 0.0
        
        return P_opt, val_opt

    # Try optimization from uniform
    if time.time() < deadline - 60:
        P_opt, val_opt = optimize_from_initial(P_uniform)
        if val_opt > best_value:
            best_value = val_opt
            best_P = P_opt

    # Try some structured distributions
    # 1. Concentrate on x=0
    P_x0 = np.zeros((10, 10))
    P_x0[:, 0] = 1.0 / 10
    D, M, _, _, _, _, _, _, _ = compute_entropies(P_x0)
    if M > 1e-9:
        val = D / M
        if val > best_value:
            best_value = val
            best_P = P_x0

    # 2. Concentrate on y=0
    P_y0 = np.zeros((10, 10))
    P_y0[0, :] = 1.0 / 10
    D, M, _, _, _, _, _, _, _ = compute_entropies(P_y0)
    if M > 1e-9:
        val = D / M
        if val > best_value:
            best_value = val
            best_P = P_y0

    # 3. Concentrate on x=y (main diagonal)
    P_diag = np.zeros((10, 10))
    for i in range(10):
        P_diag[i, i] = 1.0 / 10
    D, M, _, _, _, _, _, _, _ = compute_entropies(P_diag)
    if M > 1e-9:
        val = D / M
        if val > best_value:
            best_value = val
            best_P = P_diag

    # 4. Random restarts
    np.random.seed(42)
    for i in range(10):
        if time.time() > deadline - 30:
            break
        
        # Random distribution
        P_rand = np.random.rand(10, 10)
        P_rand /= np.sum(P_rand)
        
        P_opt, val_opt = optimize_from_initial(P_rand)
        if val_opt > best_value:
            best_value = val_opt
            best_P = P_opt

    # Final check and normalization
    best_P = np.clip(best_P, 0, None)
    s = np.sum(best_P)
    if s < 1e-9:
        # Fallback to uniform
        best_P = np.ones((10, 10)) / 100.0
    else:
        best_P /= s
        
    return best_P

# EVOLVE-BLOCK-END

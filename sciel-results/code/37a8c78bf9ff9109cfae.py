# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    """
    start_time = time.time()
    DEADLINE = 630.0
    SAFETY_MARGIN = 30.0
    
    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((13, 13))  # A[k, j] = coeff of x^j in L_k^{(1)}(x)
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    # p is vector of coeffs in standard basis (ascending)
    # c is vector of coeffs in Laguerre basis
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)
        
    # Precompute M = A_T_inv
    M = A_T_inv
    # Weights for norm: w_k = k+1
    w = np.arange(1, 14)

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        # Build polynomial coefficients in standard basis (ascending)
        # P(x) = prod(x - r_i)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1] # ascending
        
        # c = M @ p
        c = M @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        # ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * w)
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_and_grad(log_roots):
        """
        Compute cost (1/norm_sq) and gradient w.r.t log_roots.
        """
        roots = np.exp(log_roots)
        
        # Check for numerical issues
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10, np.zeros(12)
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10, np.zeros(12)
            
        cost = 1.0 / norm_sq
        
        # Gradient calculation
        # d(cost)/d(r_i) = -1/norm_sq^2 * d(norm_sq)/d(r_i)
        # d(norm_sq)/d(r_i) = 2 * c^T W M (dp/dr_i)
        # dp/dr_i = -coeffs(Q_i) where Q_i(x) = P(x)/(x-r_i)
        
        # Compute P coefficients (ascending)
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]
        c = M @ p
        
        # Gradient w.r.t roots
        grad_r = np.zeros(12)
        for i in range(12):
            # Q_i(x) = prod_{j != i} (x - r_j)
            roots_minus_i = np.delete(roots, i)
            q_coeffs_desc = np.poly(roots_minus_i)
            q = q_coeffs_desc[::-1] # ascending
            dp_dr_i = -q
            
            # d(norm_sq)/d(r_i)
            # term = 2 * (c * w) @ (M @ dp_dr_i)
            # Let v = M @ dp_dr_i
            v = M @ dp_dr_i
            # d_norm_sq = 2 * sum(c * w * v)
            d_norm_sq = 2.0 * np.sum(c * w * v)
            
            grad_r[i] = -1.0 / (norm_sq**2) * d_norm_sq
            
        # Chain rule: d(cost)/d(y_i) = d(cost)/d(r_i) * r_i
        grad_y = grad_r * roots
        
        return cost, grad_y

    # Optimization Strategy:
    # 1. Generate multiple initial guesses.
    # 2. Run L-BFGS-B with analytical gradient from each.
    # 3. Keep the best result.
    
    best_val = 1e10
    best_log_roots = None
    best_roots = None
    
    # Initial guesses
    guesses = []
    
    # 1. Geometric sequence a=4, q=2
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    guesses.append(seed_log_roots)
    
    # 2. Random perturbations around the seed
    rng = np.random.default_rng(42)
    for _ in range(20):
        perturbation = rng.normal(0, 0.5, 12)
        guesses.append(seed_log_roots + perturbation)
        
    # 3. Different geometric bases
    # Try a=2, q=1.5
    guesses.append(np.log(2.0) + np.arange(12) * np.log(1.5))
    # Try a=10, q=1.2
    guesses.append(np.log(10.0) + np.arange(12) * np.log(1.2))
    
    # 4. Random uniform in log space
    for _ in range(10):
        guesses.append(rng.uniform(-2, 8, 12))

    bounds_log = [(-4.0, 10.0)] * 12
    
    for i, guess in enumerate(guesses):
        if not time_ok():
            break
            
        # Clip initial guess to bounds
        guess_clipped = np.clip(guess, -4.0, 10.0)
        
        try:
            res = minimize(
                cost_and_grad,
                guess_clipped,
                method='L-BFGS-B',
                bounds=bounds_log,
                jac=True, # Use analytical gradient
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 1000}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # Final high-precision polish
    if best_log_roots is not None and time_ok():
        try:
            res = minimize(
                cost_and_grad,
                best_log_roots,
                method='L-BFGS-B',
                bounds=bounds_log,
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 2000}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_log_roots = res.x
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
# EVOLVE-BLOCK-END

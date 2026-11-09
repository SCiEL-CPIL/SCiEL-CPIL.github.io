import numpy as np
from scipy.optimize import minimize
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590

    m = 12
    n = 1  # alpha = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((13, 13))
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

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
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        # ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(roots):
        """
        Minimize the upper bound on C4.
        Bound ~ 1 / ||P||^2.
        So we minimize 1/norm_sq, which is equivalent to maximizing norm_sq.
        We return 1/norm_sq.
        """
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        norm_sq = compute_norm_sq(roots)
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq

    best_val = 1e10
    best_roots = None

    # Strategy: Parametric Optimization
    # Hypothesis: The optimal roots are scaled/shifted versions of roots of known orthogonal polynomials.
    
    # 1. Get candidate root structures
    candidate_sets = []
    
    # Roots of Laguerre L_12^(alpha) for various alpha
    for alpha in [0, 1, 2, 3, 0.5, -0.5]:
        try:
            # roots_laguerre(n, alpha) returns roots of L_n^(alpha)
            roots_base = roots_laguerre(12, alpha)[0]
            # Ensure they are positive (they should be for alpha > -1)
            if np.all(roots_base > 0):
                candidate_sets.append(roots_base)
        except:
            pass
            
    # Roots of Hermite H_12 shifted to be positive
    # H_12 roots are symmetric around 0. Shift by max(root) + margin
    try:
        # Using scipy.special.roots_hermite
        from scipy.special import roots_hermite
        h_roots = roots_hermite(12)[0]
        # Shift to make all positive
        shift = -np.min(h_roots) + 1.0
        candidate_sets.append(h_roots + shift)
    except:
        pass

    # 2. For each candidate set, optimize Scale (s) and Shift (t)
    # r_i = s * base_i + t
    
    for base in candidate_sets:
        # Optimize s and t
        # Use log-space for s to ensure s > 0
        # t can be any real, but roots must be positive.
        
        def cost_st(log_s, t):
            s = np.exp(log_s)
            roots = s * base + t
            if np.any(roots <= 0):
                return 1e10
            return cost_func(roots)
        
        # Initial guesses for s and t
        # s=1, t=0
        init_log_s = 0.0
        init_t = 0.0
        
        # Try a few initial guesses for t to cover different shifts
        for init_t_guess in [0.0, 1.0, 5.0, -1.0]:
             try:
                 res = minimize(
                     cost_st,
                     [init_log_s, init_t_guess],
                     method='Nelder-Mead',
                     options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                 )
                 if res.fun < best_val:
                     best_val = res.fun
                     s_opt = np.exp(res.x[0])
                     t_opt = res.x[1]
                     best_roots = s_opt * base + t_opt
             except Exception:
                 pass

    # 3. Also try pure scaling (t=0) with minimize_scalar for robustness
    for base in candidate_sets:
        def cost_s(log_s):
            s = np.exp(log_s)
            roots = s * base
            if np.any(roots <= 0):
                return 1e10
            return cost_func(roots)
            
        # Grid search for initial log_s
        # Roots range from ~1 to ~100 usually.
        # Let's scan log_s from -2 to 5
        min_s_val = 1e10
        best_log_s = 0
        for ls in np.linspace(-2, 5, 100):
            val = cost_s(ls)
            if val < min_s_val:
                min_s_val = val
                best_log_s = ls
                
        # Refine
        try:
            res = minimize_scalar(cost_s, bounds=(best_log_s - 1, best_log_s + 1), method='bounded')
            if res.fun < best_val:
                best_val = res.fun
                s_opt = np.exp(res.x)
                best_roots = s_opt * base
        except:
            pass

    # 4. If we found a good parametric solution, do a final local polish on the 12 roots
    # to allow deviations from the strict parametric form.
    if best_roots is not None and time.time() < deadline - 10:
        # Convert to log space for positivity constraint in local opt
        log_roots = np.log(best_roots)
        
        def cost_log_roots(lr):
            r = np.exp(lr)
            return cost_func(r)
            
        try:
            res = minimize(
                cost_log_roots,
                log_roots,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_roots = np.exp(res.x)
        except:
            pass

    # Fallback if parametric search failed or found nothing better than seed
    if best_roots is None or best_val > 1.0:
        # Use the seed from the original program as fallback
        best_roots = np.array([4.0 * (2.0 ** i) for i in range(12)])
        best_val = cost_func(best_roots)

    return list(best_roots)

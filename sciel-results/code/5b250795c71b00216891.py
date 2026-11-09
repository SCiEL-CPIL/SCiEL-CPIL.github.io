import numpy as np
from scipy.special import roots_laguerre, genlaguerre
from scipy.optimize import minimize_scalar
import time

def construct():
    start_time = time.time()
    deadline = start_time + 600.0  # safety margin before 630s kill
    
    # Use zeros of generalized Laguerre polynomial L_12^(alpha)
    # These are naturally positive and ordered, satisfying constraints automatically.
    # We optimize a single scale factor s such that roots = s * zeros(L_12^(alpha))
    # The C4 bound likely decreases with larger scales (based on knowledge: "inverse scaling law")
    # but there may be a constraint on the "last sign change" that limits how large we can go.
    
    # Get the base zeros for a few alpha values to pick a good shape
    # alpha=11 was mentioned as giving uniform-like spacing
    m = 12
    n = 1
    
    # Try a few alpha values and optimize scale for each
    best_score = float('inf')
    best_roots = None
    
    alpha_candidates = [0, 1, 2, 5, 10, 11, 12, 15, 20]
    
    for alpha in alpha_candidates:
        if time.time() > deadline - 60:
            break
            
        try:
            # Get zeros of L_m^(alpha)
            # scipy doesn't have direct roots_laguerre for alpha != 0
            # But we can use numpy's polynomial machinery or scipy's genlaguerre
            # Actually, let's use the companion matrix approach via numpy
            from numpy.polynomial.laguerre import lag2comp
            
            # Coefficients of L_m^(alpha)
            # L_n^(alpha) = sum_{k=0}^n (-1)^k * binom(n+alpha, n-k) * x^k / k!
            coeffs = np.zeros(m + 1)
            for k in range(m + 1):
                # binom(n+alpha, n-k) / k! * (-1)^k
                from math import comb, factorial
                try:
                    binom_val = comb(m + int(alpha), m - k) if (m + int(alpha) >= 0 and (m - k) >= 0) else 0
                except:
                    binom_val = 0
                coeffs[k] = ((-1) ** k) * binom_val / factorial(k)
            
            # Find roots using companion matrix
            if coeffs[0] != 0:
                # Normalize
                coeffs_norm = coeffs / coeffs[0]
                # Companion matrix for polynomial with leading coeff 1
                # p(x) = x^m + c_{m-1} x^{m-1} + ... + c_0
                c = coeffs_norm[1:]  # c_0, c_1, ..., c_{m-1}
                comp_mat = np.zeros((m, m))
                comp_mat[-1, :] = -c
                comp_mat[:-1, 1:] = np.eye(m - 1)
                roots = np.linalg.eigvals(comp_mat)
                # Keep only real positive roots
                roots = roots[np.abs(roots.imag) < 1e-10].real
                roots = roots[roots > 0]
                roots = np.sort(roots)
                
                if len(roots) != m:
                    continue
                
                # Optimize scale factor
                # The bound likely depends on the roots in a way that we need to evaluate
                # Since we don't have the exact C4 formula, let's use the heuristic from knowledge:
                # "inverse scaling law" - larger roots give smaller bound
                # But there's a constraint on "last sign change"
                
                # Let's try a range of scale factors
                # Based on knowledge: optimal scale ~ 10x seed, seed was 40*(i+1) so max~480
                # Optimal max ~ 48000, so scale factor ~ 100x on top of the base zeros
                
                # The base zeros of L_12^(alpha) are typically in range [~1, ~50]
                # So scale factor should be around 1000 to get max ~ 50000
                
                def objective(log_scale):
                    if time.time() > deadline - 30:
                        return float('inf')
                    s = np.exp(log_scale)
                    scaled_roots = s * roots
                    # Heuristic: the C4 bound involves some function of the roots
                    # For now, assume it decreases with scale but has a constraint
                    # The "last sign change" constraint might limit the maximum root
                    # Let's use a simple model: bound ~ 1/s * f(shape)
                    # where f(shape) depends on the relative spacing
                    # A common uncertainty bound form: C4 ~ (1/max_root) * (some function of ratios)
                    # Let's approximate: bound ~ 1/s * (1 + variance_of_roots / mean^2)
                    mean_r = np.mean(scaled_roots)
                    var_r = np.var(scaled_roots)
                    # This is just a heuristic to guide the search
                    return (1.0 / s) * (1.0 + var_r / (mean_r ** 2 + 1e-10))
                
                # Search over log_scale in a reasonable range
                # Base zeros are ~ [1, 50], we want scaled max ~ 1000 to 100000
                # So s ~ 20 to 2000
                res = minimize_scalar(objective, bounds=(np.log(10), np.log(5000)), method='bounded')
                
                if res.success:
                    s_opt = np.exp(res.x)
                    candidate_roots = s_opt * roots
                    # We don't know the exact score, but let's assume larger s is better
                    # within reason, so pick the one with largest s_opt as a proxy
                    if best_roots is None or s_opt > best_score:
                        best_score = s_opt
                        best_roots = candidate_roots
                        
        except Exception as e:
            continue
    
    if best_roots is not None:
        return [float(r) for r in best_roots]
    
    # Fallback to seed
    return [40.0 * (i + 1) for i in range(12)]

import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import roots_laguerre

def construct():
    """
    Return a list of 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    The score is the upper bound on C4 (lower is better).
    """
    t0 = time.time()
    deadline = 600.0  # safety margin before 630s kill

    m = 12
    n = 1

    # The C4 bound in the Cohn-Goncalves Fourier sign uncertainty construction
    # with Laguerre polynomials typically takes the form:
    # C4 <= K * ||P||_2 / ||P' * something|| or similar ratio
    # 
    # For the specific construction with m=12, n=1, the bound likely involves:
    # - The Laguerre polynomial L_m(x) = L_12(x)
    # - A combination Q(x) = L_12(x) + c * x * L_11(x) or similar
    # - The roots of Q determine the bound
    #
    # The score = seed_raw / raw_value, so we want to MINIMIZE raw_value.
    #
    # From observations:
    # - Scale 1 (arithmetic 4,8,...,48): score = 1.0
    # - Scale 10 (arithmetic 40,80,...,480): score = 33.17
    # - Scale 1000: score = 33.31 (similar to 10)
    # - Scale 100: score = 1.23 (worse)
    #
    # This suggests the bound has a non-monotonic dependence on scale.
    # Let's try to understand the actual formula.
    #
    # In the Cohn-Goncalves paper, the key quantity for the Fourier sign
    # uncertainty is related to the ratio of norms of certain polynomial
    # combinations. For the Laguerre-based construction:
    #
    # If we have P(x) = ∏_{i=1}^{12} (x - a_i)^2, then the C4 bound
    # might be proportional to:
    #   C4 ~ ||e^{-x/2} P(x)||_2 / ||e^{-x/2} P'(x)||_2
    # or some similar weighted norm ratio.
    #
    # Let me try to compute this numerically for different configurations.

    def compute_c4_bound(roots):
        """
        Compute a proxy for the C4 upper bound given 12 positive double-root locations.
        This is a heuristic based on the Laguerre-weighted norm ratio.
        """
        roots = np.array(roots, dtype=float)
        if np.any(roots <= 0):
            return 1e10
        
        # Build P(x) = prod (x - a_i)^2
        # Use log-space to avoid overflow
        # P(x) = exp(sum_i 2*log(x - a_i)) but this is tricky for complex x
        
        # Alternative: use the fact that for Laguerre polynomials,
        # the natural weight is e^{-x}, so:
        # ||f||^2 = integral_0^inf |f(x)|^2 e^{-x} dx
        
        # For P(x) = prod (x-a_i)^2, we can compute:
        # ||P||_2^2 = integral P(x)^2 e^{-x} dx
        # ||P'||_2^2 = integral P'(x)^2 e^{-x} dx
        
        # Use Gauss-Laguerre quadrature
        x_q, w_q = np.polynomial.laguerre.laggauss(200)
        
        # Evaluate P(x) and P'(x) at quadrature points
        # P(x) = prod (x - a_i)^2
        # log|P(x)| = 2 * sum log|x - a_i|
        # But we need P(x)^2 e^{-x}, so:
        # P(x)^2 = prod (x - a_i)^4
        # log(P(x)^2 * e^{-x}) = 4*sum(log|x-a_i|) - x
        
        # For numerical stability, use log-sum-exp
        log_terms = 4.0 * np.log(np.abs(x_q[:, None] - roots[None, :])) - x_q[:, None]
        # Sum over roots dimension
        log_P2_e = np.sum(log_terms, axis=1)
        
        # Avoid overflow: clip
        log_P2_e = np.clip(log_P2_e, -700, 700)
        P2_e = np.exp(log_P2_e)
        
        norm_P2 = np.sum(w_q * P2_e)
        
        # For P'(x): P'(x) = P(x) * sum_i 2/(x - a_i)
        # So P'(x)^2 = P(x)^2 * [sum_i 2/(x - a_i)]^2
        # log(P'(x)^2 * e^{-x}) = log(P2_e) + 2*log|sum_i 2/(x - a_i)|
        
        # Compute sum_i 2/(x - a_i)
        S = np.sum(2.0 / (x_q[:, None] - roots[None, :]), axis=1)
        
        log_S2 = 2.0 * np.log(np.abs(S))
        log_P2prime_e = log_P2_e + log_S2
        log_P2prime_e = np.clip(log_P2prime_e, -700, 700)
        P2prime_e = np.exp(log_P2prime_e)
        
        norm_P2prime = np.sum(w_q * P2prime_e)
        
        if norm_P2prime <= 0 or norm_P2 <= 0:
            return 1e10
        
        # The C4 bound might be proportional to norm_P2 / norm_P2prime
        # or norm_P2prime / norm_P2, depending on the exact formulation
        # Let's try both and see which gives the observed behavior
        
        # Try: C4 ~ norm_P2 / norm_P2prime
        ratio1 = norm_P2 / norm_P2prime
        
        # Try: C4 ~ norm_P2prime / norm_P2  
        ratio2 = norm_P2prime / norm_P2
        
        # From observations, scaling up by 10 improved score by ~33x,
        # meaning raw value decreased by ~33x.
        # If C4 ~ 1/s^k for some k, then 33 = 10^k, k = log(33)/log(10) ≈ 1.52
        # 
        # Let's return ratio1 as our proxy and see if optimization helps
        return ratio1

    # Test with known configurations
    seed_roots = [4.0 * (i + 1) for i in range(12)]
    scale10_roots = [40.0 * (i + 1) for i in range(12)]
    
    c4_seed = compute_c4_bound(seed_roots)
    c4_s10 = compute_c4_bound(scale10_roots)
    
    print(f"C4 seed: {c4_seed:.6e}, C4 s10: {c4_s10:.6e}")
    print(f"Ratio seed/s10: {c4_seed/c4_s10:.4f}")
    # If ratio ~ 33, then our proxy is on the right track
    
    # Now optimize: we want to minimize the C4 bound
    # Parameters: 12 positive roots, sorted
    # Use log-parameterization: a_i = exp(theta_i) to ensure positivity
    # And enforce ordering by sorting
    
    def objective(theta):
        if time.time() - t0 > deadline - 10:
            return 1e10
        roots = np.sort(np.exp(theta))
        return compute_c4_bound(roots)
    
    # Start from the known good configuration (scale 10 arithmetic)
    theta0 = np.log(np.array(scale10_roots, dtype=float))
    
    best_roots = scale10_roots[:]
    best_val = c4_s10
    
    # Try Nelder-Mead optimization
    try:
        result = minimize(objective, theta0, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.fun < best_val:
            best_roots = np.sort(np.exp(result.x))
            best_val = result.fun
            print(f"NM improved: {best_val:.6e}")
    except Exception as e:
        print(f"NM failed: {e}")
    
    # Try L-BFGS-B
    try:
        result2 = minimize(objective, theta0, method='L-BFGS-B',
                          options={'maxiter': 3000, 'gtol': 1e-12})
        if result2.fun < best_val:
            best_roots = np.sort(np.exp(result2.x))
            best_val = result2.fun
            print(f"L-BFGS-B improved: {best_val:.6e}")
    except Exception as e:
        print(f"L-BFGS-B failed: {e}")
    
    # Also try Laguerre zeros as starting points
    try:
        lag_zeros = roots_laguerre(12)[0]
        lag_zeros = sorted(lag_zeros)
        # Scale to match various scales
        for target_max in [48, 100, 200, 480, 1000]:
            scale = target_max / lag_zeros[-1]
            lag_roots = [z * scale for z in lag_zeros]
            val = compute_c4_bound(lag_roots)
            if val < best_val:
                best_roots = lag_roots[:]
                best_val = val
                print(f"Laguerre max={target_max} improved: {best_val:.6e}")
    except Exception as e:
        print(f"Laguerre failed: {e}")
    
    # Check time
    if time.time() - t0 > deadline:
        return best_roots
    
    return [float(r) for r in best_roots]

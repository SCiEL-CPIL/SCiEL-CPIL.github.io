"""Refined uncertainty inequality - Cohn-Goncalves Laguerre construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_laguerre

# Precompute Gauss-Laguerre quadrature nodes and weights (fixed for performance)
_N_QUAD = 400
_X_NODES, _X_WEIGHTS = np.polynomial.laguerre.laggauss(_N_QUAD)
_LOG_WEIGHTS = np.log(_X_WEIGHTS + 1e-300)

def _compute_bound(roots):
    """
    Compute an upper bound related to the Cohn-Goncalves Laguerre construction.
    roots: array of 12 positive double-root locations.
    Returns a scalar bound (lower is better).
    """
    roots = np.asarray(roots, dtype=np.float64)
    if len(roots) != 12:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    
    # Sort roots
    roots = np.sort(roots)
    
    # Check for reasonable spacing (no overlaps)
    if np.any(np.diff(roots) < 1e-8):
        return 1e10
    
    # Compute the Laguerre-weighted L2 norm of P(x) = prod (x - r_i)^2
    # Integral I = ∫_0^∞ |P(x)|^2 e^{-x} dx = ∫_0^∞ [prod (x-r_i)^4] e^{-x} dx
    # Using Gauss-Laguerre quadrature: I ≈ sum w_i * [prod (x_i - r_j)^4]
    
    # log|P(x)|^2 = 4 * sum_j log|x - r_j|
    # We compute log_terms = log(w_i) + 4 * sum_j log|x_i - r_j|
    
    log_abs_P_sq = np.zeros_like(_X_NODES)
    for r in roots:
        # Avoid log(0)
        log_abs_P_sq += 4.0 * np.log(np.abs(_X_NODES - r) + 1e-300)
    
    log_terms = log_abs_P_sq + _LOG_WEIGHTS
    max_log = np.max(log_terms)
    
    # If the value is astronomically large, return a large penalty
    if max_log > 700:
        return np.exp(700)
    
    # Compute sum of exp(log_terms - max_log) * exp(max_log)
    integral = np.exp(max_log) * np.sum(np.exp(log_terms - max_log))
    
    return integral

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin from 630s limit
    
    # The seed returns [4, 8, 12, ..., 48] and scores 1.0
    seed_roots = np.array([4.0 * (i + 1) for i in range(12)])
    seed_bound = _compute_bound(seed_roots)
    
    best_roots = seed_roots.copy()
    best_bound = 1.0  # Normalized score (raw / seed_bound)

    def objective(x):
        if time.time() > deadline - 5:
            return best_bound
        if np.any(x <= 0):
            return 1e10
        raw = _compute_bound(x)
        if raw >= 1e9:
            return 1e10
        return raw / seed_bound

    # Strategy: Physics-informed initialization based on Laguerre polynomial zeros.
    # The polynomial P(x) = prod (x-r_i)^2 has degree 24.
    # To minimize the L2 norm with a fixed leading coefficient (or similar constraints),
    # the roots should be distributed similarly to the zeros of L_24(x) or a scaled version.
    # However, the roots are double, so the structure is more rigid.
    
    # Let's try roots near the zeros of L_12(x) and L_24(x) with various scalings.
    # Zeros of L_n(x) are roughly in [0, 4n].
    
    candidates = []
    
    # 1. Zeros of L_12(x) scaled by various factors
    # Since we have 12 roots, and they are double, maybe they map to the 12 zeros of L_12?
    # Or maybe the 12 roots correspond to a subset or a scaled set of L_24 zeros?
    
    # Get zeros of L_12 and L_24
    # np.polynomial.laguerre.laggauss gives quadrature nodes, not polynomial zeros directly.
    # We need the roots of the Laguerre polynomial L_n(x).
    # We can use np.polynomial.laguerre.lagroots or find roots via companion matrix.
    
    try:
        # Roots of L_12
        c12 = np.polynomial.laguerre.laggauss(12) # This is quadrature, not roots.
        # Correct way to get roots of L_n:
        # The roots of L_n(x) are the same as the nodes of Gauss-Laguerre quadrature? No.
        # Gauss-Laguerre nodes are roots of L_n(x). Yes, they are.
        # Wait, Gauss-Laguerre quadrature nodes x_i are the roots of L_n(x).
        # So _X_NODES for n=12 are the roots of L_12(x).
        
        roots_L12 = np.polynomial.laguerre.laggauss(12)[0]
        roots_L24 = np.polynomial.laguerre.laggauss(24)[0]
        
        # Hypothesis 1: Roots are scaled versions of roots_L12
        # P(x) = prod (x - r_i)^2.
        # If r_i = s * roots_L12[i], what s minimizes the bound?
        # Let's try a range of scales.
        for s in np.linspace(0.5, 5.0, 20):
            candidates.append(np.sort(s * roots_L12))
            
        # Hypothesis 2: Roots are the first 12 roots of L_24, or some subset?
        # Or maybe the roots are clustered around the mean?
        
        # Hypothesis 3: The seed [4, 8, ..., 48] is an arithmetic progression.
        # Maybe an optimized arithmetic progression is better?
        # Let's search over start 'a' and step 'd'.
        # Roots: a, a+d, ..., a+11d
        # This is a 2D search.
        
    except Exception:
        pass

    # Add the seed as a candidate
    candidates.append(seed_roots.copy())
    
    # Generate candidates for arithmetic progression search
    # a in [1, 10], d in [1, 5]
    a_vals = np.linspace(1.0, 10.0, 15)
    d_vals = np.linspace(1.0, 5.0, 15)
    for a in a_vals:
        for d in d_vals:
            roots = a + d * np.arange(12)
            if np.any(roots <= 0):
                continue
            candidates.append(roots)

    # Evaluate all candidates and pick the best few for local refinement
    candidate_scores = []
    for c in candidates:
        if time.time() > deadline - 100:
            break
        score = objective(c)
        candidate_scores.append((score, c))
        
    # Sort by score
    candidate_scores.sort(key=lambda x: x[0])
    
    # Refine the top 10 candidates with L-BFGS-B
    refined_count = 0
    for score, c in candidate_scores[:10]:
        if time.time() > deadline - 20:
            break
        try:
            result = minimize(
                objective,
                c,
                method='L-BFGS-B',
                bounds=[(0.01, 100)] * 12,
                options={'maxiter': 200, 'maxfun': 5000}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_roots = result.x.copy()
                refined_count += 1
        except Exception:
            pass

    # If we still have time, do a random restart local search
    if time.time() < deadline - 50:
        for _ in range(20):
            if time.time() > deadline - 20:
                break
            # Random perturbation of the best found so far
            noise = np.random.normal(0, 1.0, 12)
            x0 = best_roots + noise
            if np.any(x0 <= 0):
                continue
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(0.01, 100)] * 12,
                    options={'maxiter': 100, 'maxfun': 2000}
                )
                if result.fun < best_bound:
                    best_bound = result.fun
                    best_roots = result.x.copy()
            except Exception:
                pass

    # Ensure roots are sorted and positive
    best_roots = np.sort(best_roots)
    if np.any(best_roots <= 0):
        best_roots = seed_roots.copy()
    
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END

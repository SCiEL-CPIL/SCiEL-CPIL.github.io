"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre, eval_genlaguerre
import time

def _compute_c4(roots):
    """
    Compute the C4 upper bound for the Cohn-Goncalves Laguerre construction.
    P(x) = L_1^{(24)}(x) * prod_{i=1}^{12} (x - r_i)^2
    L_1^{(24)}(x) = -x + 25
    C4 = 1 / sqrt( ||P||^2_{L2(e^{-x})} )
    """
    roots = np.asarray(roots, dtype=float)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return 1e10
    
    # Sort roots to ensure consistent ordering for potential numerical stability
    roots = np.sort(roots)
    
    # Check for distinct roots (strictly increasing)
    if np.any(np.diff(roots) <= 1e-12):
        return 1e10

    # Build Q(x) = prod_i (x - r_i)^2 as polynomial coefficients
    # Using numpy polynomial tools for robustness, though manual is fine for n=12
    # Let's stick to the manual method from the seed for speed, but verify correctness
    q_coeffs = np.array([1.0])
    for r in roots:
        # Multiply by (x-r)^2 = x^2 - 2rx + r^2
        # q_coeffs is [c0, c1, ..., ck] representing c0*x^k + ... + ck
        # Wait, the seed code uses:
        # new_coeffs[i] += q_coeffs[i]       -> x * q(x)
        # new_coeffs[i+1] += -2.0 * r * q_coeffs[i]
        # new_coeffs[i+2] += r*r * q_coeffs[i]
        # This implies q_coeffs[i] is coefficient of x^i?
        # Let's trace: q = [1] (x^0)
        # r=4:
        # new[0] += 1 (x^0 * x^2? No. (x-r)^2 * 1 = x^2 - 8x + 16)
        # The seed code:
        # new_coeffs[i] += q_coeffs[i]       # coeff of x^(i+2)? No.
        # Let's re-read seed code carefully.
        # q_coeffs starts [1.0]. This is 1.
        # Loop r:
        # new_coeffs has len+2.
        # i=0: q[0]=1.
        # new[0] += 1
        # new[1] += -2r
        # new[2] += r^2
        # So new is [1, -2r, r^2]. This corresponds to 1*x^2 - 2r*x + r^2?
        # No, standard poly coeffs in numpy are highest power first.
        # But here index 0 is added to index 0.
        # If q_coeffs[i] is coeff of x^i, then:
        # (x^2 - 2rx + r^2) * sum q_i x^i
        # Term x^2 * q_i x^i -> q_i x^(i+2). So new[i+2] += q_i.
        # Term -2rx * q_i x^i -> -2r q_i x^(i+1). So new[i+1] += -2r q_i.
        # Term r^2 * q_i x^i -> r^2 q_i x^i. So new[i] += r^2 q_i.
        
        # The seed code does:
        # new_coeffs[i] += q_coeffs[i]       # This adds to x^i? No, it adds to index i.
        # new_coeffs[i+1] += -2.0 * r * q_coeffs[i]
        # new_coeffs[i+2] += r*r * q_coeffs[i]
        
        # This matches:
        # new[i] gets contribution from r^2 * q_i ? No, the seed code adds q_i to new[i].
        # Let's check the seed code again.
        # new_coeffs[i] += q_coeffs[i]
        # new_coeffs[i+1] += -2.0 * r * q_coeffs[i]
        # new_coeffs[i+2] += r*r * q_coeffs[i]
        
        # This implies:
        # new[i] receives q_i.
        # new[i+1] receives -2r q_i.
        # new[i+2] receives r^2 q_i.
        
        # This corresponds to multiplying by (1 - 2r/x + r^2/x^2)? No.
        # It corresponds to multiplying by (x^2 ...)?
        # If we assume q_coeffs[i] is coeff of x^i.
        # Then r^2 q_i x^i goes to new[i].
        # -2r q_i x^i * x = -2r q_i x^(i+1) goes to new[i+1].
        # q_i x^i * x^2 = q_i x^(i+2) goes to new[i+2].
        # So the multiplier is (r^2 - 2rx + x^2) = (x-r)^2.
        # So q_coeffs[i] IS the coefficient of x^i.
        # And new_coeffs[i] is the coefficient of x^i in the new polynomial.
        
        # So q_coeffs is in ascending order of powers: [c0, c1, ..., ck] for c0 + c1 x + ...
        
        new_coeffs = np.zeros(len(q_coeffs) + 2)
        for i in range(len(q_coeffs)):
            new_coeffs[i] += r*r * q_coeffs[i]
            new_coeffs[i+1] += -2.0 * r * q_coeffs[i]
            new_coeffs[i+2] += q_coeffs[i]
        q_coeffs = new_coeffs
    
    # P(x) = (25 - x) * Q(x)
    # Q(x) = sum q_i x^i
    # P(x) = 25 Q(x) - x Q(x)
    # P(x) = sum 25 q_i x^i - sum q_i x^(i+1)
    # p_j = 25 q_j - q_{j-1} (with q_{-1}=0)
    
    n_q = len(q_coeffs)
    p_coeffs = np.zeros(n_q + 1)
    for i in range(n_q):
        p_coeffs[i] += 25.0 * q_coeffs[i]
        if i + 1 < n_q + 1:
            p_coeffs[i+1] += -q_coeffs[i]
            
    # p_coeffs[i] is coefficient of x^i in P(x)
    # Note: degree of Q is 24. Degree of P is 25.
    # p_coeffs length is 26.
    
    # c[k] = p_coeffs[k]
    c = p_coeffs
    
    # ||P||^2 = sum_{k,j} c_k c_j Gamma(k+j+1)
    max_deg = len(c) - 1
    norm_sq = 0.0
    
    # Precompute factorials/gamma values?
    # Gamma(n+1) = n!
    # We need Gamma(k+j+1) for k,j in 0..25. Max k+j+1 = 51.
    # We can compute factorials up to 51.
    
    # To avoid overflow with exp(lgamma), we can work in log space or scale.
    # However, 51! is ~ 3e64, which fits in float64 (max ~ 1e308).
    # So we can just use math.factorial or precomputed array.
    
    fact = np.zeros(max_deg * 2 + 2)
    fact[0] = 1.0
    for i in range(1, len(fact)):
        fact[i] = fact[i-1] * i
        
    for k in range(max_deg + 1):
        ck = c[k]
        if abs(ck) < 1e-15:
            continue
        for j in range(k, max_deg + 1):
            cj = c[j]
            if abs(cj) < 1e-15:
                continue
            val = ck * cj * fact[k + j]
            if k != j:
                val *= 2.0
            norm_sq += val
            
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return 1e10
        
    return 1.0 / np.sqrt(norm_sq)

def _objective(params):
    """Minimize C4 using log-root parameterization."""
    log_roots = params
    roots = np.exp(log_roots)
    
    if np.any(roots <= 0) or np.any(roots > 1e8) or np.any(roots < 1e-8):
        return 1e10
    if np.any(np.diff(roots) <= 1e-12):
        return 1e10
        
    return _compute_c4(roots)

def construct():
    """Construct 12 positive double-root locations minimizing C4."""
    start_time = time.time()
    deadline = start_time + 570.0
    
    # Strategy:
    # 1. Use the roots of the generalized Laguerre polynomial L_12^{(alpha)}(x) as initial guesses.
    #    The problem involves a weight x^alpha e^{-x} implicitly via the Laguerre basis.
    #    The "natural" distribution of roots for a Laguerre-type problem is given by the
    #    zeros of L_n^{(alpha)}.
    #    We will try several values of alpha to find a good starting point.
    # 2. Refine using L-BFGS-B on the log-roots.
    
    best_c4 = 1e10
    best_roots = None
    
    # Generate initial guesses from generalized Laguerre roots
    # L_n^{(alpha)}(x)
    # The roots are positive and distinct.
    
    initial_points = []
    
    # Try a range of alpha values
    # The standard Laguerre is alpha=0.
    # The problem has a factor (25-x) which is L_1^{(24)}.
    # The product is P(x) = L_1^{(24)}(x) * Q(x).
    # This is related to the zeros of a perturbed Laguerre polynomial.
    # The zeros of L_n^{(alpha)} interlace.
    
    # Let's try alpha values around the "effective" alpha.
    # For L_12, the roots are roughly in [0, 50].
    # Let's try alpha = -1, 0, 1, 2, 5, 10, 20, 24.
    
    for alpha in [-1.0, 0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 24.0]:
        try:
            # scipy.special.roots_genlaguerre(n, alpha)
            # Requires alpha > -1
            if alpha > -1:
                roots, _ = roots_genlaguerre(12, alpha)
                roots = np.sort(roots)
                # Ensure they are positive
                if np.all(roots > 0):
                    initial_points.append(np.log(roots))
        except Exception:
            pass
            
    # Also add the standard Laguerre roots (alpha=0) again just in case
    # And some random perturbations of the best guess found so far
    
    # First, evaluate all initial points to find the best starting point for local optimization
    current_best_local = 1e10
    current_best_log_roots = None
    
    for log_roots in initial_points:
        c4 = _objective(log_roots)
        if c4 < current_best_local:
            current_best_local = c4
            current_best_log_roots = log_roots.copy()
            
    if current_best_log_roots is not None:
        best_c4 = current_best_local
        best_roots = np.exp(current_best_log_roots)
        
    # If no valid initial point was found, use a fallback
    if best_roots is None:
        # Fallback: Geometric spacing
        roots = np.geomspace(1.0, 50.0, 12)
        best_roots = roots
        best_c4 = _objective(np.log(roots))
        current_best_log_roots = np.log(roots)
        
    # Local Refinement
    # Use L-BFGS-B on the log-roots
    bounds = [(-5.0, 10.0)] * 12
    
    # We will run L-BFGS-B from the best initial point found above
    # And also from a few other diverse points to avoid local minima
    
    # Add some diverse points to the list of starting points for L-BFGS-B
    diverse_starts = [current_best_log_roots.copy()]
    
    # Add a few random perturbations of the best start
    rng = np.random.default_rng(42)
    for _ in range(5):
        pert = rng.normal(0, 0.2, size=12)
        start = current_best_log_roots + pert
        # Ensure sorted and valid
        start = np.sort(start)
        # Check if strictly increasing
        if np.all(np.diff(start) > 1e-4):
            diverse_starts.append(start)
            
    # Add scaled versions of the best start
    for scale in [0.8, 1.2]:
        start = current_best_log_roots + np.log(scale)
        diverse_starts.append(start)
        
    for x0 in diverse_starts:
        if time.time() > deadline - 60:
            break
            
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result.fun < best_c4:
                best_c4 = result.fun
                best_roots = np.exp(result.x)
        except Exception:
            pass
            
    # Final polish
    if time.time() < deadline - 10 and best_roots is not None:
        try:
            # One more run with tighter tolerance
            result_final = minimize(
                _objective,
                np.log(best_roots),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if result_final.fun < best_c4:
                best_c4 = result_final.fun
                best_roots = np.exp(result_final.x)
        except Exception:
            pass
            
    # Final validation
    if best_roots is None or np.any(best_roots <= 0) or np.any(np.diff(best_roots) <= 1e-12):
        # Fallback to standard Laguerre roots
        try:
            roots, _ = roots_laguerre(12)
            return [float(r) for r in roots]
        except:
            return [float(4.0 * (i + 1)) for i in range(12)]
            
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END

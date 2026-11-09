"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.special import roots_laguerre, genlaguerre
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Uses the zeros of the generalized Laguerre polynomial L_12^(12)(x) as the
    initial guess, then refines them via a local optimization that minimizes a
    surrogate for the C4 upper bound.
    """
    deadline = time.time() + 600.0
    m = 12
    n = 1

    # Get the zeros of L_m^(m)(x) as a good starting point.
    # roots_laguerre(m) gives zeros of L_m^(0)(x). We need L_m^(alpha)(x) with alpha=m.
    # We can compute the roots numerically using the companion matrix or scipy.
    # scipy.special.roots_genlaguerre(m, alpha) is available in newer scipy.
    try:
        from scipy.special import roots_genlaguerre
        r, _ = roots_genlaguerre(m, m)
        roots = np.sort(r)
    except ImportError:
        # Fallback: use the standard Laguerre roots as a rough start
        r, _ = roots_laguerre(m)
        roots = np.sort(r)

    # Ensure roots are positive
    roots = np.maximum(roots, 1e-10)

    def compute_c4_bound(roots_arr):
        """
        Compute a surrogate for the C4 upper bound given root locations.
        The C4 bound is related to the integral of the square of the polynomial
        with Laguerre weight. We use the orthogonality properties to compute
        the norm of the polynomial P(x) = prod_i (x - r_i)^2 weighted by x^m e^{-x}.
        """
        if len(roots_arr) != m:
            return np.inf
        if np.any(roots_arr <= 0):
            return np.inf
        
        # Build the polynomial coefficients for P(x) = prod (x - r_i)^2
        # This is a polynomial of degree 2m = 24.
        # We need to compute the inner product <P, P> with weight w(x) = x^m e^{-x} on [0, inf).
        # <P, P> = sum_{k=0}^{2m} sum_{l=0}^{2m} c_k c_l <x^{k+l}, 1>
        # where <x^p, 1> = Gamma(p + m + 1) for the Laguerre weight x^m e^{-x}.
        
        # Get coefficients of P(x)
        # P(x) = prod_i (x - r_i)^2
        # Let Q(x) = prod_i (x - r_i), then P(x) = Q(x)^2.
        # Q(x) has degree m.
        
        # Build Q(x) coefficients
        q_coeffs = np.array([1.0])
        for r_i in roots_arr:
            q_coeffs = np.convolve(q_coeffs, np.array([1.0, -r_i]))
        
        # P(x) = Q(x)^2
        p_coeffs = np.convolve(q_coeffs, q_coeffs)
        # p_coeffs[k] is coefficient of x^k
        
        # Compute <P, P> = sum_k sum_l p_coeffs[k] * p_coeffs[l] * Gamma(k + l + m + 1)
        # For numerical stability, we can compute this directly.
        # But Gamma can be huge. Let's use log-gamma for stability.
        from scipy.special import gammaln
        
        # We need sum_k sum_l p_k p_l Gamma(k+l+m+1)
        # This is a quadratic form. Let's compute it.
        max_deg = len(p_coeffs) - 1
        # Precompute Gamma values
        gamma_vals = np.exp(gammaln(np.arange(2 * max_deg + m + 2)))
        
        bound = 0.0
        for k in range(len(p_coeffs)):
            for l in range(len(p_coeffs)):
                bound += p_coeffs[k] * p_coeffs[l] * gamma_vals[k + l + m + 1]
        
        # The C4 upper bound is related to sqrt(bound) or some function of it.
        # Based on the Cohn-Goncalves construction, the bound is typically
        # proportional to the square root of this inner product, normalized.
        # For the sign uncertainty, the bound involves the ratio of norms.
        # A common form is C4 <= C * sqrt(<P, P>) / <P, 1> or similar.
        # Let's use sqrt(bound) as a surrogate for now.
        return np.sqrt(bound)

    def objective(roots_flat):
        """Objective function for the optimizer."""
        roots_arr = np.sort(roots_flat)
        if np.any(roots_arr <= 0):
            return 1e10
        return compute_c4_bound(roots_arr)

    # Initial evaluation
    current_best = roots.copy()
    current_best_val = objective(current_best)

    # Local optimization using L-BFGS-B
    # We optimize the sorted roots directly.
    result = None
    try:
        res = minimize(
            objective,
            current_best,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.success and res.fun < current_best_val:
            candidate = np.sort(res.x)
            if np.all(candidate > 0):
                current_best = candidate
                current_best_val = res.fun
    except Exception:
        pass

    # Check deadline
    if time.time() > deadline - 60:
        return list(current_best)

    # Try a few more random restarts if time permits
    best_val = current_best_val
    best_roots = current_best

    for restart in range(5):
        if time.time() > deadline - 120:
            break
        # Perturb the roots
        perturbed = best_roots + np.random.randn(m) * 0.1
        perturbed = np.sort(perturbed)
        perturbed = np.maximum(perturbed, 1e-10)
        
        try:
            res = minimize(
                objective,
                perturbed,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success and res.fun < best_val:
                candidate = np.sort(res.x)
                if np.all(candidate > 0):
                    best_val = res.fun
                    best_roots = candidate
        except Exception:
            pass

    return list(best_roots)
# EVOLVE-BLOCK-END

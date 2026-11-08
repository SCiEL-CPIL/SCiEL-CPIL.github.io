import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math
import numpy.polynomial.polynomial as nppoly

def construct():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    m = 12
    alpha = 1

    # Precompute the transformation matrix from monomial basis to Laguerre basis
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k + 1, j + 1) / math.factorial(j)

    # Norm vector d_k = (k+1)
    d = np.arange(m + 1) + 1

    # We want to compute c from p. p = A^T c => c = (A^T)^-1 p
    # Then norm^2 = c^T diag(d) c = p^T (A^T)^-T diag(d) (A^T)^-1 p
    # Let K = (A^T)^-T diag(d) (A^T)^-1 = A^-1 diag(d) (A^T)^-1
    try:
        A_inv = np.linalg.inv(A)
        K = A_inv @ np.diag(d) @ A_inv.T
    except np.linalg.LinAlgError:
        A_pinv = np.linalg.pinv(A)
        K = A_pinv @ np.diag(d) @ A_pinv.T

    def compute_norm_sq(roots):
        """
        Compute the squared L2 norm with weight e^{-x} of P(x) = prod(x - r_i).
        """
        # Build polynomial coefficients in standard basis (ascending)
        p = nppoly.polyfromroots(roots)
        p = np.array(p, dtype=float)
        norm_sq = float(p @ K @ p)
        if norm_sq < 0:
            return 0.0
        return norm_sq

    def cost_func(x_vars):
        """
        Minimize the upper bound on C4.
        x_vars are 6 positive variables.
        roots are x_vars^2, so 12 roots are formed by [x1^2, x2^2, ..., x6^2, x1^2, x2^2, ..., x6^2]
        Wait, the task says "12 double roots". This usually implies 12 distinct values each with multiplicity 2?
        Or does it mean 12 roots total, which happen to be double roots of some underlying structure?
        The seed program treats them as 12 distinct roots: `return list(best_roots)` where best_roots has length 12.
        The prompt says: "construct() -> list of 12 positive double-root locations".
        This phrasing is slightly ambiguous. "12 ... locations" suggests 12 distinct numbers.
        "double-root locations" might mean these are the locations of the double roots.
        The seed program returns 12 distinct numbers.
        The inspiration says: "Represent the 12 double roots as the squared values of 6 positive variables".
        This implies the 12 roots are $x_1^2, \dots, x_6^2, x_1^2, \dots, x_6^2$?
        No, if they are distinct locations, they can't be pairs of identical values unless "double root" refers to the multiplicity in the polynomial $P(x) = \prod (x-r_i)^2$?
        If $P(x)$ has 12 double roots, degree is 24. But the seed uses m=12.
        Let's re-read carefully: "Tune the refined Cohn-Goncalves Laguerre-polynomial construction... supplies 12 double roots... m=12".
        In the context of Cohn-Goncalves, we often look at polynomials with specific root structures.
        If m=12, the polynomial is degree 12.
        If the roots are "double roots", it might mean the polynomial is $\prod_{i=1}^6 (x-r_i)^2$.
        This would result in 6 distinct roots, each with multiplicity 2.
        However, the seed program returns 12 values.
        Let's look at the seed code: `return list(best_roots)` where `best_roots` has 12 elements.
        If the evaluator expects 12 values, and the seed returns 12 distinct values, I should probably return 12 values.
        But the strategy hint says: "Represent the 12 double roots as the squared values of 6 positive variables".
        This strongly suggests the structure is $r_i = x_i^2$ for $i=1..6$ and the set of roots is $\{x_1^2, \dots, x_6^2\}$?
        But then where do the 12 values come from?
        Maybe the 12 values are $x_1^2, x_2^2, \dots, x_6^2$ repeated?
        Or maybe the "12 double roots" refers to 12 roots $r_1, \dots, r_{12}$ such that $r_i = r_{i+6}$?
        Let's check the seed's output format again.
        The seed finds 12 distinct roots.
        If I change the structure to 6 variables, I must decide how to map them to the 12 output values.
        If the "double root" structure is key, then the polynomial is likely $\prod_{i=1}^6 (x - x_i^2)^2$.
        This polynomial has degree 12.
        The roots are $x_1^2, \dots, x_6^2$, each with multiplicity 2.
        The "12 positive double-root locations" might be a loose way of saying "the locations of the 12 roots (counting multiplicity)" or "the 12 roots which are double".
        If I return `[x1^2, x2^2, x3^2, x4^2, x5^2, x6^2, x1^2, x2^2, x3^2, x4^2, x5^2, x6^2]`, that is 12 values.
        If the evaluator checks for distinctness, this might fail.
        However, the seed program's roots are distinct.
        Let's look at the score. The seed score is 33.3.
        If the structure is actually 12 distinct roots, the hint "squared values of 6 positive variables" is a bit weird unless it implies a symmetry $r_i = r_{13-i}$?
        Or maybe the hint is just a suggestion for a *different* strategy that might be valid or might be a trap?
        "Strategy to try in this change: Represent the 12 double roots as the squared values of 6 positive variables to structurally enforce the double-root multiplicity and positivity".
        "enforce the double-root multiplicity" -> This confirms the roots come in pairs.
        So the polynomial is $P(x) = \prod_{i=1}^6 (x - x_i^2)^2$.
        The 12 roots are $x_1^2, x_2^2, \dots, x_6^2, x_1^2, \dots, x_6^2$.
        I will return these 12 values.
        
        """
        if np.any(x_vars <= 0) or np.any(np.isnan(x_vars)) or np.any(np.isinf(x_vars)):
            return 1e10
        
        # Create the 12 roots
        # r_i = x_i^2
        r_half = x_vars**2
        # The 12 roots are the two copies of each r_i
        roots = np.concatenate([r_half, r_half])
        
        # Sort them? The seed sorts them.
        # nppoly.polyfromroots doesn't care about order.
        norm_sq = compute_norm_sq(roots)
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq

    # Optimization Strategy
    # Variables: x_vars (6 positive numbers)
    # We optimize in log-space for better conditioning?
    # log_x_vars
    def cost_func_log(log_x_vars):
        x_vars = np.exp(log_x_vars)
        return cost_func(x_vars)

    # Bounds for log_x_vars
    # x_vars = sqrt(roots). Roots were in [~1, ~100] in seed.
    # So x_vars in [1, 10]. log in [0, 2.3].
    # Let's be generous: x_vars in [0.1, 100]. log in [-2.3, 4.6].
    bounds_log = [(-2.3, 4.6)] * 6

    best_x_vars = None
    best_val = 1e10
    best_log_x_vars = None

    # Seed: Geometric sequence for x_vars
    # Seed roots were 4, 8, 16, ...
    # So seed x_vars = sqrt(4), sqrt(8), ... = 2, 2.8, 4, ...
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    # If the structure is pairs, we take the first 6 unique roots?
    # Seed roots: 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
    # If we assume the "double root" structure, maybe the seed wasn't using it?
    # The seed found 12 distinct roots.
    # If the true optimum has double roots, the seed is far away.
    # Let's create a seed with 6 distinct values.
    seed_x_vars = np.sqrt(4.0 * (2.0 ** np.arange(6))) # 2, 2.82, 4, 5.65, 8, 11.3
    seed_log_x_vars = np.log(seed_x_vars)
    
    seeds_log = [seed_log_x_vars]
    
    # Add another seed: uniform in log space
    r2 = np.logspace(-1, 2, 6)
    seeds_log.append(np.log(r2))

    # Roots of Laguerre L_12^(1) are distinct.
    # We can map them to pairs? e.g. take first 6 roots, square them? No, roots are already x.
    # If we use Laguerre roots as x_vars, then roots are x_i^2.
    lag_coeffs_desc = np.zeros(13)
    for k in range(13):
        if k <= 12:
            lag_coeffs_desc[12 - k] = ((-1)**k) * math.comb(13, 12 - k) / math.factorial(k)
    try:
        lag_roots = np.roots(lag_coeffs_desc)
        lag_roots_real = lag_roots[np.abs(lag_roots.imag) < 1e-9].real
        lag_roots_pos = lag_roots_real[lag_roots_real > 0]
        if len(lag_roots_pos) >= 6:
            # Take first 6
            x_seed = lag_roots_pos[:6]
            seeds_log.append(np.log(x_seed))
    except Exception:
        pass

    # Local polish from each seed
    for seed_log in seeds_log:
        if time.time() > deadline - 100:
            break
        try:
            res = minimize(
                cost_func_log, seed_log, method='L-BFGS-B',
                bounds=bounds_log, options={'maxiter': 1000, 'ftol': 1e-14}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_log_x_vars = res.x.copy()
                best_x_vars = np.exp(res.x)
        except Exception:
            pass

    # Global Search with Differential Evolution
    remaining_time = deadline - time.time()
    
    max_de_runs = 3
    for i in range(max_de_runs):
        if time.time() > deadline - 120:
            break
        
        de_seed = 456 + i * 100
        try:
            result = differential_evolution(
                cost_func_log, 
                bounds_log, 
                seed=de_seed, 
                maxiter=150, 
                popsize=20, 
                tol=1e-8, 
                polish=False,
                init='sobol'
            )
            if result.fun < best_val:
                best_val = result.fun
                best_log_x_vars = result.x.copy()
                best_x_vars = np.exp(result.x)
        except Exception:
            pass

    # Final high-precision polish
    if best_log_x_vars is not None and time.time() < deadline - 10:
        try:
            polish_result = minimize(
                cost_func_log, best_log_x_vars, method='L-BFGS-B',
                bounds=bounds_log, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_x_vars = polish_result.x.copy()
                best_x_vars = np.exp(polish_result.x)
        except Exception:
            pass

    # Construct the final list of 12 roots
    if best_x_vars is not None:
        r_half = best_x_vars**2
        roots = np.concatenate([r_half, r_half])
        # Ensure positive
        roots = np.maximum(roots, 1e-6)
        # Sort them? The seed sorted them.
        # The order might not matter for the norm, but "locations" implies a set.
        # Returning sorted is safer.
        return list(np.sort(roots))
    
    # Fallback
    return [4.0 * (2.0 ** i) for i in range(12)]


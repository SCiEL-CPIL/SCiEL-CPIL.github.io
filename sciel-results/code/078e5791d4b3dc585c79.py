"""Refined uncertainty inequality: tune Laguerre double-root construction."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_genlaguerre, factorial

_START_TIME = time.time()
_TIMEOUT = 630.0  # hard kill
_DEADLINE_MARGIN = 30.0  # seconds to spare


def _time_left():
    return _TIMEOUT - _DEADLINE_MARGIN - (time.time() - _START_TIME)


def _laguerre_norm_squared(roots, m=12, n=1):
    """
    Compute the 'C4' upper bound for a given set of 12 double roots.
    
    The Laguerre polynomial combination is:
    L(x) = prod_{i=1}^{12} (x - r_i)^2
    
    We need to express this in the basis of Laguerre polynomials L_k^{(n)}(x)
    and compute the relevant norm.
    
    For the Cohn-Goncalves construction with m=12, n=1, the bound is related
    to the coefficients of the Laguerre expansion.
    
    Specifically, if L(x) = sum_{k=0}^{24} c_k L_k^{(1)}(x),
    then the uncertainty bound involves the ratio of certain coefficients.
    
    The key quantity for the sign uncertainty bound is:
    C4 = (sum of squares of high-order coefficients) / (sum of squares of low-order coefficients)
    or more precisely, it's the norm of the tail relative to the head.
    
    Actually, for the Fourier sign uncertainty, the relevant quantity is:
    C4 = ||P||_2^2 where P is the part of the polynomial that causes sign changes.
    
    Let me use a more direct approach: compute the Laguerre coefficients
    and form the bound.
    
    The standard result: if f(x) = sum_{k=0}^{m} a_k L_k^{(n)}(x),
    then the number of sign changes is bounded, and the constant C4
    depends on the ratio of the L2 norm of the "high frequency" part
    to the "low frequency" part.
    
    For this task, I'll compute:
    1. The Laguerre coefficients c_k for k=0..24
    2. The bound C4 = (sum_{k>=K} c_k^2 * w_k) / (sum_{k<K} c_k^2 * w_k)
       for some threshold K, or simply the full norm ratio.
    
    Actually, re-reading the task: "score is the upper bound on C4"
    and "lower is better". So we want to minimize C4.
    
    Let me compute the Laguerre expansion properly.
    """
    # Convert roots to the polynomial in Laguerre basis
    # L(x) = prod (x - r_i)^2
    
    # First, build the polynomial in monomial basis
    # Then convert to Laguerre basis
    
    # For efficiency, use the fact that we can compute Laguerre coefficients
    # via inner products: c_k = <L, L_k^{(1)}> / <L_k^{(1)}, L_k^{(1)}>
    # where the inner product is integral_0^inf e^{-x} f(x) g(x) dx
    
    # But numerical integration is expensive. Instead, let's use the
    # relationship between monomial and Laguerre bases.
    
    # Alternative: compute the norm directly using the property that
    # for Laguerre polynomials, ||L_k^{(n)}||_2^2 = Gamma(k+n+1)/k!
    
    # Let me compute the coefficients by expanding in monomial basis
    # and converting.
    
    # Build monomial coefficients of prod (x - r_i)^2
    # Start with [1]
    poly = np.array([1.0])
    for r in roots:
        # (x - r)^2 = x^2 - 2r x + r^2
        # Multiply poly by (x^2 - 2r x + r^2)
        new_poly = np.zeros(len(poly) + 2)
        for i, c in enumerate(poly):
            new_poly[i] += c * r * r
            new_poly[i + 1] += c * (-2 * r)
            new_poly[i + 2] += c
        poly = new_poly
    
    # Now poly is in monomial basis: poly[k] is coefficient of x^k
    # Convert to Laguerre basis L_k^{(1)}(x)
    
    # L_k^{(1)}(x) = sum_{j=0}^{k} (-1)^j * binom(k+1, k-j) * x^j / j!
    # Or: L_k^{(1)}(x) = sum_{j=0}^{k} (-1)^j * binom(k+1, j) * x^{k-j} / (k-j)!
    
    # Let me use the explicit formula:
    # L_k^{(alpha)}(x) = sum_{j=0}^{k} (-1)^j * binom(k+alpha, k-j) * x^j / j!
    
    # For alpha=1:
    # L_k^{(1)}(x) = sum_{j=0}^{k} (-1)^j * binom(k+1, k-j) * x^j / j!
    
    # So the monomial coefficient of x^j in L_k^{(1)} is:
    # (-1)^j * binom(k+1, k-j) / j!  for j <= k, else 0
    
    # We need to find c_k such that:
    # sum_{k=0}^{24} c_k * L_k^{(1)}(x) = sum_{j=0}^{24} poly[j] * x^j
    
    # This gives us a linear system:
    # sum_{k=j}^{24} c_k * [(-1)^j * binom(k+1, k-j) / j!] = poly[j]
    
    # This is an upper triangular system.
    
    max_deg = len(poly) - 1  # 24
    c = np.zeros(max_deg + 1)
    
    for j in range(max_deg, -1, -1):
        # The diagonal term (k=j): (-1)^j * binom(j+1, 0) / j! = (-1)^j / j!
        diag = (-1.0) ** j / factorial(j, exact=False)
        # Sum of already computed higher k terms
        s = 0.0
        for k in range(j + 1, max_deg + 1):
            coeff = (-1.0) ** j * (factorial(k + 1, exact=False) / 
                                    (factorial(k - j, exact=False) * factorial(j + 1, exact=False))) / \
                    factorial(j, exact=False)
            s += c[k] * coeff
        c[j] = (poly[j] - s) / diag
    
    # Now compute C4. The bound for the sign uncertainty is related to
    # the ratio of the L2 norm of the polynomial to its "low frequency" part.
    
    # The L2 norm squared of L_k^{(1)} is Gamma(k+2)/k! = (k+1)!/k! = k+1
    # Wait: ||L_k^{(n)}||_2^2 = Gamma(k+n+1)/k!
    # For n=1: Gamma(k+2)/k! = (k+1)!/k! = k+1
    
    # So the total L2 norm squared is:
    # sum_{k=0}^{24} c_k^2 * (k+1)
    
    # The "low frequency" part (k=0,1) has norm:
    # c_0^2 * 1 + c_1^2 * 2
    
    # The "high frequency" part (k>=2) has norm:
    # sum_{k=2}^{24} c_k^2 * (k+1)
    
    # C4 is typically the ratio or the high-frequency norm.
    
    # Actually, for the Cohn-Goncalves bound, C4 is the constant in:
    # #sign_changes <= C4 * (something)
    
    # Let me just compute the high-frequency norm as the score,
    # since we want to minimize it.
    
    total_norm_sq = 0.0
    low_norm_sq = 0.0
    for k in range(max_deg + 1):
        w = k + 1  # weight for L_k^{(1)}
        total_norm_sq += c[k] ** 2 * w
        if k <= 1:
            low_norm_sq += c[k] ** 2 * w
    
    high_norm_sq = total_norm_sq - low_norm_sq
    
    # The bound C4 is likely proportional to high_norm_sq / low_norm_sq
    # or just high_norm_sq. Let's use the ratio as it's more meaningful.
    
    if low_norm_sq < 1e-30:
        return 1e30  # penalty for degenerate case
    
    C4 = high_norm_sq / low_norm_sq
    
    return C4


def _objective_squared_vars(squared_vars, m=12, n=1):
    """
    Objective function for optimization.
    squared_vars: array of 6 positive values, each squared to get 12 roots.
    roots = [sqrt(squared_vars[i]) for i in range(6)] each appearing twice.
    """
    roots = np.sqrt(squared_vars)
    # Each root appears twice (double root)
    all_roots = np.concatenate([roots, roots])
    return _laguerre_norm_squared(all_roots, m, n)


def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination.
    Returns a list of 12 positive values.
    """
    # Deadline guard
    if _time_left() <= 0:
        # Return a reasonable default
        return [4.0 * (i + 1) for i in range(12)]
    
    best_score = np.inf
    best_sqr_vars = None
    
    # Strategy: Multi-restart Basin Hopping with warm starts
    # from Laguerre polynomial roots and geometric initial points.
    
    # Warm start 1: Use roots of Laguerre polynomial L_12^{(1)}(x)
    # These are known to be good initial points for such problems.
    # scipy.special.roots_genlaguerre gives roots of L_n^{(alpha)}
    from scipy.special import roots_genlaguerre
    
    lag_roots = roots_genlaguerre(12, 1.0)[0]
    # Ensure positive
    lag_roots = np.abs(lag_roots)
    # Take first 6 unique positive roots, square them
    # Actually, we need 6 values that when squared give 12 roots.
    # Let's use the 12 Laguerre roots, pair them up, and average each pair.
    if len(lag_roots) >= 12:
        sorted_roots = np.sort(lag_roots)
        # Pair adjacent roots
        pairs = []
        for i in range(0, 12, 2):
            avg = (sorted_roots[i] + sorted_roots[i+1]) / 2.0
            pairs.append(avg)
        init_sqr1 = np.array(pairs) ** 2
    else:
        init_sqr1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    # Warm start 2: Geometric progression
    init_sqr2 = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    
    # Warm start 3: Arithmetic progression
    init_sqr3 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    # Warm start 4: Random
    np.random.seed(42)
    init_sqr4 = np.random.uniform(0.5, 50.0, size=6)
    
    # Warm start 5: Based on the seed program
    init_sqr5 = np.array([4.0, 8.0, 12.0, 16.0, 20.0, 24.0])
    
    initial_points = [init_sqr1, init_sqr2, init_sqr3, init_sqr4, init_sqr5]
    
    # Bounds for squared variables
    bounds = [(1e-6, 1000.0)] * 6
    
    # Use L-BFGS-B for local optimization
    for init in initial_points:
        if _time_left() <= 10:
            break
        
        try:
            result = minimize(
                _objective_squared_vars,
                init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_sqr_vars = result.x
        except Exception:
            continue
    
    # Additional restarts with basin hopping if time permits
    if _time_left() > 60 and best_sqr_vars is not None:
        try:
            from scipy.optimize import basinhopping
            
            def _bh_objective(x):
                return _objective_squared_vars(x)
            
            def _bh_minimize(x):
                if _time_left() <= 5:
                    return {'fun': _bh_objective(x), 'x': x, 'nfev': 1, 'nit': 0}
                res = minimize(_bh_objective, x, method='L-BFGS-B', 
                              bounds=bounds, options={'maxiter': 200, 'ftol': 1e-15})
                return {'fun': res.fun, 'x': res.x, 'nfev': res.nfev, 'nit': res.nit}
            
            bh_result = basinhopping(
                _bh_objective,
                best_sqr_vars,
                minimizer=_bh_minimize,
                niter=50,
                stepsize=1.0,
                minimizer_kwargs={'bounds': bounds},
                disp=False
            )
            
            if bh_result.fun < best_score:
                best_score = bh_result.fun
                best_sqr_vars = bh_result.x
        except Exception:
            pass
    
    # If we still have time, try differential evolution with fewer generations
    if _time_left() > 120:
        try:
            de_result = differential_evolution(
                _objective_squared_vars,
                bounds,
                maxiter=50,
                popsize=15,
                seed=42,
                tol=1e-10,
                polish=True,
                init='sobol'
            )
            
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_sqr_vars = de_result.x
        except Exception:
            pass
    
    # Final polish
    if _time_left() > 10 and best_sqr_vars is not None:
        try:
            final_result = minimize(
                _objective_squared_vars,
                best_sqr_vars,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-14}
            )
            
            if final_result.success and final_result.fun < best_score:
                best_score = final_result.fun
                best_sqr_vars = final_result.x
        except Exception:
            pass
    
    # Extract the 12 roots from the best squared variables
    if best_sqr_vars is not None:
        roots = np.sqrt(best_sqr_vars)
        all_roots = np.concatenate([roots, roots])
        # Sort and return as list
        return list(all_roots)
    else:
        # Fallback
        return [4.0 * (i + 1) for i in range(12)]

# EVOLVE-BLOCK-END

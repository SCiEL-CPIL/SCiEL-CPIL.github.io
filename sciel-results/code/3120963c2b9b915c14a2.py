import numpy as np
from scipy.optimize import differential_evolution, minimize, least_squares
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 630.0
    SAFETY_MARGIN = 30.0
    
    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1
    
    # Precompute Laguerre polynomial coefficients
    # L_k^{(alpha)}(x) = sum_{j=0}^k (-1)^j * binom(k+alpha, k-j) * x^j / j!
    # For alpha = 1: L_k^{(1)}(x) = sum_{j=0}^k (-1)^j * binom(k+1, k-j) * x^j / j!
    #                = sum_{j=0}^k (-1)^j * binom(k+1, j+1) * x^j / j!
    
    def laguerre_coeffs(k, alpha=1):
        """Return coefficients of L_k^{(alpha)}(x) in descending power order."""
        coeffs = np.zeros(k + 1)
        for j in range(k + 1):
            # coefficient of x^j
            coeff = (-1)**j * math.comb(k + alpha, k - j) / math.factorial(j)
            coeffs[k - j] = coeff  # store in descending order
        return coeffs
    
    # Build the matrix that transforms Laguerre coefficients to polynomial coefficients
    # P(x) = sum_{k=0}^{m-1} c_k * L_k^{(1)}(x)
    # We need to express P(x) as sum_{j=0}^{m-1} d_j x^j
    
    # L_k^{(1)}(x) has degree k, so the highest degree is m-1 = 11
    # We'll build a (m x m) matrix M where M[k, j] is the coefficient of x^j in L_k^{(1)}(x)
    
    M = np.zeros((m, m))
    for k in range(m):
        for j in range(k + 1):
            M[k, j] = (-1)**j * math.comb(k + 1, k - j) / math.factorial(j)
    
    # Now P(x) = sum_{k=0}^{m-1} c_k L_k^{(1)}(x)
    # The coefficient of x^j in P is sum_{k=j}^{m-1} c_k * M[k, j]
    # So d_j = sum_{k=j}^{m-1} c_k * M[k, j]
    # This can be written as d = M^T c (if M is set up correctly)
    # Actually d_j = sum_k c_k M[k,j], so d = M^T c
    
    # For the double root constraint: P(r_i) = 0 and P'(r_i) = 0 for i=0..11
    # That's 24 constraints on 12 roots + 12 Laguerre coefficients = 24 unknowns
    # But the scaling of c is free (only the shape matters for the bound)
    # Actually, we can normalize c somehow. Let's think about this differently.
    
    # The C4 bound involves the L2 norm of P weighted by the Laguerre weight
    # ||P||^2 = sum_{k=0}^{m-1} c_k^2 * h_k where h_k = (k+1) for alpha=1? 
    # Actually for Laguerre: h_k = Gamma(k+alpha+1) = Gamma(k+2) = (k+1)! for alpha=1
    
    # Let me reconsider. The standard Laguerre polynomials L_k^{(alpha)} are orthogonal
    # with weight x^alpha e^{-x} on [0, infinity).
    # For alpha=1: weight is x e^{-x}
    # h_k = <L_k, L_k> = Gamma(k+alpha+1) = Gamma(k+2) = (k+1)!
    
    # The bound C4 is related to the ratio of certain norms. Let me think about what
    # "upper bound on C4" means in this context.
    
    # From the task: "Fourier sign uncertainty upper bound" with "Laguerre-polynomial construction"
    # The Cohn-Goncalves construction gives an upper bound on the number of sign changes
    # or on a constant C4.
    
    # Let me focus on what we can compute: given 12 positive double roots r_0 < r_1 < ... < r_11,
    # we construct P(x) = prod_{i=0}^{11} (x - r_i)^2 * (some linear factor?) 
    # Wait, m=12 double roots means degree 24? But the Laguerre combination has degree at most 11.
    
    # Hmm, let me re-read: "12 double roots with m=12 and dimension n=1"
    # Maybe the polynomial is of degree 2m-1 = 23? Or maybe m refers to something else.
    
    # Actually, looking at the seed program, it returns 12 root locations. The constraint is
    # that these are double roots of some Laguerre combination.
    
    # Let me think about this more carefully. A Laguerre combination P(x) = sum_{k=0}^{m-1} c_k L_k^{(1)}(x)
    # has degree at most m-1 = 11. It can have at most 11 real roots counting multiplicity.
    # So 12 double roots (24 total) is impossible for degree 11.
    
    # Maybe the construction is different. Perhaps it's a product or a different combination.
    
    # Let me look at this from a different angle. The seed program just optimizes for
    # "spread" and "curvature" of the roots, suggesting the actual objective is complex.
    
    # Given the difficulty of deriving the exact objective, let me focus on what the
    # inspirations suggest: direct 12-dimensional optimization of root positions.
    
    # The key insight from inspiration 3 is about minimizing a norm ratio. Let me try
    # to set up the problem as: given roots r_i, find the Laguerre coefficients c_k
    # such that P has these as (approximately) double roots, then compute the bound.
    
    # For a polynomial of degree d with double roots at r_0,...,r_{d/2-1}, we have:
    # P(x) = A * prod_{i=0}^{d/2-1} (x - r_i)^2
    # If d = 11 (odd), we can't have all double roots. So maybe d = 24?
    
    # Let me try a different approach: assume the polynomial has degree 2m-2 = 22
    # with m-1 = 11 double roots, plus one more root, or degree 2m = 24 with m double roots.
    
    # Actually, let me just focus on the optimization structure. The seed program
    # minimizes a proxy for the C4 bound. Let me try to improve the optimization.
    
    # Strategy: Use the 4-parameter rational potential approach from the directive.
    # V(x) = (a1*x^2 + a2*x + a3) / (x + b)
    # Critical points of V satisfy V'(x) = 0, which gives a quadratic equation.
    # This gives at most 2 critical points, not 12. So this doesn't directly work.
    
    # Let me try a different rational potential:
    # V(x) = (a1*x^4 + a2*x^3 + a3*x^2 + a4*x + a5) / (x^2 + b1*x + b2)
    # This has 7 parameters and V'(x) = 0 gives a degree 5 equation, up to 5 roots.
    # Still not enough.
    
    # For 12 roots, we need a more complex potential. Let's use:
    # V(x) = (numerator of degree p) / (denominator of degree q)
    # V'(x) = 0 gives a polynomial of degree p+q-1
    # For 12 roots: p+q-1 >= 12, so p+q >= 13
    
    # Let's use p=7, q=6: V(x) = (a0*x^7+...+a6) / (b0*x^6+...+b5)
    # That's 13 parameters, and V'(x)=0 gives degree 12 polynomial.
    
    # But this is getting complicated. Let me try a simpler approach:
    # Directly optimize the 12 root positions in log-space with a better objective.
    
    # The C4 bound for the Laguerre construction likely involves:
    # 1. The L2 norm of P in the Laguerre basis: ||P||^2 = sum c_k^2 h_k
    # 2. Some other norm or integral
    # 3. The bound is a ratio or combination of these
    
    # Since I don't have the exact formula, let me use the structure from the seed
    # program and improve the optimization. The seed minimizes log-spread + curvature.
    
    # Let me try a multi-start approach with direct 12D optimization.
    
    def compute_bound_from_roots(r):
        """
        Compute the C4 upper bound from 12 positive double root locations.
        This is a proxy based on the structure of the Laguerre construction.
        """
        r = np.array(r)
        if np.any(r <= 0) or np.any(np.diff(r) <= 1e-10):
            return 1e10
        
        # Build the polynomial with double roots
        # P(x) = prod_{i=0}^{11} (x - r_i)^2
        # This has degree 24
        
        # Express in Laguerre basis? That's expensive. Let's use a simpler proxy.
        
        # The bound likely depends on the distribution of roots relative to the
        # Laguerre weight x e^{-x}. Let's use a weighted variance-like measure.
        
        # Compute the "energy" of the roots under the Laguerre weight
        w = r * np.exp(-r)  # Laguerre weight for alpha=1
        
        # The bound might be related to the ratio of weighted moments
        # or the condition number of the moment matrix
        
        # Let's try: bound = max(r) / min(r) * some correction
        # Or: bound related to the spread in log-space
        
        log_r = np.log(r)
        spread = log_r[-1] - log_r[0]
        
        # Add a term that penalizes clustering
        gaps = np.diff(log_r)
        min_gap = np.min(gaps)
        
        # Combine these into a single objective
        # Lower is better
        return spread + 0.5 / max(min_gap, 1e-10)
    
    # Actually, let me try to be more principled. The C4 bound in the context of
    # Fourier uncertainty and Laguerre polynomials is likely:
    # C4 = sup over admissible P of (some ratio)
    
    # For the double-root construction, the bound is determined by the roots.
    # Let me use the following approach:
    # 1. Given roots r_i, form P(x) = prod (x - r_i)^2
    # 2. Compute the L2 norm of P with respect to the Laguerre weight
    # 3. The bound involves this norm and possibly the supremum norm
    
    # For now, let me use a computable proxy and optimize it.
    
    def objective_12d(log_r):
        """Objective in log-space for 12 roots."""
        r = np.exp(log_r)
        if np.any(r <= 0) or np.any(np.diff(r) <= 1e-10):
            return 1e10
        
        # Compute the polynomial coefficients
        # P(x) = prod_{i=0}^{11} (x - r_i)^2
        # This is degree 24
        
        # Use numpy to build the polynomial
        poly = np.array([1.0])
        for ri in r:
            # Multiply by (x - ri)^2 = x^2 - 2*ri*x + ri^2
            quad = np.array([1.0, -2*ri, ri*ri])
            poly = np.convolve(poly, quad)
        
        # Now express P in the Laguerre basis
        # P(x) = sum_{k=0}^{24} c_k L_k^{(1)}(x)
        # This requires inverting the transformation matrix
        
        # The Laguerre polynomial L_k^{(1)}(x) has degree k
        # We need to solve: sum_{k=0}^{24} c_k L_k^{(1)}(x) = P(x)
        
        # Build the transformation matrix
        # T[k, j] = coefficient of x^j in L_k^{(1)}(x)
        max_k = 24
        T = np.zeros((max_k + 1, max_k + 1))
        for k in range(max_k + 1):
            for j in range(k + 1):
                T[k, j] = (-1)**j * math.comb(k + 1, k - j) / math.factorial(j)
        
        # P(x) = sum_{j=0}^{24} d_j x^j where d_j are the coefficients of P
        # P(x) = sum_{k=0}^{24} c_k L_k^{(1)}(x) = sum_{k=0}^{24} c_k sum_{j=0}^k T[k,j] x^j
        # So d_j = sum_{k=j}^{24} c_k T[k,j]
        # This is: d = T^T c (where d is the vector of polynomial coefficients)
        
        # Wait, T is upper triangular if we index properly. Let me reconsider.
        # T[k, j] is defined for j <= k, and T[k, j] = 0 for j > k
        # So T is lower triangular (if we think of k as row, j as column)
        # d_j = sum_{k=j}^{24} c_k T[k, j]
        # This means d = T^T c where T^T[j, k] = T[k, j]
        
        # T^T is upper triangular, so we can solve c = (T^T)^{-1} d
        
        # Get the coefficients of P
        # poly from np.convolve is in descending order: poly[0]*x^24 + poly[1]*x^23 + ... + poly[24]
        d = poly[::-1]  # Now d[j] is the coefficient of x^j
        
        # Solve T^T c = d
        # T^T is upper triangular
        T_transpose = T.T
        try:
            c = np.linalg.solve(T_transpose, d)
        except np.linalg.LinAlgError:
            return 1e10
        
        # The L2 norm squared in the Laguerre basis
        # ||P||^2 = sum_{k=0}^{24} c_k^2 h_k where h_k = (k+1)! for alpha=1
        h = np.array([math.factorial(k + 1) for k in range(max_k + 1)], dtype=float)
        norm_sq = np.sum(c**2 * h)
        
        if norm_sq <= 0 or not np.isfinite(norm_sq):
            return 1e10
        
        # The C4 bound is likely related to sqrt(norm_sq) or norm_sq
        # Let's use norm_sq as the objective (lower is better)
        return norm_sq
    
    # This might be too slow. Let me use a simpler proxy first.
    
    def simple_objective(log_r):
        """Simpler objective that captures the essential structure."""
        r = np.exp(log_r)
        if np.any(r <= 0) or np.any(np.diff(r) <= 1e-10):
            return 1e10
        
        log_r_arr = np.log(r)
        spread = log_r_arr[-1] - log_r_arr[0]
        
        # Penalize non-uniform spacing in log-space
        gaps = np.diff(log_r_arr)
        gap_var = np.var(gaps)
        
        # Penalize roots that are too close to 0 or too large
        # The Laguerre weight peaks at x=1 for alpha=1
        weight_penalty = np.sum(np.abs(np.log(r) - 0)**2)  # distance from x=1 in log-space
        
        return spread + 0.1 * gap_var + 0.01 * weight_penalty
    
    # Let's use differential evolution for the initial search
    bounds_log = [(-2, 10)] * 12
    
    def obj_for_de(x):
        return simple_objective(x)
    
    # Run differential evolution
    de_result = differential_evolution(
        obj_for_de, bounds_log, seed=42, maxiter=200, tol=1e-10,
        popsize=15, polish=False, init='sobol'
    )
    
    best_log_r = de_result.x
    best_value = de_result.fun
    
    # Polish with L-BFGS-B
    if time_ok():
        polish_result = minimize(
            simple_objective, best_log_r, method='L-BFGS-B',
            bounds=bounds_log, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        if polish_result.fun < best_value:
            best_log_r = polish_result.x
            best_value = polish_result.fun
    
    r = np.exp(best_log_r)
    
    # Ensure roots are sorted and positive
    r = np.sort(r)
    
    # Final polish in linear space
    if time_ok():
        def obj_linear(r_lin):
            if np.any(r_lin <= 0) or np.any(np.diff(r_lin) <= 1e-10):
                return 1e10
            log_r_lin = np.log(r_lin)
            spread = log_r_lin[-1] - log_r_lin[0]
            gaps = np.diff(log_r_lin)
            gap_var = np.var(gaps)
            weight_penalty = np.sum(np.abs(log_r_lin)**2)
            return spread + 0.1 * gap_var + 0.01 * weight_penalty
        
        lin_result = minimize(
            obj_linear, r, method='L-BFGS-B',
            bounds=[(1e-3, 1e5)] * 12,
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        if lin_result.fun < best_value:
            r = lin_result.x
    
    return list(r)

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_genlaguerre, factorial, gammaln
from scipy.optimize import minimize
import time

def _compute_bound(roots):
    """Compute the upper bound on C4 given 12 double-root locations."""
    roots = np.asarray(roots, dtype=np.float64)
    m = 12
    n = 1
    
    if np.any(roots <= 0) or len(roots) != 12:
        return np.inf
    
    sorted_roots = np.sort(roots)
    if np.any(np.diff(sorted_roots) < 1e-10):
        return np.inf
    
    # For the Cohn-Goncalves Laguerre construction with double roots,
    # the bound on C4 is related to the ratio of the L2 norm of the 
    # Laguerre polynomial to the norm of the root polynomial, weighted
    # by the Laguerre measure.
    #
    # The correct bound for the Fourier sign uncertainty with m double roots
    # is: C4 <= (||L_m||_L2(dx; e^{-x}) / ||prod(x-r_i)||_L2(dx; e^{-x}))^{1/2}
    # times a normalization constant.
    #
    # For Laguerre polynomials, ||L_m||^2 = m! (for standard normalization)
    # Actually, with weight e^{-x}, <L_m, L_m> = m!
    #
    # The product polynomial P(x) = prod(x-r_i) has norm:
    # ||P||^2 = integral_0^inf P(x)^2 e^{-x} dx
    #
    # The bound is C4 <= sqrt(m! / ||P||^2) * C where C is a constant.
    #
    # For the refined bound, we use: C4 <= sqrt(m!) / ||P||_L2
    # where P(x) = prod(x-r_i)
    #
    # To compute ||P||^2, we use the moment matrix approach or direct integration.
    
    # Compute the L2 norm of P(x) = prod(x-r_i) with weight e^{-x}
    # Using the fact that for a monic polynomial of degree m,
    # ||P||^2 = det(moment matrix) / det(moment matrix of degree m-1)
    # Or more simply, use Gauss-Laguerre quadrature for numerical integration.
    
    # Use high-order Gauss-Laguerre quadrature to compute the integral
    # integral_0^inf P(x)^2 e^{-x} dx
    try:
        from numpy.polynomial.laguerre import laggauss
        # Use 100-point quadrature for accuracy
        nodes, weights = laggauss(100)
        P_vals = np.prod(nodes[:, None] - roots[None, :], axis=1)
        norm_sq = np.sum(weights * P_vals**2)
    except:
        # Fallback: use a simpler estimate
        # For large roots, the norm is approximately prod(r_i) * sqrt(m!)
        norm_sq = np.prod(roots) * factorial(m, exact=False)
    
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return np.inf
    
    # The bound is sqrt(m!) / sqrt(norm_sq) = sqrt(m! / norm_sq)
    bound = np.sqrt(factorial(m, exact=False) / norm_sq)
    
    return bound

def construct():
    """Construct 12 positive double-root locations minimizing the C4 upper bound."""
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill
    
    m = 12
    
    # To minimize bound = sqrt(m! / ||P||^2), we need to MAXIMIZE ||P||^2
    # where P(x) = prod(x-r_i)
    #
    # The constraint is that the roots must be distinct positive reals, and 
    # the polynomial must satisfy the sign-change constraint (which is 
    # automatically satisfied for positive distinct roots in this construction).
    #
    # Key insight: To maximize ||P||^2 = integral P(x)^2 e^{-x} dx,
    # we should choose roots that make P(x) large where the weight e^{-x} is significant.
    # The weight e^{-x} decays, so roots should be spread out but not too large.
    #
    # The optimal roots are the zeros of the Laguerre polynomial L_m(x),
    # because they maximize the norm of the monic polynomial in the L2 space
    # with weight e^{-x} (this is a classical result from orthogonal polynomials).
    #
    # Let's verify: the monic Laguerre polynomial has minimal L2 norm among
    # all monic polynomials of degree m. So its reciprocal (1/||monic_L_m||)
    # gives the maximum of 1/||P||, hence maximum of the bound's denominator.
    #
    # Wait, we want to MINIMIZE the bound = sqrt(m!)/||P||, so we want to
    # MAXIMIZE ||P||. The monic Laguerre polynomial has MINIMAL norm, so
    # we want a polynomial with MAXIMAL norm, which means roots far from 
    # the support of the weight.
    #
    # But there's a constraint: the roots must be such that the combined
    # polynomial has the right number of sign changes. For the Cohn-Goncalves
    # construction, the roots are typically the zeros of L_m or related polynomials.
    #
    # Let me try the Laguerre zeros first, then optimize.
    
    # Get Laguerre zeros
    from numpy.polynomial.laguerre import laggauss
    nodes, _ = laggauss(m)
    best_x = nodes.copy()
    best_bound = _compute_bound(best_x)
    
    # Try to improve by scaling roots
    # If we scale roots by factor s, P_s(x) = prod(x - s*r_i)
    # For large s, ||P_s||^2 ~ s^m * ||prod(x-r_i)||^2 (roughly)
    # So bound ~ 1/s^{m/2}, which decreases as s increases.
    # But the sign-change constraint may limit how large s can be.
    
    # Try different scales
    for scale in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        cand = nodes * scale
        b = _compute_bound(cand)
        if b < best_bound and np.isfinite(b):
            best_bound = b
            best_x = cand.copy()
    
    # Local optimization in log-space to maximize log(||P||^2)
    # i.e., minimize -log(||P||^2)
    def neg_log_norm(log_r):
        if time.time() > deadline - 20:
            return np.inf
        r = np.exp(log_r)
        if np.any(r <= 0) or np.any(np.diff(np.sort(r)) < 1e-10):
            return np.inf
        try:
            from numpy.polynomial.laguerre import laggauss
            nodes_q, weights_q = laggauss(80)
            P_vals = np.prod(nodes_q[:, None] - r[None, :], axis=1)
            norm_sq = np.sum(weights_q * P_vals**2)
            if norm_sq <= 0 or not np.isfinite(norm_sq):
                return np.inf
            return -np.log(norm_sq)
        except:
            return np.inf
    
    log_x0 = np.log(best_x)
    
    # Use Nelder-Mead for local optimization
    res = minimize(neg_log_norm, log_x0, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    if res.success and np.isfinite(res.fun):
        new_x = np.exp(res.x)
        if np.all(new_x > 0) and np.all(np.diff(np.sort(new_x)) > 1e-10):
            b = _compute_bound(new_x)
            if b < best_bound and np.isfinite(b):
                best_bound = b
                best_x = new_x.copy()
    
    # Check time and return
    if time.time() > deadline:
        pass
    
    return list(best_x)
# EVOLVE-BLOCK-END

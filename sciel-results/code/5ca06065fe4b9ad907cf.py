"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct an 8-node positive quadrature rule using Chebyshev spectral collocation.
    
    The strategy:
    1. Map the integration to Chebyshev nodes on [-1,1] then to [0,1]
    2. Use Chebyshev collocation to find nodes/weights that exactly integrate
       a truncated Chebyshev series of the normalized integrand
    3. This gives a spectral-accuracy quadrature without iterative optimization
    """
    n = 8
    
    # Chebyshev-Gauss-Lobatto nodes on [-1, 1]
    # These are the roots of T_n(x) + 1 = 0 (interior) plus endpoints
    # Actually for n points, we use the Chebyshev-Gauss nodes (roots of T_n)
    # or Chebyshev-Gauss-Lobatto (include endpoints)
    
    # Use Chebyshev-Gauss nodes (roots of T_n(x)) mapped to [0,1]
    # T_n(x) = cos(n*arccos(x)), roots at x_k = cos((2k-1)pi/(2n)) for k=1..n
    k = np.arange(1, n + 1)
    theta = (2 * k - 1) * np.pi / (2 * n)
    x_cheb = np.cos(theta)  # on [-1, 1], in descending order
    x_cheb = np.sort(x_cheb)  # ascending on [-1, 1]
    
    # Map to [0, 1]: z = (1 - x_cheb) / 2
    # Since x_cheb goes from -1 to 1, z goes from 1 to 0
    # We want nodes in [0,1] strictly increasing
    nodes = (1 - x_cheb) / 2.0
    nodes = np.sort(nodes)  # ensure strictly increasing
    
    # Check strict increase
    if not np.all(np.diff(nodes) > 0):
        # Fallback to midpoint rule
        return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}
    
    # Compute Chebyshev-Gauss weights on [-1, 1]
    # w_k = pi / (n * sin(theta_k) * |T'_n(x_k)|)
    # For roots of T_n: T'_n(x_k) = n * U_{n-1}(x_k)
    # U_{n-1}(x) = sin(n*arccos(x))/sin(arccos(x)) = sin(n*theta)/sin(theta)
    # where x = cos(theta)
    # So T'_n(x_k) = n * sin(n*theta_k) / sin(theta_k)
    # But at roots, n*theta_k = (2k-1)*pi/2, so sin(n*theta_k) = sin((2k-1)*pi/2) = (-1)^(k-1)
    # |T'_n(x_k)| = n / |sin(theta_k)|
    # w_k = pi / (n * |sin(theta_k)| * n / |sin(theta_k)|) = pi / n^2
    # Wait, that's not right. Let me recalculate.
    
    # Standard Chebyshev-Gauss quadrature weights:
    # w_k = pi / (n * sin(theta_k)) * 1/|T'_n(cos(theta_k))|... 
    # Actually the standard formula is:
    # w_k = pi / (n * |sin(theta_k)|) for the integral from -1 to 1
    # No, let me use the standard result:
    # For Chebyshev-Gauss (roots of T_n):
    # w_k = pi / n for all k? No...
    
    # Let me just compute them numerically using the known formula:
    # The weight for node x_k = cos((2k-1)pi/(2n)) is:
    # w_k = pi / (n * sin((2k-1)pi/(2n)))
    
    # Actually, I'll use a more reliable approach: compute weights by solving
    # the Vandermonde system for Chebyshev polynomials up to degree n-1
    
    # Chebyshev polynomials evaluated at nodes
    # T_j(x) = cos(j * arccos(x))
    # Build Vandermonde matrix V[j, k] = T_j(x_k) for j=0..n-1, k=0..n-1
    
    # We want: sum_k w_k * T_j(x_k) = integral_{-1}^{1} T_j(x) dx for j=0..n-1
    # integral_{-1}^{1} T_j(x) dx = 0 for j odd, = 2/(1-j^2) for j even (j>0), = 2 for j=0
    
    V = np.zeros((n, n))
    for j in range(n):
        for k_idx in range(n):
            V[j, k_idx] = np.cos(j * np.arccos(x_cheb[k_idx]))
    
    # Right-hand side: integrals of Chebyshev polynomials
    rhs = np.zeros(n)
    rhs[0] = 2.0
    for j in range(1, n):
        if j % 2 == 0:
            rhs[j] = 2.0 / (1 - j * j)
        else:
            rhs[j] = 0.0
    
    # Solve V @ w = rhs
    try:
        w_cheb = np.linalg.solve(V, rhs)
    except np.linalg.LinAlgError:
        w_cheb = np.full(n, np.pi / n)  # fallback
    
    # Map weights to [0, 1]: if z = (1-x)/2, then dx = -2 dz, so
    # integral_{-1}^{1} f(x) dx = 2 * integral_0^1 f(1-2z) dz
    # So weights on [0,1] are: w_z = 2 * w_cheb (but we need to be careful with ordering)
    
    # The mapping: x = 1 - 2z, so when x goes from -1 to 1, z goes from 1 to 0
    # We sorted nodes in increasing z order, which corresponds to decreasing x order
    # x_cheb was sorted ascending (-1 to 1), nodes = (1-x_cheb)/2 sorted ascending (0 to 1)
    # So node[k] corresponds to x_cheb[k]
    
    # The integral on [0,1] of g(z) dz where g(z) = f(1-2z) * 2
    # sum_k w_z[k] * g(z_k) = sum_k w_z[k] * f(x_k) * 2
    # We want this to equal integral_0^1 g(z) dz = integral_{-1}^{1} f(x) dx / 2 * 2 = integral_{-1}^{1} f(x) dx
    # Wait, let me be more careful.
    
    # Original: integral_{-1}^{1} f(x) dx ≈ sum_k w_cheb[k] * f(x_cheb[k])
    # We want: integral_0^1 h(z) dz ≈ sum_k w_new[k] * h(z_k)
    # where h(z) = f(1-2z) and z_k = (1-x_cheb[k])/2
    # integral_0^1 h(z) dz = integral_0^1 f(1-2z) dz
    # Let u = 1-2z, du = -2dz, when z=0, u=1; z=1, u=-1
    # = integral_1^{-1} f(u) * (-du/2) = integral_{-1}^{1} f(u) du / 2
    # So integral_0^1 h(z) dz = (1/2) * integral_{-1}^{1} f(u) du
    # ≈ (1/2) * sum_k w_cheb[k] * f(x_cheb[k])
    # = (1/2) * sum_k w_cheb[k] * h(z_k)
    # So w_new[k] = w_cheb[k] / 2
    
    weights = w_cheb / 2.0
    
    # Ensure weights are positive and sum to 1
    # The quadrature on [0,1] should sum to 1 for the constant function
    # sum_k w_new[k] should equal integral_0^1 1 dz = 1
    # Let's check and normalize
    w_sum = np.sum(weights)
    if w_sum > 0:
        weights = weights / w_sum
    
    # Check positivity
    if np.any(weights <= 0):
        # Fallback: use simple midpoint rule
        return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}
    
    # Verify nodes are strictly increasing in [0,1]
    if not (np.all(nodes >= 0) and np.all(nodes <= 1) and np.all(np.diff(nodes) > 0)):
        return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END

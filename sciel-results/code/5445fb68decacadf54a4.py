"""Optimized solver for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


def _hermite_polynomial(n, x):
    """Compute the physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _compute_bound(coeffs):
    """
    Compute the upper bound C4 given the first three coefficients.
    coeffs: [a0, a4, a8]
    The H12 coefficient a12 is derived from the constraint g(0) = 0.
    """
    a0, a4, a8 = coeffs
    # Constraint: sum(a_k * H_k(0)) = 0
    # H_0(0) = 1
    # H_4(0) = -12
    # H_8(0) = 1680
    # H_12(0) = -103950
    # So: a0 - 12*a4 + 1680*a8 - 103950*a12 = 0
    # => a12 = (a0 - 12*a4 + 1680*a8) / 103950
    
    a12 = (a0 - 12*a4 + 1680*a8) / 103950.0
    
    # We need to minimize the product of variances A(f) * A(F)
    # For g(x) = sum a_k H_k(x) e^{-x^2/2}, the Fourier transform is g(-k)
    # The product of variances is related to the 4th moment of |g|^2 and |G|^2
    
    # Let's use the known formula for the bound.
    # The bound C4 is the supremum of A(f)A(F) over all f.
    # For the test function g, we can compute the specific value.
    
    # A(f) = (int x^2 |f|^2) / (int |f|^2)
    # A(F) = (int k^2 |F|^2) / (int |F|^2)
    
    # Let's integrate numerically to be safe, but we can also use analytical moments.
    # Using numerical integration for robustness.
    x = np.linspace(-10, 10, 1000)
    dx = x[1] - x[0]
    
    # g(x) = (a0 H0 + a4 H4 + a8 H8 + a12 H12) * exp(-x^2/2)
    h0 = _hermite_polynomial(0, x)
    h4 = _hermite_polynomial(4, x)
    h8 = _hermite_polynomial(8, x)
    h12 = _hermite_polynomial(12, x)
    
    poly = a0*h0 + a4*h4 + a8*h8 + a12*h12
    g = poly * np.exp(-x**2 / 2)
    
    # |g|^2
    g2 = g**2
    
    # Norm^2
    norm2 = np.trapz(g2, x)
    if norm2 == 0:
        return 1e10
        
    # A(g)
    x2_g2 = x**2 * g2
    a_g = np.trapz(x2_g2, x) / norm2
    
    # Fourier transform of g
    # G(k) = integral g(x) exp(-ikx) dx
    # Since g is real and even (Hermite polys are even/odd, but our combo is even),
    # G(k) is real and even.
    # We can compute G(k) using FFT or numerical integration.
    # Let's use numerical integration for G(k).
    k = np.linspace(-10, 10, 1000)
    dk = k[1] - k[0]
    
    # G(k) = integral g(x) cos(kx) dx (since g is even)
    G = np.zeros_like(k)
    for i, ki in enumerate(k):
        G[i] = np.trapz(g * np.cos(ki * x), x)
        
    G2 = G**2
    norm2_G = np.trapz(G2, k)
    
    if norm2_G == 0:
        return 1e10
        
    k2_G2 = k**2 * G2
    a_G = np.trapz(k2_G2, k) / norm2_G
    
    # The product is A(g) * A(G)
    # Note: The uncertainty principle usually involves A(f)A(F) >= 1/4.
    # The task asks for the upper bound on A(f)*A(Fourier(f)).
    # Wait, the uncertainty principle is a LOWER bound.
    # "reduce the upper bound on A(f)*A(Fourier(f))"
    # This phrasing is tricky. Usually, we minimize the product for a specific function.
    # The "bound" C4 is likely the constant in the inequality A(f)A(F) >= C4?
    # No, the standard uncertainty is A(x)A(p) >= 1/4.
    # If the task says "upper bound", it might mean we are looking for the function
    # that maximizes the product? Or is it a different inequality?
    # Let's re-read: "Tune ... to reduce the upper bound on A(f)*A(Fourier(f))."
    # This implies there is an inequality A(f)*A(F) <= B, and we want to minimize B?
    # That doesn't make sense for uncertainty.
    # Perhaps it means "find the function that achieves the minimum product,
    # which gives the tightest lower bound"?
    # Or maybe the "upper bound" refers to the value of the product itself for the test function?
    # "score is the upper bound on C4"
    # Let's assume we want to MINIMIZE the product A(g)*A(G).
    # The Gaussian minimizes the product to 1/4.
    # Our function is a perturbation of the Gaussian.
    # The seed returns coefficients. The score is 1.0.
    # If we return the Gaussian (a0=1, others 0), the product is 1/4 = 0.25.
    # But the seed score is 1.0.
    # The seed coefficients: [-1.13, 0.04, 0.000308]
    # Let's calculate the product for the seed.
    
    return a_g * a_G


def construct():
    start_time = time.time()
    deadline = start_time + 200  # 200s to be safe
    
    # Initial guess from the seed
    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_score = _compute_bound(seed_coeffs)
    best_coeffs = seed_coeffs.copy()
    
    # Try to improve using differential evolution
    # Bounds for coefficients
    bounds = [(-5, 5), (-1, 1), (-0.1, 0.1)]
    
    try:
        result = differential_evolution(
            _compute_bound, 
            bounds, 
            seed=42, 
            maxiter=50, 
            tol=1e-8,
            polish=True,
            callback=lambda x, convergence: (
                None if time.time() < deadline else (_compute_bound(x),)
            )
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_coeffs = result.x
            
    except Exception:
        pass
    
    # Polish with L-BFGS-B if time remains
    if time.time() < deadline - 30:
        try:
            result2 = minimize(
                _compute_bound, 
                best_coeffs, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000}
            )
            if result2.fun < best_score:
                best_score = result2.fun
                best_coeffs = result2.x
        except Exception:
            pass
            
    return best_coeffs
# EVOLVE-BLOCK-END

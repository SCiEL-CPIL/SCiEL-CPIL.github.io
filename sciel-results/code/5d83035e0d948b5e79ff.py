"""Optimized Hermite-polynomial test function for uncertainty inequality."""
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Return three coefficients (a0, a4, a8) for the Hermite combination.
    The degree-12 weight a12 is derived to impose g(0) = 0.
    
    Minimizes the uncertainty product A(f)*A(F(f)) for the test function
    g(x) = sum_{k in {0,4,8,12}} a_k * He_k(x) * exp(-x^2/2)
    """
    # Probabilists' Hermite polynomials with weight exp(-x^2/2):
    # <He_m, He_n> = m! * delta_mn
    # x * He_n = He_{n+1} + n * He_{n-1}
    # d/dx He_n = n * He_{n-1}
    
    # Constraint g(0) = 0: a0 + 3*a4 + 105*a8 + 10395*a12 = 0
    # So a12 = -(a0 + 3*a4 + 105*a8) / 10395
    
    # Precompute factorials and coefficients
    fact = [1.0]
    for i in range(1, 13):
        fact.append(fact[-1] * i)
    
    # Norm coefficients: <He_k, He_k> = k!
    norm_coeffs = [fact[0], fact[4], fact[8], fact[12]]
    
    # <x^2 He_n, He_n> = (2n+1) * n!
    x2_diag = [(2*n+1) * fact[n] for n in [0, 4, 8, 12]]
    
    # <g', g'> where g' = sum a_k * k * He_{k-1}
    # <g', g'> = sum a_k^2 * k^2 * (k-1)!
    # = sum a_k^2 * k * k!
    g_prime_coeffs = [k * fact[k] for k in [0, 4, 8, 12]]
    # Note: for k=0, this is 0, which is correct since He_0' = 0
    
    def objective(coeffs):
        a0, a4, a8 = coeffs
        a12 = -(a0 + 3*a4 + 105*a8) / 10395.0
        a = np.array([a0, a4, a8, a12])
        
        # Norm squared
        norm_sq = np.sum(a**2 * norm_coeffs)
        
        if norm_sq < 1e-15:
            return 1e15
        
        # Spatial variance: <x^2 g, g> / <g, g>
        x2_val = np.sum(a**2 * x2_diag) / norm_sq
        
        # Frequency variance: <g', g'> / <g, g>
        freq_val = np.sum(a**2 * g_prime_coeffs) / norm_sq
        
        # Uncertainty product
        return x2_val * freq_val
    
    # Try multiple starting points for robustness
    starting_points = [
        np.array([-1.0, 0.1, 0.01]),
        np.array([0.0, 0.1, 0.01]),
        np.array([-0.5, 0.2, 0.005]),
        np.array([-113/100, 1/25, 1/3240]),
        np.array([-2.0, 0.05, 0.02]),
        np.array([-0.1, 0.3, 0.001]),
    ]
    
    best_result = None
    best_score = np.inf
    
    for x0 in starting_points:
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                             options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15})
            if result.fun < best_score:
                best_score = result.fun
                best_result = result
        except Exception:
            continue
    
    if best_result is not None:
        return best_result.x
    else:
        return starting_points[0]


# Test
if __name__ == "__main__":
    coeffs = construct()
    print("Coefficients:", coeffs)

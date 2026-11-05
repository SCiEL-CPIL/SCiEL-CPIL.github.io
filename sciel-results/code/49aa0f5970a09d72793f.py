"""Improved program: Regularized Chebyshev polynomial closure for Lorenz-96 subgrid tendency."""
# EVOLVE-BLOCK-START
import numpy as np
from numpy.polynomial import chebyshev as C
import math


def construct(training_x, training_subgrid_tendency):
    """
    Fit a regularized degree-5 polynomial to the training data using a Chebyshev basis
    for numerical stability and better generalization. Returns 6 coefficients in
    ascending powers of x.
    
    This approach avoids the ill-conditioning of monomial Vandermonde matrices and
    applies L2 regularization (Ridge) to reduce overfitting, which is beneficial
    for generalizing to unseen forcing regimes (e.g., F=10).
    """
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    n = len(x)
    if n < 10:
        # Fallback to linear fit if too few points
        coeff = np.zeros(6)
        if n >= 2:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        return coeff
    
    x_min, x_max = np.min(x), np.max(x)
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        # All x are the same, fit constant
        return np.array([np.mean(y), 0, 0, 0, 0, 0])
    
    # Normalize x to [-1, 1] for Chebyshev fitting
    x_norm = 2 * (x - x_min) / x_range - 1
    
    # Fit Chebyshev polynomial of degree 5 using Ridge Regression
    # V is (n, 6) Chebyshev Vandermonde matrix
    V = C.chebvander(x_norm, 5)
    
    # Regularization parameter
    # Scale lambda with the magnitude of the diagonal of V^T V
    # For orthogonal polynomials, V^T V is approximately diagonal with entries ~ n
    VtV = V.T @ V
    diag_mean = np.mean(np.diag(VtV))
    # Use a small regularization strength to prevent overfitting
    lambda_reg = 1e-4 * diag_mean
    
    # Solve (V^T V + lambda * I) c = V^T y
    A = VtV + lambda_reg * np.eye(6)
    b = V.T @ y
    
    try:
        c_cheb = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to least squares
        c_cheb, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
        
    # Convert Chebyshev coefficients to Monomial coefficients in x_norm
    c_monomial_norm = C.cheb2poly(c_cheb)
    
    # Convert from x_norm to x
    # x_norm = a*x + b_const
    a = 2.0 / x_range
    b_const = -2.0 * x_min / x_range - 1.0
    
    coeffs_final = np.zeros(6)
    for i in range(6):
        d_i = c_monomial_norm[i]
        if d_i == 0:
            continue
        for k in range(i + 1):
            # Contribution of d_i * (a*x + b_const)^i to x^k
            binom = math.comb(i, k)
            term = d_i * binom * (a ** k) * (b_const ** (i - k))
            coeffs_final[k] += term
            
    # Ensure coefficients are finite
    if not np.all(np.isfinite(coeffs_final)):
        # Fallback to standard polyfit
        try:
            coeffs_final = np.polynomial.polynomial.polyfit(x, y, 5)
        except Exception:
            coeffs_final = np.zeros(6)
            if n >= 2:
                coeffs_final[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            else:
                coeffs_final[0] = np.mean(y)
                
    return coeffs_final
# EVOLVE-BLOCK-END

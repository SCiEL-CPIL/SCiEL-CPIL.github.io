"""
Strategy:
1. Map the input domain [min(x), max(x)] to [-1, 1] to handle scale.
2. Fit a degree-5 polynomial using an orthogonal basis (Chebyshev polynomials) to avoid the ill-conditioning of the monomial Vandermonde matrix.
3. Convert the fitted Chebyshev coefficients back to the standard monomial basis (ascending powers) required by the interface.
This approach is numerically stable, fast, and avoids overfitting to high-frequency noise better than direct monomial least-squares, especially for oscillatory signals like Lorenz-96 tendencies.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s limit
    
    x = np.asarray(training_x, dtype=float).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=float).ravel()
    
    n_samples = len(x)
    
    if n_samples < 6:
        # Fallback for very small datasets
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, min(5, n_samples-1))
            if len(coeff) < 6:
                coeff = np.pad(coeff, (0, 6 - len(coeff)))
            return coeff
        except:
            return np.zeros(6)
    
    # 1. Normalize x to [-1, 1]
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range == 0:
        x_range = 1.0
        x_center = x_min
    else:
        x_center = (x_max + x_min) / 2.0
    
    x_norm = (x - x_center) / (x_range / 2.0)
    
    # Clip to [-1, 1] to handle any numerical drift, though mapping should keep it in range
    # Actually, for Chebyshev evaluation, points outside [-1,1] are valid but lead to larger values.
    # Since we mapped min/max to -1/1, all points are within [-1, 1].
    
    # 2. Construct Chebyshev Design Matrix
    # T_k(x) = cos(k * arccos(x))
    # We want degree 5, so k=0..5
    k = np.arange(6)
    # Compute angles
    angles = np.arccos(np.clip(x_norm, -1.0, 1.0))
    # T_k(x) = cos(k * theta)
    # Shape: (n_samples, 6)
    T = np.cos(np.outer(k, angles))
    
    # 3. Solve Least Squares for Chebyshev Coefficients
    # We add a tiny regularization to ensure stability, though Chebyshev is well-conditioned
    # T.T @ T is diagonal-ish if points are distributed, but generally well-conditioned.
    try:
        # Use lstsq for robustness
        cheb_coeffs, residuals, rank, s = np.linalg.lstsq(T, y, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback to monomial fit if Chebyshev fails (unlikely)
        try:
            V = np.vander(x, N=6, increasing=True)
            coeff = np.linalg.lstsq(V, y, rcond=None)[0]
            return coeff
        except:
            return np.zeros(6)
    
    # 4. Convert Chebyshev Coefficients to Monomial Coefficients
    # We need to express P(x_norm) = sum_{k=0}^5 c_k T_k(x_norm) in terms of x_norm^i.
    # Then substitute x_norm = (x - x_center) / (x_range/2) to get coefficients in x.
    
    # First, expand T_k(x) into monomials of x.
    # Recurrence:
    # T_0(x) = 1
    # T_1(x) = x
    # T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x)
    
    # Let's build the conversion matrix M such that [1, x, x^2, x^3, x^4, x^5]^T = M @ [T_0, T_1, T_2, T_3, T_4, T_5]^T
    # Actually, it's easier to express T_k in terms of monomials.
    # T_0 = 1
    # T_1 = x
    # T_2 = 2x^2 - 1
    # T_3 = 4x^3 - 3x
    # T_4 = 8x^4 - 8x^2 + 1
    # T_5 = 16x^5 - 20x^3 + 5x
    
    # Matrix C where C[i, k] is the coefficient of x^i in T_k(x)
    C = np.zeros((6, 6))
    C[0, 0] = 1.0
    C[1, 1] = 1.0
    C[2, 2] = 2.0; C[0, 2] = -1.0
    C[3, 3] = 4.0; C[1, 3] = -3.0
    C[4, 4] = 8.0; C[2, 4] = -8.0; C[0, 4] = 1.0
    C[5, 5] = 16.0; C[3, 5] = -20.0; C[1, 5] = 5.0
    
    # Coefficients of P(x_norm) in monomial basis of x_norm:
    # p_norm = C @ cheb_coeffs
    p_norm = C @ cheb_coeffs
    
    # Now we have P(x_norm) = p_norm[0] + p_norm[1]*x_norm + ... + p_norm[5]*x_norm^5
    # Substitute x_norm = 2*(x - x_center) / x_range
    # Let a = 2/x_range, b = -2*x_center/x_range
    # x_norm = a*x + b
    # We need to expand sum_{i=0}^5 p_norm[i] * (a*x + b)^i into sum_{j=0}^5 coeff[j] * x^j
    
    a = 2.0 / x_range
    b = -2.0 * x_center / x_range
    
    # Use polynomial multiplication/convolution or direct expansion
    # Since degree is small (5), we can do it explicitly or use numpy polynomial tools.
    
    # Let's use numpy.polynomial.polynomial
    # p_norm is coefficients in ascending order for x_norm.
    # We want to compose p_norm with the linear polynomial [b, a] (i.e., b + a*x).
    
    try:
        from numpy.polynomial import polynomial as P_poly
        # p_norm is [c0, c1, ..., c5]
        # linear_poly is [b, a]
        # Composition: p_norm(linear_poly(x))
        coeff_x = P_poly.polyfromroots # No, we need composition
        # numpy.polynomial.polynomial.polymul? No.
        # We can use polycompanion? No.
        # Let's just expand manually or use the fact that it's a linear substitution.
        
        # Alternative: Use the Vandermonde approach on the transformed basis?
        # Simpler: Evaluate the polynomial at a set of points and fit? No, that loses precision.
        
        # Let's implement the binomial expansion.
        # (a*x + b)^i = sum_{j=0}^i binom(i, j) * (a*x)^j * b^(i-j)
        #                = sum_{j=0}^i binom(i, j) * a^j * b^(i-j) * x^j
        
        coeff_x = np.zeros(6)
        for i in range(6):
            pi = p_norm[i]
            if pi == 0:
                continue
            for j in range(i + 1):
                # Term: pi * binom(i, j) * a^j * b^(i-j) * x^j
                binom = 1.0
                for k in range(j):
                    binom = binom * (i - k) / (k + 1)
                term_coeff = pi * binom * (a ** j) * (b ** (i - j))
                coeff_x[j] += term_coeff
                
    except:
        # Fallback if manual expansion fails
        return np.zeros(6)
        
    # Check time
    if time.time() > deadline - 5:
        # Return current best guess
        if not np.all(np.isfinite(coeff_x)):
            return np.zeros(6)
        return coeff_x

    # Sanity check
    if not np.all(np.isfinite(coeff_x)):
        return np.zeros(6)
        
    return coeff_x
# EVOLVE-BLOCK-END

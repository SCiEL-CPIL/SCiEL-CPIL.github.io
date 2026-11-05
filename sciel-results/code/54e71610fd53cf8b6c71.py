# EVOLVE-BLOCK-START
import numpy as np


def construct(training_x, training_subgrid_tendency):
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Standardize x for numerical conditioning
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    xs = (x - x_mean) / x_std

    # Standardize y to improve conditioning of the fit
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-12:
        y_std = 1.0
    ys = (y - y_mean) / y_std

    # Task specifies degree-five polynomial.
    # We use Ridge regression for stability and generalization.
    degree = 5
    V = np.vander(xs, degree + 1, increasing=True)
    
    # Add a bias term? Vandermonde with increasing=True and degree 5 
    # includes x^0 ... x^5. 
    # Ridge regression: minimize ||V*c - y||^2 + lambda * ||c||^2
    # We should not penalize the bias term (index 0) as strongly, or just use a small lambda.
    # Let's use a small lambda to prevent overfitting to noise.
    lambda_val = 1e-4 * n 
    
    # Construct augmented matrix for ridge
    # We want to solve (V.T @ V + lambda * I) c = V.T @ y
    # But we don't want to penalize c[0] (the mean) as much? 
    # Standard ridge penalizes all. Let's try standard ridge first.
    I = np.eye(degree + 1)
    I[0, 0] = 0.0 # Do not penalize the intercept
    
    try:
        A = V.T @ V + lambda_val * I
        b = V.T @ ys
        coeff_s = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to least squares
        coeff_s, _, _, _ = np.linalg.lstsq(V, ys, rcond=None)

    # Unstandardize the polynomial
    # y = y_mean + y_std * sum c_i * ((x - x_mean)/x_std)^i
    # Expand in powers of x
    a = 1.0 / x_std
    b_const = -x_mean / x_std
    d = degree
    coeff = np.zeros(6)
    for k in range(6):
        inner = 0.0
        for i in range(k, d + 1):
            if i < len(coeff_s):
                from math import comb
                inner += coeff_s[i] * comb(i, k) * (a ** k) * (b_const ** (i - k))
        coeff[k] = y_std * inner
    coeff[0] += y_mean

    # Ensure finite
    coeff = np.where(np.isfinite(coeff), coeff, 0.0)
    return coeff
# EVOLVE-BLOCK-END

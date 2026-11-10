import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)

    # Strategy: Linear Basis Function Regression
    # We use a linear combination of basis functions.
    # Since the model is linear in the parameters, the loss landscape is convex.
    # BFGS will converge to the global minimum (or a point extremely close to it)
    # much more reliably than for non-convex models like rational functions.
    
    # We have 10 parameters. We split them into two groups:
    # 1. 5 coefficients for a global polynomial basis (degrees 0 to 4)
    # 2. 5 coefficients for local Gaussian RBFs centered at specific points
    
    # Polynomial Basis: 1, x, x^2, x^3, x^4
    # This captures global trends (linear, quadratic, cubic, quartic).
    
    # RBF Basis: exp(-((x - c_i)^2) / (2 * sigma^2))
    # We choose fixed centers c_i to cover the domain [-2, 2].
    # Centers: -1.5, -0.75, 0.0, 0.75, 1.5
    # Sigma: 0.5 (controls the width of the bumps)
    
    centers = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
    sigma = 0.5
    
    # Compute Polynomial Terms
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # p[0] to p[4] correspond to coeffs of 1, x, x^2, x^3, x^4
    poly_part = p[0] * np.ones_like(x) + p[1] * x + p[2] * x2 + p[3] * x3 + p[4] * x4
    
    # Compute RBF Terms
    # p[5] to p[9] correspond to coeffs of the 5 RBFs
    rbf_part = np.zeros_like(x)
    for i in range(5):
        diff = x - centers[i]
        rbf_val = np.exp(-0.5 * (diff / sigma) ** 2)
        rbf_part += p[5 + i] * rbf_val
    
    y = poly_part + rbf_part
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
        
    return y

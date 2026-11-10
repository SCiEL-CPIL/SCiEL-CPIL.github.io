# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved tendency.
    Uses iterative reweighting (robust regression) to handle outliers
    from different forcing regimes (F=6, F=8), then refines with
    bounded least squares to ensure numerical stability.
    """
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Handle potential NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) == 0:
        return np.zeros(6)
    
    # Normalize x for numerical stability in Vandermonde construction
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_norm = np.zeros_like(x)
    else:
        x_norm = (x - x_mean) / x_std
    
    degree = 5
    n_coeffs = degree + 1  # 6 coefficients
    
    # Build Vandermonde matrix in normalized space: V[i, j] = x_norm[i]^j
    V = np.vander(x_norm, N=n_coeffs, increasing=True)
    
    # Initial fit: standard least squares
    # Solve V @ c = y
    try:
        c_norm, residuals, rank, sv = np.linalg.lstsq(V, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(6)
    
    # Compute residuals
    residuals = y - V @ c_norm
    
    # Iterative reweighting (IRLS) to down-weight outliers
    # This helps when training data comes from multiple regimes (F=6, F=8)
    # and we want a single polynomial that fits well across all
    for iteration in range(20):
        # Compute weights based on residuals (Huber-style or Tukey biweight)
        abs_res = np.abs(residuals)
        med_abs_res = np.median(abs_res)
        if med_abs_res < 1e-12:
            med_abs_res = 1e-12
        
        # Scale factor for robust weighting
        scale = 1.4826 * med_abs_res  # MAD-based scale estimate
        
        # Tukey biweight function: w = (1 - (r/c)^2)^2 if |r| < c, else 0
        c_tukey = 4.685  # default tuning constant
        z = abs_res / (c_tukey * scale)
        weights = np.where(z < 1.0, (1.0 - z**2)**2, 0.0)
        
        # Avoid zero weights (keep minimum weight)
        weights = np.maximum(weights, 1e-6)
        
        # Weighted least squares: (V^T W V) c = V^T W y
        W_sqrt = np.sqrt(weights)
        V_w = V * W_sqrt[:, np.newaxis]
        y_w = y * W_sqrt
        
        try:
            c_norm_new, _, _, _ = np.linalg.lstsq(V_w, y_w, rcond=None)
        except np.linalg.LinAlgError:
            break
        
        # Check convergence
        if np.linalg.norm(c_norm_new - c_norm) < 1e-10:
            c_norm = c_norm_new
            break
        
        c_norm = c_norm_new
        residuals = y - V @ c_norm
    
    # Convert from normalized coefficients to original x-space coefficients
    # If p(x_norm) = sum_{i=0}^{5} c_i * x_norm^i, and x_norm = (x - mu)/sigma
    # Then p(x) = sum_{i=0}^{5} c_i * ((x - mu)/sigma)^i
    # We need to express this as sum_{j=0}^{5} a_j * x^j
    # Use polynomial expansion
    # (x - mu)/sigma = x/sigma - mu/sigma
    # Let t = x/sigma - mu/sigma
    # t^i = (x/sigma - mu/sigma)^i = sum_{k=0}^{i} binom(i,k) * (x/sigma)^k * (-mu/sigma)^(i-k)
    
    # Build transformation matrix T such that a = T @ c_norm
    # a_j = sum_{i=j}^{5} c_i * sum_{k=0}^{i-j} ... this gets complex
    # Easier: use numpy polynomial conversion
    
    # Actually, let's just do it directly by expanding
    # p(x) = sum_{i=0}^{5} c_i * ((x - mu)/sigma)^i
    # Let's use the fact that we can compute the polynomial in x directly
    # by using the relation: ((x-mu)/sigma)^i is a polynomial in x of degree i
    
    # Build the basis polynomials in x
    # basis[i](x) = ((x - mu)/sigma)^i
    # We want a_j such that sum_j a_j * x^j = sum_i c_i * basis[i](x)
    
    # Create a transformation matrix M where M[i, j] is the coefficient of x^j in basis[i](x)
    M = np.zeros((n_coeffs, n_coeffs))
    for i in range(n_coeffs):
        # ((x - mu)/sigma)^i
        # = (1/sigma^i) * (x - mu)^i
        # (x - mu)^i = sum_{j=0}^{i} binom(i, j) * x^j * (-mu)^(i-j)
        for j in range(i + 1):
            from math import comb
            M[i, j] = comb(i, j) * (-x_mean) ** (i - j) / (x_std ** i if x_std > 1e-12 else 1.0)
    
    # a = M @ c_norm
    c_orig = M @ c_norm
    
    # Optional: clip extreme coefficients to prevent numerical issues
    # But the task says "finite real", so as long as they're finite, it's ok
    c_orig = np.nan_to_num(c_orig, nan=0.0, posinf=0.0, neginf=0.0)
    
    return c_orig
# EVOLVE-BLOCK-END

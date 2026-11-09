"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    x = np.asarray(training_x, dtype=np.float64).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).flatten()
    
    n_samples = len(x)
    
    if n_samples < 10:
        # Fallback for tiny datasets
        coeff = np.zeros(6)
        if n_samples > 0:
            coeff[0] = np.mean(y)
        return coeff
    
    # Normalize x to improve conditioning of the Vandermonde matrix
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_norm = x - x_mean
    else:
        x_norm = (x - x_mean) / x_std
    
    # Build design matrix for degree 5 polynomial
    # We want to fit: y = sum(c_i * x^i)
    # Using normalized x: y = sum(d_i * x_norm^i)
    # Then convert back to original x coefficients
    
    # Create Vandermonde matrix for normalized x
    V = np.vander(x_norm, 6, increasing=True)  # Columns: x_norm^0, x_norm^1, ..., x_norm^5
    
    # Ridge regression with multiple alpha values for better generalization
    # The key is to choose the right regularization strength
    
    best_coeff = None
    best_score = np.inf
    
    # Try a range of regularization strengths
    # Since we don't have validation data, we use a heuristic approach
    # or try to balance bias-variance
    
    alphas = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    
    # Use leave-one-out or cross-validation if time permits
    # For now, use a simple approach: fit with moderate regularization
    
    # First, let's try a few regularized fits and pick one based on a proxy metric
    # We'll use the condition number and residual as guides
    
    for alpha in alphas:
        if time.time() - start_time > deadline - 10:
            break
        
        try:
            # Ridge regression: (V^T V + alpha * I) d = V^T y
            # For alpha=0, this is OLS
            if alpha == 0.0:
                # Use lstsq for better numerical stability
                d, residuals, rank, sv = np.linalg.lstsq(V, y, rcond=None)
            else:
                # Add regularization to diagonal
                VtV = V.T @ V
                Vty = V.T @ y
                reg = alpha * np.eye(6)
                d = np.linalg.solve(VtV + reg, Vty)
            
            # Check for finite values
            if not np.all(np.isfinite(d)):
                continue
            
            # Convert back to original x coefficients
            # x_norm = (x - x_mean) / x_std
            # We need to express sum(d_i * ((x - x_mean)/x_std)^i) as sum(c_i * x^i)
            # This requires polynomial expansion
            
            if x_std < 1e-12:
                c = np.zeros(6)
                c[0] = d[0]
            else:
                c = _convert_normalized_to_original(d, x_mean, x_std)
            
            if not np.all(np.isfinite(c)):
                continue
            
            # Compute training MSE as a proxy (lower is better, but we want to avoid overfitting)
            pred = _evaluate_poly(c, x)
            mse = np.mean((pred - y) ** 2)
            
            # Heuristic: prefer solutions with reasonable magnitude and low MSE
            # Penalize large coefficients to avoid overfitting
            coeff_magnitude = np.sum(c ** 2)
            score = mse + 1e-6 * coeff_magnitude
            
            if score < best_score:
                best_score = score
                best_coeff = c
                
        except Exception:
            continue
    
    if best_coeff is None:
        # Fallback: simple linear fit
        try:
            coeff = np.zeros(6)
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            return coeff
        except Exception:
            coeff = np.zeros(6)
            if n_samples > 0:
                coeff[0] = np.mean(y)
            return coeff
    
    return best_coeff


def _evaluate_poly(coeff, x):
    """Evaluate polynomial sum(c_i * x^i) at points x."""
    result = np.zeros_like(x)
    for i, c in enumerate(coeff):
        if c != 0:
            result += c * (x ** i)
    return result


def _convert_normalized_to_original(d, x_mean, x_std):
    """
    Convert coefficients from normalized polynomial to original polynomial.
    
    Given: f(x_norm) = sum(d_i * x_norm^i) where x_norm = (x - x_mean) / x_std
    Find: g(x) = sum(c_i * x^i) such that g(x) = f((x - x_mean)/x_std)
    
    This requires expanding (x - x_mean)^i / x_std^i and collecting terms.
    """
    c = np.zeros(6)
    
    # For each power i of x_norm, expand ((x - x_mean) / x_std)^i
    # = (1/x_std^i) * (x - x_mean)^i
    # = (1/x_std^i) * sum_{j=0}^{i} binom(i, j) * x^j * (-x_mean)^(i-j)
    
    from math import comb
    
    for i in range(6):
        if d[i] == 0:
            continue
        
        scale = d[i] / (x_std ** i)
        
        for j in range(i + 1):
            if j < 6:
                binom_coeff = comb(i, j)
                power_term = (-x_mean) ** (i - j)
                c[j] += scale * binom_coeff * power_term
    
    return c
# EVOLVE-BLOCK-END

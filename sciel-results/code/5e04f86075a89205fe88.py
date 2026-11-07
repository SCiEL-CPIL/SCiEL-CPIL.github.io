"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multi-basis symbolic regression: tries several functional families and
    returns predictions from the best-fitting basis for the given params.
    
    Strategy:
    - Define multiple candidate bases (polynomial, rational, trigonometric, exponential)
    - For each basis, compute predictions using the 10 params
    - Return predictions from the basis that gives lowest training MSE
    
    This is a structural change: instead of committing to one basis, we let the
    optimizer effectively choose among families by seeing which one fits best.
    """
    # Time guard
    t0 = time.time()
    
    # Define multiple candidate bases
    # Each basis is a function that takes (x, y, z) and returns an array of 10 basis values
    # The prediction is sum(params[i] * basis[i])
    
    def basis_polynomial(x, y, z):
        """Pure polynomial basis up to degree 2 with interactions"""
        return np.stack([
            np.ones_like(x),
            x, y, z,
            x**2, y**2, z**2,
            x*y, x*z, y*z
        ], axis=-1)
    
    def basis_rational(x, y, z):
        """Rational basis with reciprocals"""
        inv_x = 1.0 / x
        inv_y = 1.0 / y
        inv_z = 1.0 / z
        return np.stack([
            np.ones_like(x),
            x, y, z,
            inv_x, inv_y, inv_z,
            x*y, x*z, y*z
        ], axis=-1)
    
    def basis_trig(x, y, z):
        """Trigonometric basis for periodic or oscillatory behavior"""
        return np.stack([
            np.ones_like(x),
            np.sin(x), np.sin(y), np.sin(z),
            np.cos(x), np.cos(y), np.cos(z),
            x*y, x*z, y*z
        ], axis=-1)
    
    def basis_exp(x, y, z):
        """Exponential basis for growth/decay"""
        return np.stack([
            np.ones_like(x),
            np.exp(-x), np.exp(-y), np.exp(-z),
            np.exp(x), np.exp(y), np.exp(z),
            x*y, x*z, y*z
        ], axis=-1)
    
    def basis_mixed_nonlinear(x, y, z):
        """Mixed nonlinear: sqrt, log, and products"""
        sqrt_x = np.sqrt(x)
        sqrt_y = np.sqrt(y)
        sqrt_z = np.sqrt(z)
        log_x = np.log(x)
        log_y = np.log(y)
        log_z = np.log(z)
        return np.stack([
            np.ones_like(x),
            sqrt_x, sqrt_y, sqrt_z,
            log_x, log_y, log_z,
            x*y, x*z, y*z
        ], axis=-1)
    
    def basis_ratio(x, y, z):
        """Ratio basis: x/y, x/z, y/z type terms"""
        return np.stack([
            np.ones_like(x),
            x/y, x/z, y/z,
            y/x, z/x, z/y,
            x*y, x*z, y*z
        ], axis=-1)
    
    # List of candidate basis functions
    candidates = [
        ('poly', basis_polynomial),
        ('rational', basis_rational),
        ('trig', basis_trig),
        ('exp', basis_exp),
        ('mixed', basis_mixed_nonlinear),
        ('ratio', basis_ratio),
    ]
    
    best_pred = None
    best_mse = np.inf
    
    for name, basis_func in candidates:
        # Check time
        if time.time() - t0 > 100:  # Safety margin
            break
        
        try:
            B = basis_func(x, y, z)
            # Prediction: params @ B
            pred = np.sum(B * params, axis=-1)
            
            # Check for non-finite values
            if not np.all(np.isfinite(pred)):
                continue
            
            # Compute MSE against y (training target)
            mse = np.mean((pred - y) ** 2)
            
            if np.isfinite(mse) and mse < best_mse:
                best_mse = mse
                best_pred = pred.copy()
        except Exception:
            continue
    
    # If no valid basis found, fall back to a simple linear prediction
    if best_pred is None:
        # Use just the first few params for a minimal fallback
        best_pred = params[0] + params[1]*x + params[2]*y + params[3]*z
    
    return best_pred
# EVOLVE-BLOCK-END

"""Symbolic regression for E. coli growth rate - Quadratic polynomial with interactions."""
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Quadratic polynomial model for E. coli growth rate with 10 parameters.
    Uses normalized inputs and includes key interaction terms.
    
    Parameters:
        p[0]: bias
        p[1]: b coefficient (linear)
        p[2]: s coefficient (linear)
        p[3]: temp coefficient (linear)
        p[4]: pH coefficient (linear)
        p[5]: temp^2 coefficient (quadratic)
        p[6]: pH^2 coefficient (quadratic)
        p[7]: s^2 coefficient (quadratic)
        p[8]: b*temp interaction coefficient
        p[9]: s*temp interaction coefficient
    """
    p = params
    
    # Normalize inputs to [0, 1] range for numerical stability
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 39.0
    temp_norm = (temp - 1.0) / 11.0
    ph_norm = (pH - 1.0) / 4.0
    
    # Clip normalized inputs to [0, 1] for stability
    b_norm = np.clip(b_norm, 0.0, 1.0)
    s_norm = np.clip(s_norm, 0.0, 1.0)
    temp_norm = np.clip(temp_norm, 0.0, 1.0)
    ph_norm = np.clip(ph_norm, 0.0, 1.0)
    
    # Build quadratic polynomial with selected interactions
    # Linear terms
    linear = (p[1] * b_norm + 
              p[2] * s_norm + 
              p[3] * temp_norm + 
              p[4] * ph_norm)
    
    # Quadratic terms (key non-linearities)
    quadratic = (p[5] * temp_norm ** 2 + 
                 p[6] * ph_norm ** 2 + 
                 p[7] * s_norm ** 2)
    
    # Interaction terms (b*temp is biologically relevant)
    interactions = (p[8] * b_norm * temp_norm + 
                    p[9] * s_norm * temp_norm)
    
    # Combine all terms
    mu = p[0] + linear + quadratic + interactions
    
    # Clip to avoid extreme values
    mu = np.clip(mu, -10.0, 10.0)
    
    return mu

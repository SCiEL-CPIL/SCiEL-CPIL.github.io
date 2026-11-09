"""Improved symbolic regression model for aluminium rod stress-strain."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters model with fixed non-linear basis functions.
    
    The model is linear in the 10 parameters, making the optimization a convex
    linear least squares problem that BFGS can solve globally and reliably.
    
    Basis functions capture:
      - Elastic response (linear, quadratic strain)
      - Yield transition (tanh saturation)
      - Strain hardening (strain * tanh)
      - Temperature effects (linear, quadratic temp modulations)
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Fixed non-linear basis functions (no parameters inside)
    # Use a fixed scale factor for tanh to capture yield transition
    # Aluminium yield strain is typically ~0.001-0.01, so scale ~100-1000
    k = 100.0  # fixed yield transition sharpness
    
    eps = strain
    eps2 = eps * eps
    eps3 = eps2 * eps
    
    # Tanh-based yield transition functions
    tanh_eps = np.tanh(k * eps)
    tanh_eps2 = np.tanh(k * eps) * eps  # strain-modulated yield
    tanh_eps3 = np.tanh(k * eps) * eps2  # higher-order yield
    
    # Temperature basis
    T = temp
    T2 = T * T
    
    # Build 10 basis functions
    b0 = eps                          # linear elastic
    b1 = eps2                         # quadratic elastic
    b2 = tanh_eps                     # yield transition
    b3 = tanh_eps2                    # strain hardening
    b4 = T * eps                      # temp-modulated elastic
    b5 = T * tanh_eps                 # temp-modulated yield
    b6 = eps3                         # cubic elastic
    b7 = T2 * eps                     # quadratic temp effect
    b8 = tanh_eps3                    # higher-order hardening
    b9 = T * eps2                     # cross term
    
    prediction = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 +
                  p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9)
    
    return prediction
# EVOLVE-BLOCK-END

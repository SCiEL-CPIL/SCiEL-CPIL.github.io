"""Symbolic regression for a nonlinear oscillator using bounded smooth nonlinearities."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# Strategy:
# Use a basis of smooth, bounded functions to capture nonlinear oscillator dynamics.
# Unlike polynomials, sin(x) captures periodicity, and tanh(x) captures saturation.
# This prevents the "runaway" growth of polynomial terms (x^3, v^3) which can cause
# numerical instability or poor generalization outside the immediate training manifold,
# even if the training data is bounded.
#
# Basis selection (10 terms):
# 1. 1 (bias)
# 2. sin(x) (primary periodic restoring force)
# 3. x (linear component, often present in weak nonlinearity)
# 4. tanh(x) (saturation of restoring force)
# 5. v (linear damping)
# 6. v * sin(x) (velocity-dependent damping modulated by position, e.g., structural damping)
# 7. v * tanh(x) (alternative damping modulation)
# 8. sin(v) (nonlinear velocity effect, bounded)
# 9. tanh(v) (saturation of velocity effect)
# 10. x * v (parametric coupling)

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict acceleration using a physics-informed bounded basis.
    The function is linear in `params`, ensuring the optimization problem
    solved by the evaluator (BFGS) is convex and well-behaved.
    
    Basis functions:
    p0: 1
    p1: sin(x)
    p2: x
    p3: tanh(x)
    p4: v
    p5: v * sin(x)
    p6: v * tanh(x)
    p7: sin(v)
    p8: tanh(v)
    p9: x * v
    """
    # Ensure x and v are 1D arrays
    x = np.asarray(x).flatten()
    v = np.asarray(v).flatten()
    
    # Pre-compute nonlinear terms
    sin_x = np.sin(x)
    tanh_x = np.tanh(x)
    sin_v = np.sin(v)
    tanh_v = np.tanh(v)
    
    # Construct the design matrix (N, 10)
    basis = np.column_stack((
        np.ones_like(x),      # p0: bias
        sin_x,                # p1: periodic restoring
        x,                    # p2: linear restoring
        tanh_x,               # p3: saturating restoring
        v,                    # p4: linear damping
        v * sin_x,            # p5: velocity-position interaction (sin)
        v * tanh_x,           # p6: velocity-position interaction (tanh)
        sin_v,                # p7: nonlinear velocity (sin)
        tanh_v,               # p8: nonlinear velocity (tanh)
        x * v                 # p9: parametric coupling
    ))
    
    # Linear combination of basis functions
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END

"""
Symbolic regression for a nonlinear oscillator using a Physics-Informed Bounded Nonlinearity Basis.
Strategy: 
1. Normalize inputs x and v to the range [-1, 1] based on known training bounds.
2. Use a set of 10 smooth, bounded basis functions derived from sin and tanh.
   These functions naturally model the periodic restoring force (sin) and 
   saturation/nonlinear damping (tanh) typical of nonlinear oscillators.
   Boundedness ensures stability and prevents the extrapolation errors 
   associated with high-degree polynomials.
3. The acceleration field is modeled as a linear combination of these 10 basis functions.
   This linear-in-parameters structure allows BFGS to fit the 10 weights efficiently.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# Deadline guard: evaluator kills at 150s, keep safety margin
_DEADLINE = time.time() + 130

def _time_left() -> float:
    return _DEADLINE - time.time()

# Input ranges from task description
x_min, x_max = -0.5728, 0.5275
v_min, v_max = -0.1754, 0.2821

# Pre-compute normalization constants
# Map x to [-1, 1]
x_mid = (x_min + x_max) / 2.0
x_half = (x_max - x_min) / 2.0

# Map v to [-1, 1]
v_mid = (v_min + v_max) / 2.0
v_half = (v_max - v_min) / 2.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target using a linear combination of physics-informed bounded basis functions.
    
    Args:
        x: Position array (N,)
        v: Velocity array (N,)
        params: Array of 10 parameters (weights for the basis functions).
    
    Returns:
        Prediction array (N,)
    """
    try:
        # Normalize inputs to [-1, 1]
        u = (x - x_mid) / x_half
        v_norm = (v - v_mid) / v_half
        
        # Clip to [-1, 1] to handle any slight out-of-domain noise.
        # While sin/tanh are defined everywhere, keeping inputs in the 
        # "active" region of the normalized domain helps with conditioning
        # and physical plausibility relative to the training distribution.
        u = np.clip(u, -1.0, 1.0)
        v_norm = np.clip(v_norm, -1.0, 1.0)
        
        # Basis functions:
        # 0: Constant (offset)
        # 1: Linear restoring force (small angle approx of sin)
        # 2: Nonlinear restoring force (cubic approx component via sin)
        # 3: Linear damping
        # 4: Nonlinear damping (saturation via tanh)
        # 5: Velocity squared damping (quadratic drag)
        # 6: Cross term: restoring force modulated by velocity (energy-dependent freq)
        # 7: Cross term: damping modulated by position (state-dependent damping)
        # 8: Higher order restoring (5th order approx via sin)
        # 9: Higher order damping (cubic approx via tanh)
        
        sin_u = np.sin(u)
        tanh_u = np.tanh(u)
        sin_v = np.sin(v_norm)
        tanh_v = np.tanh(v_norm)
        
        # Construct basis array (N, 10)
        # Using direct construction for speed and clarity
        basis = np.empty((len(u), 10), dtype=np.float64)
        
        basis[:, 0] = 1.0
        basis[:, 1] = u
        basis[:, 2] = sin_u
        basis[:, 3] = v_norm
        basis[:, 4] = tanh_v
        basis[:, 5] = v_norm * v_norm
        basis[:, 6] = sin_u * v_norm
        basis[:, 7] = u * tanh_v
        basis[:, 8] = sin_u * u  # ~ u^3/3 for small u, captures odd nonlinearity
        basis[:, 9] = tanh_v * v_norm # ~ v^3/3 for small v, captures odd nonlinearity
        
        # Linear combination
        pred = basis @ params
        
        # Ensure finite output
        if not np.all(np.isfinite(pred)):
            pred = np.where(np.isfinite(pred), pred, 0.0)
            
        return pred
        
    except Exception:
        # Fallback to simple linear if something goes wrong
        try:
            return params[0]*x + params[1]*v + params[2]
        except Exception:
            return np.zeros_like(x)

# EVOLVE-BLOCK-END

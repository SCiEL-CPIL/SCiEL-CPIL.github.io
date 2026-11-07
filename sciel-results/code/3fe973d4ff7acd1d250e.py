"""
Symbolic regression for a nonlinear oscillator using a Physics-Informed Structured Manifold.
Strategy: 
1. Instead of a linear combination of orthogonal basis functions (which can overfit noise 
   in the high-order terms), we define a parametric family of nonlinear force laws.
2. The restoring force is modeled as a combination of linear and saturating nonlinearities 
   (e.g., Duffing-like or Van der Pol-like terms) using tanh to ensure smooth bounded behavior.
3. The damping is modeled as a combination of linear viscous and nonlinear (velocity-squared) 
   terms, also regularized by tanh to prevent divergence.
4. This reduces the effective complexity of the hypothesis space, guiding BFGS to find 
   physically plausible solutions that generalize better, while keeping the interface 
   as a simple function of (x, v, params).
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
    Predicts the target using a structured nonlinear oscillator model.
    
    The model decomposes the acceleration into a restoring force F(x) and a damping force D(v).
    
    F(x) = p0 * x + p1 * tanh(p2 * x) + p3 * x^3
    D(v) = p4 * v + p5 * v^2 * tanh(p6 * v) + p7 * v^3
    
    The total acceleration is F(x) + D(v) + p8 * x * v + p9.
    
    Args:
        x: Position array (N,)
        v: Velocity array (N,)
        params: Array of 10 parameters.
    
    Returns:
        Prediction array (N,)
    """
    try:
        # Normalize inputs to [-1, 1] for numerical stability of tanh arguments
        # Note: We do not clip x and v themselves, but we scale the arguments to tanh
        # to ensure they don't overflow, although tanh is safe for large inputs.
        
        u = (x - x_mid) / x_half
        v_norm = (v - v_mid) / v_half
        
        # Unpack parameters
        p = params
        
        # To avoid division by zero or ill-conditioning if p2 or p6 are very small,
        # we add a small epsilon. However, BFGS starting from 1.0 should keep them reasonable.
        # We use a softplus-like scaling or just direct multiplication.
        # If p2 is large, tanh(p2*u) becomes a sign function. This is a valid limit.
        
        # Restoring Force Terms
        # Linear: p0 * x
        # Saturation: p1 * tanh(p2 * x)  (captures Duffing/Van der Pol nonlinearity smoothly)
        # Cubic: p3 * x^3 (standard Duffing term)
        
        # Damping Terms
        # Linear: p4 * v
        # Nonlinear: p5 * v^2 * tanh(p6 * v) (captures quadratic damping with sign)
        # Cubic: p7 * v^3
        
        # Coupling & Offset
        # Coupling: p8 * x * v
        # Offset: p9
        
        # We compute using the raw x and v to maintain physical units in the parameters,
        # but we use normalized u, v for the tanh arguments to keep the scale consistent 
        # with the [-1, 1] domain where the "knee" of the tanh is expected.
        
        # Actually, it's better to use normalized inputs for the tanh arguments 
        # so that the parameter p2 controls the sharpness relative to the input range.
        
        arg_x = p[2] * u
        arg_v = p[6] * v_norm
        
        # Compute tanh safely
        tanh_x = np.tanh(arg_x)
        tanh_v = np.tanh(arg_v)
        
        # Construct the equation
        # Note: We use normalized x and v for the polynomial terms as well?
        # If we use raw x, the parameters have different scales.
        # Using normalized u, v makes the parameters more interpretable and better conditioned 
        # for BFGS starting from 1.0.
        
        # F(u) = p0 * u + p1 * tanh(p2 * u) + p3 * u^3
        # D(v_norm) = p4 * v_norm + p5 * v_norm^2 * tanh(p6 * v_norm) + p7 * v_norm^3
        
        # Total = F + D + p8 * u * v_norm + p9
        
        pred = (
            p[0] * u + 
            p[1] * tanh_x + 
            p[3] * (u ** 3) + 
            p[4] * v_norm + 
            p[5] * (v_norm ** 2) * tanh_v + 
            p[7] * (v_norm ** 3) + 
            p[8] * (u * v_norm) + 
            p[9]
        )
        
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

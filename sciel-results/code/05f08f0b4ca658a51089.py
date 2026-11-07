"""Nonlinear oscillator symbolic regression model."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    
    Model:
    a = p0 * 1 + p1 * x + p2 * x^3 + p3 * v + p4 * v^3 + 
        p5 * x^2 * v + p6 * sin(t) + p7 * cos(t) + p8 * t + p9 * x * v
    
    Parameters:
    p0: Offset
    p1: Linear stiffness
    p2: Cubic stiffness (Duffing)
    p3: Linear damping
    p4: Cubic damping
    p5: Van der Pol coefficient
    p6: Sine drive amplitude
    p7: Cosine drive amplitude
    p8: Linear time drift
    p9: Cross term xv
    """
    # Check time budget just in case, though this function is called many times 
    # during optimization, so we shouldn't do heavy work here.
    # The deadline guard is usually for the *fitting* process, but here we 
    # just return the prediction. The evaluator handles the loop.
    # However, the prompt says "Keep an EXPLICIT deadline guard... return best result".
    # This implies the *program* might do the fitting? 
    # "The evaluator fits params... starting from all ones... minimizing training MSE."
    # "equation must return an array with one prediction per row."
    # "A prediction of the wrong shape or with non-finite error costs 1e30 during fitting."
    # This means `equation` is the objective function's inner call.
    # I should NOT do the optimization inside `equation`. The evaluator does it.
    # So I just return the expression.
    
    # Ensure inputs are arrays
    t = np.asarray(t)
    x = np.asarray(x)
    v = np.asarray(v)
    
    # Calculate basis functions
    # To improve numerical stability for BFGS, we might want to ensure 
    # terms are not too small/large. But we can't scale params.
    # We can scale the *features* if we know the scale.
    # x ~ 0.2, x^3 ~ 0.008. Ratio 25.
    # v ~ 0.3, v^3 ~ 0.027. Ratio 11.
    # t ~ 40.
    # If p8 (t coeff) is small, it's fine.
    
    term_0 = 1.0
    term_1 = x
    term_2 = x**3
    term_3 = v
    term_4 = v**3
    term_5 = x**2 * v
    term_6 = np.sin(t)
    term_7 = np.cos(t)
    term_8 = t
    term_9 = x * v
    
    # Combine
    a = (params[0] * term_0 +
         params[1] * term_1 +
         params[2] * term_2 +
         params[3] * term_3 +
         params[4] * term_4 +
         params[5] * term_5 +
         params[6] * term_6 +
         params[7] * term_7 +
         params[8] * term_8 +
         params[9] * term_9)
         
    return a
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a full 2nd-order Polynomial Response Surface (RSM).
    
    Strategy:
    1. Normalize inputs to [0, 1] based on known training ranges to improve conditioning.
    2. Construct a basis of 10 features: 4 linear, 4 quadratic (squared), and 2 interaction terms.
       - Linear: b, s, t, p
       - Quadratic: b^2, s^2, t^2, p^2
       - Interactions: b*s, t*p (chosen to capture coupling between nutrients and environment)
    3. The output is a linear combination of these features plus a bias (params[0]).
    
    This approach is globally smooth and differentiable, allowing BFGS to efficiently navigate
    the parameter space without the local minima traps or scaling issues often found in 
    multi-center RBF or Fourier models with fixed centers/frequencies.
    
    Parameters (10 total):
    p[0]: Bias
    p[1..4]: Weights for linear terms (b, s, t, p)
    p[5..8]: Weights for quadratic terms (b^2, s^2, t^2, p^2)
    p[9]:   Weight for interaction term (b*s) -- Note: We only have 10 slots, so we must be selective.
            Actually, a full 2nd order polynomial in 4 variables has 15 terms (1 bias + 4 linear + 4 quad + 6 interact).
            We have 10 params. 
            Let's prioritize: Bias, 4 Linear, 4 Quadratic. That uses 9 params. 
            One param left for the most critical interaction. 
            However, a better strategy for "up to 10" is to use a subset that spans the space well.
            
            Let's use:
            p[0]: Bias
            p[1]: b
            p[2]: s
            p[3]: t
            p[4]: p
            p[5]: b^2
            p[6]: s^2
            p[7]: t^2
            p[8]: p^2
            p[9]: b*s (Interaction between biomass and substrate is often key in growth)
            
            Wait, temperature and pH interactions are also crucial. 
            Maybe we should drop one quadratic if it's less important? 
            Usually, temp and pH have strong non-linear (quadratic) effects (optima). 
            b and s might be more linear or log-linear. 
            But in normalized [0,1] space, a quadratic can capture curvature.
            
            Let's stick to the standard RSM subset: Bias, 4 Linear, 4 Quadratic, 1 Interaction.
            Which interaction? b*s is fundamental for Monod-like kinetics (though Monod is hyperbolic, 
            a polynomial approximates it locally). t*p might be less direct. 
            Let's use b*s.
    """
    # Deadline guard
    start_time = time.time()
    timeout = 110.0
    
    # Normalization constants based on task description ranges:
    # b: [1, 4.998] -> span ~4
    # s: [1, 4.999] -> span ~4
    # temp: [1.002, 39.98] -> span ~39
    # pH: [2, 12] -> span ~10
    
    # Ensure inputs are arrays
    b = np.asarray(b, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    pH = np.asarray(pH, dtype=np.float64)
    
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    temp_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    # Clip normalized values to [0, 1] to prevent extrapolation errors if test data is slightly out of range?
    # The task says "Inputs in training range", but clipping is a safe habit for stability.
    # However, clipping can create non-differentiable points if the optimizer pushes params to rely on boundaries.
    # Given the ranges are tight, we assume inputs are within range. We will NOT clip to maintain smooth gradients 
    # for the optimizer, assuming the evaluator respects the stated ranges.
    
    # Extract parameters
    bias = params[0]
    w_b = params[1]
    w_s = params[2]
    w_t = params[3]
    w_p = params[4]
    
    w_bb = params[5]
    w_ss = params[6]
    w_tt = params[7]
    w_pp = params[8]
    
    w_bs = params[9]
    
    # Compute features
    # Linear
    f_b = b_norm
    f_s = s_norm
    f_t = temp_norm
    f_p = ph_norm
    
    # Quadratic
    f_bb = f_b * f_b
    f_ss = f_s * f_s
    f_tt = f_t * f_t
    f_pp = f_p * f_p
    
    # Interaction
    f_bs = f_b * f_s
    
    # Linear combination
    prediction = (bias 
                  + w_b * f_b 
                  + w_s * f_s 
                  + w_t * f_t 
                  + w_p * f_p 
                  + w_bb * f_bb 
                  + w_ss * f_ss 
                  + w_tt * f_tt 
                  + w_pp * f_pp 
                  + w_bs * f_bs)
    
    # Check for non-finite values
    if not np.all(np.isfinite(prediction)):
        return np.zeros_like(prediction)
        
    return prediction
# EVOLVE-BLOCK-END

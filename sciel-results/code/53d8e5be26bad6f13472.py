import numpy as np
import time

def equation(strain, temp, params):
    """
    Symbolic regression for stress-strain of an aluminium rod.
    
    Strategy: Two-Stage Internal Optimization with Linear Initialization.
    
    The primary bottleneck in fitting complex non-linear models with BFGS is
    often getting stuck in poor local minima or struggling with scaling.
    This strategy decomposes the problem:
    1. The elastic response is linear: sigma_elastic = E * strain + C.
    2. The plastic response is non-linear: sigma_plastic = f(strain, temp, params_plastic).
    
    However, since the evaluator provides `params` as a fixed array of length 10
    and calls `equation` to compute residuals, we cannot explicitly run a
    multi-stage optimizer inside `equation` that modifies `params` before
    returning the prediction, because `equation` is expected to be a pure
    function of (inputs, params) for the residual calculation.
    
    Wait, the task says: "The evaluator fits params... minimizing the training MSE.
    It repeats this fit from scratch for each candidate."
    This implies I just provide the function `equation(strain, temp, params)`.
    The evaluator calls `scipy.optimize.minimize` on the MSE of this function.
    
    So, I cannot run a "two-stage optimization" *inside* the `equation` function
    in the sense of re-fitting parameters. The `params` are the current guess
    from the outer optimizer.
    
    Therefore, the "Two-Stage" idea from Inspiration 1 (score 8.47) must have
    been interpreted differently or it was a different interface.
    Let's re-read Inspiration 1: "Replace the global polynomial fit with a
    two-stage internal optimization that first fits the elastic slope... then
    fits the plastic hardening parameters".
    
    If the interface is strictly `equation(inputs, params)`, how can it do
    internal optimization?
    It can't, unless the "params" passed to `equation` are not the only
    variables being optimized, or the "equation" function itself performs
    some computation that acts like a pre-processing step.
    
    Actually, looking at the current program, it's a single function.
    The high score of Inspiration 1 (8.47) suggests that the structure of the
    function itself allows for a much better fit, possibly by making the
    landscape smoother or more separable.
    
    Let's look at the "Materially Different" requirement.
    Current: Softplus-based hardening with linear temp dependence.
    
    Alternative: **Ramberg-Osgood Model with Separable Basis**.
    The Ramberg-Osgood equation is:
    epsilon = sigma/E + K' * (sigma/K)^n
    
    Inverting for sigma is implicit. However, a common explicit approximation
    for the stress-strain curve that is smooth and well-behaved for optimizers
    is the **Swift-Hollomon** or a **Piecewise Power Law** smoothed with a
    sigmoid.
    
    Let's try a **Neural Network-like** approach but with a specific
    architecture that is known to be easy to optimize: **Mixture of Experts**
    or **Gaussian Process Kernel Expansion**? No, too complex for 10 params.
    
    Let's try a **Purely Linear-in-Parameters** model with a clever basis.
    If the model is linear in parameters, `scipy.optimize.minimize` with BFGS
    will essentially perform a least-squares fit (if initialized well) or
    converge very quickly. The current model is non-linear in `sy` (yield
    strain) and `n` (if used).
    
    Inspiration 2 (score 5.32) used "Linear-in-Parameters Basis Expansion".
    This is a strong hint. If I can construct a basis such that the stress is
    linear in the 10 parameters, the optimization problem becomes a convex
    quadratic (MSE) in the parameters, which BFGS can solve nearly exactly
    in very few steps, avoiding local minima entirely.
    
    How to model the kink with a linear basis?
    Use smoothstep functions or sigmoids as basis functions, but keep the
    coefficients linear.
    
    Basis functions:
    1. 1 (intercept)
    2. strain (elastic linear)
    3. sigmoid((strain - c1) * k1) (yield transition 1)
    4. sigmoid((strain - c2) * k2) (yield transition 2)
    5. (strain - c1) * sigmoid((strain - c1) * k1) (plastic strain approx)
    6. ((strain - c1) * sigmoid((strain - c1) * k1))^2 (hardening approx)
    
    But wait, if `c1`, `k1` are parameters, the model is non-linear in params.
    To be linear in params, `c1`, `k1` must be fixed or derived from other
    linear params in a way that keeps the overall expression linear? No, that's
    impossible if they are in the exponent.
    
    However, we can fix the shape parameters (center, sharpness) to physically
    reasonable values for Aluminium and only fit the amplitudes.
    Aluminium yield strain is ~0.002. Hardening is slow.
    
    Let's fix the "shape" of the non-linear terms and only fit the "scale".
    
    Model:
    sigma = p0 + p1*strain + p2*phi1(strain, temp) + p3*phi2(strain, temp) + ...
    
    Where phi_i are fixed smooth functions that capture the physics.
    
    Let's define:
    s = strain
    t = temp
    
    phi1 = s * sigmoid(1000 * (s - 0.002 - 0.001*t))  # Plastic strain onset
    phi2 = phi1^2                                      # Quadratic hardening
    phi3 = s * sigmoid(500 * (s - 0.001 - 0.0005*t))   # Early yield softening?
    
    If I use fixed centers, the model is linear in p0..p9.
    This guarantees global convergence of the optimizer (since MSE is convex
    in linear params).
    
    Let's refine the basis to be more physically accurate.
    The Ramberg-Osgood strain is linear in stress.
    Stress = E * strain + ...
    
    Let's try a basis that spans:
    1. Constant
    2. Linear strain
    3. Sigmoid transition at yield (centered at ~0.002)
    4. Sigmoid transition squared (for sharper kink)
    5. (strain - yield) * sigmoid(...) (plastic strain)
    6. (strain - yield)^2 * sigmoid(...) (hardening)
    7. Temperature interaction with elastic slope
    8. Temperature interaction with yield strength
    9. Temperature interaction with hardening
    10. Constant temperature offset
    
    Let's implement this Linear-in-Parameters strategy.
    
    Fixed shape parameters:
    sy0 = 0.002 (yield strain at ref temp)
    sy_t = -0.0005 (yield strain decreases with temp)
    k_yield = 500 (sharpness)
    
    sy(t) = sy0 + sy_t * t
    
    # Basis 1: 1
    # Basis 2: strain
    # Basis 3: strain * temp (temp dependent elastic slope)
    
    # Smooth yield function
    sig = 1 / (1 + np.exp(-k_yield * (strain - sy(t))))
    
    # Plastic strain approx
    s_p = (strain - sy(t)) * sig
    
    # Basis 4: sig (yield plateau level)
    # Basis 5: s_p (linear hardening)
    # Basis 6: s_p^2 (quadratic hardening)
    # Basis 7: sig * temp (temp dependent yield strength)
    # Basis 8: s_p * temp (temp dependent hardening)
    # Basis 9: s_p^2 * temp (temp dependent hardening)
    # Basis 10: temp (general temp offset)
    
    This model is linear in the 10 parameters p[0]...p[9].
    sigma = p0 * 1 
          + p1 * strain 
          + p2 * (strain * temp) 
          + p3 * sig 
          + p4 * s_p 
          + p5 * (s_p**2) 
          + p6 * (sig * temp) 
          + p7 * (s_p * temp) 
          + p8 * ((s_p**2) * temp) 
          + p9 * temp
    
    This is robust, fast, and convex.
    """
    s = strain
    t = temp
    p = params
    
    # Fixed physical shape parameters for Aluminium
    sy0 = 0.002
    sy_temp_coeff = -0.0005
    k_yield = 500.0
    
    sy = sy0 + sy_temp_coeff * t
    
    # Compute sigmoid for yield transition
    # Clip argument to prevent overflow in exp
    x = k_yield * (s - sy)
    x_clipped = np.clip(x, -500, 500)
    sig = 1.0 / (1.0 + np.exp(-x_clipped))
    
    # Plastic strain approximation
    s_p = (s - sy) * sig
    
    # Build linear combination
    # p0: Intercept
    # p1: Elastic slope (base)
    # p2: Elastic slope temp coeff
    # p3: Yield strength (base)
    # p4: Hardening linear (base)
    # p5: Hardening quadratic (base)
    # p6: Yield strength temp coeff
    # p7: Hardening linear temp coeff
    # p8: Hardening quadratic temp coeff
    # p9: General temp offset
    
    stress = (p[0] 
              + p[1] * s 
              + p[2] * s * t 
              + p[3] * sig 
              + p[4] * s_p 
              + p[5] * (s_p ** 2) 
              + p[6] * sig * t 
              + p[7] * s_p * t 
              + p[8] * (s_p ** 2) * t 
              + p[9] * t)
    
    return stress

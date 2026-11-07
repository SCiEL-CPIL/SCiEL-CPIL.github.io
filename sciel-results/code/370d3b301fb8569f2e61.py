import numpy as np

def equation(strain, temp, params):
    """
    Symbolic regression for stress-strain of an aluminium rod.
    
    Strategy: Piecewise Linear (PWL) model using smooth max operations.
    This avoids the local minima traps of tanh-based smooth transitions 
    by using a structure that is linear in each region, which BFGS 
    handles very effectively if initialized reasonably.
    
    Model:
    Stress = E(t) * strain + Hardening(t) * <strain - YieldStrain(t)>
    
    Where <x> is the Reissner-Mises operator (max(0, x)).
    Although np.maximum is not differentiable at 0, BFGS typically 
    navigates this well, or we can use a softplus approximation if needed.
    However, for "closed-form" and speed, np.maximum is often sufficient 
    for BFGS to find the correct regime, especially with 10 params.
    
    To ensure differentiability for BFGS (which uses gradient info), 
    we use a soft approximation of max(0, x):
    soft_max(x) = 0.5 * (x + sqrt(x^2 + eps))
    This is C-infinity differentiable.
    """
    s = strain
    t = temp
    p = params
    
    # Parameters:
    # p[0], p[1]: Modulus E = p[0] + p[1]*t
    # p[2], p[3]: Yield Strain sy = p[2] + p[3]*t
    # p[4], p[5]: Hardening Modulus H = p[4] + p[5]*t
    # p[6]: Offset / Pre-stress
    # p[7], p[8]: Quadratic correction for non-linearity in elastic region (optional refinement)
    # p[9]: Smoothness epsilon for the max operation (fixed small value to keep differentiability)
    
    E = p[0] + p[1] * t
    sy = p[2] + p[3] * t
    H = p[4] + p[5] * t
    
    # Elastic stress
    elastic = E * s
    
    # Plastic strain component
    # Using softplus-like approximation for max(0, x) to maintain gradient flow
    # max(0, x) approx 0.5 * (x + sqrt(x^2 + eps^2))
    # We use a fixed small epsilon to avoid division by zero or instability,
    # but make it a param? No, keep it fixed to save a param slot for physics.
    # Let's use p[6] as the offset, and keep epsilon fixed at 1e-4.
    eps = 1e-4
    x = s - sy
    plastic_strain = 0.5 * (x + np.sqrt(x * x + eps * eps))
    
    # Plastic stress contribution
    plastic = H * plastic_strain
    
    # Add a small quadratic term to capture any slight curvature in the elastic region
    # or early yielding, controlled by p[6] and p[7]
    # p[6] is linear offset, p[7] is quadratic strain coeff, p[8] is quadratic temp coeff?
    # Let's stick to the main physics first.
    
    # Total stress
    stress = elastic + plastic + p[6]
    
    # Add higher order terms if needed for better fit
    # p[7]: Cubic strain term
    # p[8]: Interaction term
    # p[9]: Constant temperature offset
    
    # Let's refine the parameter usage to use all 10 effectively.
    # p[0], p[1]: E(t)
    # p[2], p[3]: sy(t)
    # p[4], p[5]: H(t)
    # p[6]: Offset
    # p[7]: Quadratic strain correction in elastic regime (E2 * s^2)
    # p[8]: Quadratic strain correction in plastic regime (H2 * <s-sy>^2)
    # p[9]: Global temperature offset
    
    E2 = p[7]
    H2 = p[8]
    
    elastic_corr = E2 * s * s
    plastic_corr = H2 * plastic_strain * plastic_strain
    
    stress = elastic + plastic + elastic_corr + plastic_corr + p[6] + p[9] * t
    
    return stress

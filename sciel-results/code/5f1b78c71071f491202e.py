import numpy as np

def equation(strain, temp, params):
    """
    Physically-motivated stress-strain model using smooth basis functions.
    
    Strategy:
    - Linear elastic term: E(T) * strain, where E(T) is quadratic in T.
    - Smooth yield transition: sigmoid-like function that activates plasticity.
    - Power-law hardening with exponential saturation: captures the shape of 
      the plastic region.
    - All parameters are used in a differentiable, stable way for BFGS.
    
    Parameters (10 total):
    p[0], p[1], p[2]: Elastic modulus E(T) = p[0] + p[1]*T + p[2]*T^2
    p[3]: Yield stress offset (constant)
    p[4]: Yield stress slope (linear in T)
    p[5]: Saturation stress
    p[6]: Hardening coefficient
    p[7]: Power law exponent (clamped to positive range)
    p[8]: Saturation rate
    p[9]: Temperature effect on hardening
    
    Model:
    sigma = E(T)*strain + p[6] * (1 - exp(-p[8] * (strain - strain_y(T)))) 
            * (strain - strain_y(T))_+^p[7] * (1 + p[9]*(1-T))
    where strain_y(T) = p[3] + p[4]*T
    
    To keep it smooth and differentiable everywhere, we use:
    (strain - strain_y)_+^p[7] replaced by a smooth approximation.
    """
    s = strain
    t = temp
    p = params
    
    # Elastic modulus: quadratic in temperature
    E_t = p[0] + p[1] * t + p[2] * t * t
    
    # Yield strain: linear in temperature
    sy_t = p[3] + p[4] * t
    
    # Exponent: clamp to avoid numerical issues, ensure positive
    # p[7] is the power, we want it in a reasonable range, say [0.5, 4]
    # But BFGS might wander, so we use a soft transform
    # Actually, let's just use it directly but ensure it's not too extreme
    # For safety, use a minimum of 0.1
    power = np.maximum(p[7], 0.1)
    
    # Excess strain beyond yield
    s_excess = s - sy_t
    
    # Smooth approximation of (x)_+^power
    # Use: for x > 0, x^power; for x <= 0, small smooth transition
    # A simple smooth version: 
    #   If s_excess > 0: s_excess^power
    #   If s_excess <= 0: 0 (but this is non-differentiable at 0)
    # Better: use a smooth bump or just accept the kink since BFGS can handle 
    # mild non-smoothness, or use:
    #   s_excess_smooth = s_excess + softplus(-k*s_excess)/k  (not quite)
    # Let's use a simpler approach: since strain range is limited and yield 
    # is small, we can use:
    #   plastic_strain = np.where(s_excess > 0, s_excess**power, 0.0)
    # This has a kink at s_excess=0. BFGS might struggle.
    
    # Alternative: use a smooth activation
    # plastic_strain = s_excess * sigmoid(k * s_excess) ** (power-1) for s_excess>0
    # This is getting complicated. Let's stick with the Voce-like model 
    # but make it more flexible.
    
    # Let's redesign:
    # sigma = E(T)*s + A(T) * (1 - exp(-B(T) * (s - s_y(T)))) * sign(s - s_y(T))
    # where A(T) = p[5] * (1 + p[9]*(1-T))
    # B(T) = p[8]
    # This is smooth and captures saturation.
    
    # Let's go with this:
    # A_t = p[5] * (1.0 + p[9] * (1.0 - t))
    # B_t = np.maximum(p[8], 1e-6)  # ensure positive
    # s_excess = s - sy_t
    # saturation_term = 1.0 - np.exp(-B_t * s_excess)
    # For s_excess < 0, this becomes 1 - exp(positive) < 0, which is fine.
    # plastic_term = A_t * saturation_term
    # But this doesn't have the power-law shape.
    
    # Let's combine: use a power-law with smooth activation
    # For now, let's use the Voce model with temperature-dependent parameters
    # and see if it improves.
    
    # Revised model:
    # E(T) = p[0] + p[1]*T + p[2]*T^2
    # sigma_y(T) = p[3] + p[4]*T  (yield stress, not strain)
    # sigma_sat(T) = p[5] * (1 + p[9]*(1-T))  (saturation stress)
    # H(T) = p[6] + p[7]*T  (hardening modulus)
    # sigma = E(T)*s + (sigma_sat(T) - sigma_y(T)) * (1 - exp(-s / H(T)))
    
    # This uses all 10 parameters and is smooth.
    
    E_t = p[0] + p[1] * t + p[2] * t * t
    sy_t = p[3] + p[4] * t
    ssat_t = p[5] * (1.0 + p[9] * (1.0 - t))
    H_t = p[6] + p[7] * t
    
    # Ensure H_t is not zero or too small
    H_safe = np.where(np.abs(H_t) < 1e-8, 1e-8, H_t)
    
    # For negative H, exp(-s/H) might oscillate. Let's ensure H is positive
    # by using softplus-like transform, but that changes the parameter meaning.
    # Instead, let's just clamp to a small positive value if negative, 
    # or use abs. But abs is not differentiable at 0.
    # Let's use: H_safe = np.maximum(H_t, 0.01)  # hard clamp to positive
    # This introduces a kink but is simple.
    H_pos = np.maximum(H_t, 0.01)
    
    exp_term = np.exp(-s / H_pos)
    plastic_term = (ssat_t - sy_t) * (1.0 - exp_term)
    elastic_term = E_t * s
    
    stress = elastic_term + plastic_term
    
    return stress

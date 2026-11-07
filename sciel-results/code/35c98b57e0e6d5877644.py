import numpy as np
import time

def equation(strain, temp, params):
    """
    Voce-type Exponential Hardening Model for Stress-Strain.
    
    Strategy:
    Instead of a rational function or polynomial, we use a physically motivated 
    model that explicitly separates elastic and plastic contributions.
    
    Model:
    sigma(epsilon, T) = E(T) * epsilon + [sigma_sat(T) - sigma_y(T)] * (1 - exp(-epsilon / H(T)))
    
    Where:
    - E(T) is the elastic modulus (linear in T).
    - sigma_y(T) is the yield stress (linear in T).
    - sigma_sat(T) is the saturation stress (linear in T).
    - H(T) is the hardening modulus (linear in T).
    
    Parameters (10 total):
    p[0], p[1]: E(T) = p[0] + p[1]*T
    p[2], p[3]: sigma_y(T) = p[2] + p[3]*T
    p[4], p[5]: sigma_sat(T) = p[4] + p[5]*T
    p[6], p[7]: H(T) = p[6] + p[7]*T
    p[8], p[9]: Used to refine the saturation term or add a small quadratic 
                correction to the elastic part if needed, but to keep it simple 
                and robust, we will use p[8] and p[9] to allow a slight 
                temperature-dependent curvature in the hardening or to scale 
                the plastic term. 
                
    Actually, to strictly follow the 10 param limit and maximize flexibility 
    without overcomplicating the landscape, let's define:
    
    E(T) = p[0] + p[1]*T
    sigma_y(T) = p[2] + p[3]*T
    sigma_sat(T) = p[4] + p[5]*T
    H(T) = p[6] + p[7]*T
    
    We have 2 params left. The Voce model assumes the plastic strain starts 
    from 0. In reality, there might be a small offset or the elastic-plastic 
    transition might be smoother. 
    Let's add a small linear hardening term that doesn't saturate, or adjust 
    the temperature dependence of H to be quadratic? 
    Let's keep H linear but allow the saturation term to have a small 
    temperature-dependent offset or scale.
    
    Alternative use for p[8], p[9]:
    Let's make the hardening modulus H(T) slightly more flexible or add a 
    small quadratic term to the elastic modulus.
    E(T) = p[0] + p[1]*T + p[8]*T^2
    H(T) = p[6] + p[7]*T + p[9]*T^2
    
    This uses all 10 params effectively.
    
    Stress = E(T)*strain + (sigma_sat(T) - sigma_y(T)) * (1 - exp(-strain / H(T)))
    
    This model is:
    1. Smooth and differentiable everywhere.
    2. Physically interpretable.
    3. Naturally captures saturation at high strain.
    4. Linear in the basis functions for a fixed T, but the parameters 
       interact non-linearly in the exponential. However, the landscape is 
       generally well-behaved for BFGS if initialized reasonably.
    
    Note on H(T): If H(T) becomes negative or zero, exp(-strain/H) can blow up 
    or oscillate. We should ensure H(T) stays positive. 
    Since T is in [0.06, 1] and params start at 1, H(T) ~ 1+T ~ 1.06 to 2. 
    BFGS might wander into negative H. 
    To be safe, we can use a softplus or abs for H, but that introduces 
    non-differentiability at 0 for abs. Softplus is smooth.
    H_safe = softplus(p[6] + p[7]*T + p[9]*T^2)
    But softplus(x) = log(1+exp(x)). If x is large negative, it's ~0. 
    If x is large positive, it's ~x.
    Using softplus ensures H > 0.
    
    Let's implement with softplus for H(T) to ensure stability.
    """
    
    s = strain
    t = temp
    p = params
    
    # Temperature-dependent parameters
    # Elastic Modulus: Quadratic in T
    E_t = p[0] + p[1] * t + p[8] * t * t
    
    # Yield Stress: Linear in T
    sy_t = p[2] + p[3] * t
    
    # Saturation Stress: Linear in T
    ssat_t = p[4] + p[5] * t
    
    # Hardening Modulus: Quadratic in T, passed through softplus to ensure positivity
    # H_raw = p[6] + p[7] * t + p[9] * t * t
    # H_t = np.log1p(np.exp(H_raw))  # softplus
    
    # To avoid overflow in exp for large H_raw, we can use a stable softplus implementation
    H_raw = p[6] + p[7] * t + p[9] * t * t
    
    # Stable softplus: log(1 + exp(x))
    # For x > 20, log(1+exp(x)) ~ x
    # For x < -20, log(1+exp(x)) ~ exp(x) ~ 0
    # np.logaddexp(0, H_raw) is a stable way to compute log(1 + exp(H_raw))
    H_t = np.logaddexp(0.0, H_raw)
    
    # Ensure H_t is not too small to avoid division by zero in exp(-s/H_t)
    # If H_t is very small, exp(-s/H_t) -> 0 for s>0, which is fine (instant saturation).
    # If s is negative (compression), exp(-s/H_t) -> exp(positive) -> large.
    # The problem states strain range [-0.001911, 0.9636].
    # For negative strain, the plastic term (1 - exp(-s/H)) becomes (1 - exp(positive)) which is negative.
    # This might not be physical, but the linear elastic term E*s dominates for small s.
    # Let's check the magnitude.
    # If s = -0.001, H ~ 1. exp(0.001) ~ 1.001. 1 - 1.001 = -0.001.
    # Plastic term ~ (ssat - sy) * (-0.001).
    # Elastic term ~ E * (-0.001).
    # Sum ~ (E - (ssat - sy)) * (-0.001).
    # This seems reasonable.
    
    # Calculate the plastic hardening term
    # exp_term = np.exp(-s / H_t)
    # To prevent overflow if s/H_t is very negative (large positive s/H_t), 
    # but s is max 0.96, H_t min ~ 0 (if H_raw very negative).
    # If H_t is very small, -s/H_t is very negative, exp is 0. No overflow.
    # If s is negative, -s/H_t is positive. Max s is -0.0019. 
    # Min H_t? If H_raw is -100, H_t ~ 0. exp(0.0019/0) -> inf.
    # We need to clamp H_t to a minimum value to avoid division by zero issues 
    # causing inf in exp for negative strains.
    # Let's clamp H_t to at least 1e-6.
    H_t_safe = np.maximum(H_t, 1e-6)
    
    exp_term = np.exp(-s / H_t_safe)
    
    # Plastic contribution
    # The term (1 - exp_term) is 0 at s=0.
    # For s>0, it grows towards 1.
    # For s<0, it becomes negative.
    plastic_term = (ssat_t - sy_t) * (1.0 - exp_term)
    
    # Elastic contribution
    elastic_term = E_t * s
    
    stress = elastic_term + plastic_term
    
    return stress

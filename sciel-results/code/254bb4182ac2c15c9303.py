import numpy as np
import time

def equation(strain, temp, params):
    """
    Symbolic regression for stress-strain of an aluminium rod.
    
    Strategy: Physically-Motivated Non-Linear Model (Swift-Hollomon Form).
    
    The stress-strain curve of metals typically follows:
    1. Elastic region: Linear with slope E (Young's Modulus).
    2. Yield region: Transition from linear to plastic.
    3. Plastic hardening: Follows a power law K * epsilon_plastic^n.
    
    We model the total strain s as the sum of elastic strain s_e and plastic strain s_p.
    s_e = sigma / E
    s_p = (sigma / K)^(1/n)
    
    To make this a direct function of strain s (input) to stress sigma (output),
    we use an implicit form or a smooth approximation. A common smooth approximation
    that mimics the kink at yield is:
    
    sigma = E * s * (1 - smooth_step(s - s_y)) + (sigma_y + K * (s - s_y)^n) * smooth_step(s - s_y)
    
    However, this is piecewise-like. A more robust "closed-form" non-linear model
    that is smooth everywhere and fits BFGS well is the **Neural Unit** or **Softplus**
    based hardening, but let's stick to the physical power law structure which is 
    known to be accurate for aluminium.
    
    We define the model as:
    sigma = E(t) * s + (K(t) * (softplus(s - s_y(t)) / scale)^n(t) - E(t) * s_y(t) * (1 - sigmoid(...)))
    
    Actually, a simpler and very effective smooth approximation for the Hollomon equation
    combined with elastic part is:
    
    sigma = E(t) * s + K(t) * (s - s_y(t))^n(t) * H(s - s_y(t))
    
    To avoid the non-differentiability of H (Heaviside) at s=s_y, we use a smooth
    approximation for the plastic strain term:
    s_p = softplus(s - s_y) / alpha  (where alpha controls the sharpness of the yield)
    
    But softplus is log(1+exp(x)). For small x, it's quadratic. For large x, it's linear.
    This might not capture the power law n well if n != 1.
    
    Let's use a different smooth approximation that allows for power-law behavior:
    s_p = ( (s - s_y)^2 + eps^2 )^(n/2) / 2  ... this is getting complicated.
    
    Let's try a **Rational Function** that approximates the power law and elastic part.
    
    Alternative Strategy: **Gaussian Process Kernel-like Basis**
    Instead of physical laws, use a sum of Gaussian bumps to model the non-linear parts.
    This is universal approximator but with 10 params, we can't do many bumps.
    
    Let's go back to the **Swift-Hollomon** but implemented as a smooth continuous function.
    
    Model:
    sigma = E(t) * s * sigmoid(alpha * (s_y(t) - s)) 
          + (sigma_y(t) + K(t) * (s - s_y(t))^n(t)) * sigmoid(alpha * (s - s_y(t)))
    
    This is non-linear in E, K, n, s_y, sigma_y, alpha. 
    BFGS might struggle with n if it's far from 1.
    
    Let's simplify. The key feature of Aluminium is:
    1. High E (~70 GPa).
    2. Yield point ~0.1-0.2 GPa (strain ~0.002).
    3. Hardening is relatively slow.
    
    Let's use a **Piecewise Linear with Smooth Transitions** but with **Learnable Breakpoints**
    and **Learnable Slopes**. This is similar to the current program but the current one
    had fixed breakpoints. Making breakpoints learnable allows the model to find the
    exact yield strain and hardening onset.
    
    However, the current program (score 4.2) used fixed breakpoints and linear-in-params.
    The parent (score 2.5) used a 4th order polynomial.
    The inspiration (score 2.2) used a "State a strategy MATERIALLY DIFFERENT... single-structure PWL".
    
    Let's try a **Neural Network** style activation with a single hidden layer concept, 
    implemented as a sum of "bump" functions.
    
    Actually, the most robust non-linear model for this specific shape (linear then curve)
    with BFGS is often a **Quadratic + Cubic** term but constrained.
    
    Let's try: 
    sigma = c0 + c1*s + c2*s^2 + c3*s^3 + c4*s^4
    This is the polynomial approach. It scored 2.55.
    
    What if we use **Log-Strain**?
    sigma = c0 + c1*log(1+s/s0) ...
    
    Let's try a **Hyperbolic Sigmoid** based model which is smooth and bounded.
    sigma = A * (1 + tanh(B * (s - C))) + D * s + E
    
    This has 5 parameters. We have 10. We can add temperature dependence.
    A(t) = a0 + a1*t
    B(t) = b0 + b1*t
    C(t) = c0 + c1*t
    D(t) = d0 + d1*t
    E(t) = e0 + e1*t
    
    Total 10 params.
    This model is smooth everywhere.
    The tanh term captures the "kink" or change in slope.
    The linear term captures the elastic part.
    
    Let's refine the structure to be more physically aligned.
    Elastic: Linear.
    Plastic: Saturating or Power Law.
    
    Model:
    sigma = E(t) * s + K(t) * (s - s_y(t))^p(t) * H(s - s_y(t))
    
    To make it smooth and differentiable for BFGS, replace H with a smooth step S(x) = 0.5*(1+tanh(x/w)).
    And replace (s-s_y)^p with a smooth version.
    
    Let's use:
    sigma = E(t) * s * (1 - S(s - s_y(t))) + (E(t)*s_y(t) + K(t) * (S(s - s_y(t)) * w)^p(t)) * S(s - s_y(t))
    
    This is getting too complex and non-linear.
    
    **New Strategy: Separable Temperature Dependence**
    
    The temperature dependence in stress-strain is often multiplicative on the hardening term and additive/subtractive on the yield strength.
    
    Let's assume the shape of the curve in terms of normalized strain is similar across temperatures, but scaled.
    
    sigma(s, t) = E(t) * s + (sigma_y(t) + K(t) * (s - s_y(t))^n) * H(s - s_y(t))
    
    Let's implement a **Smooth Approximation** of this using **Softplus** for the plastic strain.
    s_p = softplus(s - s_y)
    sigma = E(t)*s + K(t) * s_p^n(t)
    
    Wait, if s < s_y, s_p is small (quadratic), so sigma is roughly linear + small quadratic.
    If s > s_y, s_p ~ s - s_y, so sigma ~ linear + power law.
    This matches the physics well!
    
    Parameters:
    1. E_base (Elastic Modulus base)
    2. E_temp (Elastic Modulus temp coeff)
    3. K_base (Hardening Coeff base)
    4. K_temp (Hardening Coeff temp coeff)
    5. n (Hardening Exponent) - assume constant or weak temp dep? Let's assume constant for simplicity to save params.
    6. s_y_base (Yield Strain base)
    7. s_y_temp (Yield Strain temp coeff)
    8. offset_base (Intercept)
    9. offset_temp (Intercept temp coeff)
    10. alpha (Sharpness of softplus, fixed? or param?)
    
    Let's fix alpha to a reasonable value (e.g., 100) to make softplus behave like max(x,0) but smooth.
    Or let alpha be a param? If alpha is large, it's sharp. If small, it's smooth.
    BFGS might struggle with alpha if it's very large (stiff gradients).
    Let's fix alpha = 50.
    
    So we have 9 params. We have 10.
    Let's make n temperature dependent? Or add a second hardening term?
    Let's add a linear hardening term for high strains?
    
    Model:
    s_p = softplus(s - s_y(t))
    sigma = E(t)*s + K1(t)*s_p + K2(t)*s_p^2
    
    This uses a quadratic hardening instead of power law. Quadratic hardening is a good approximation for many metals over limited strain ranges.
    
    Params:
    p[0]: E_base
    p[1]: E_temp
    p[2]: K1_base
    p[3]: K1_temp
    p[4]: K2_base
    p[5]: K2_temp
    p[6]: s_y_base
    p[7]: s_y_temp
    p[8]: offset_base
    p[9]: offset_temp
    
    All linear in E, K1, K2, offset. Non-linear in s_y.
    This is a very strong candidate. It is physically motivated, smooth, and uses all 10 params effectively.
    
    Implementation details:
    - s_y(t) = p[6] + p[7]*temp
    - E(t) = p[0] + p[1]*temp
    - K1(t) = p[2] + p[3]*temp
    - K2(t) = p[4] + p[5]*temp
    - offset(t) = p[8] + p[9]*temp
    
    s_p = np.logaddexp(0, (s - s_y) * alpha) / alpha  # softplus(x)/alpha
    # Using logaddexp for numerical stability of log(1+exp(x))
    
    sigma = offset(t) + E(t)*s + K1(t)*s_p + K2(t)*s_p**2
    
    This should be robust.
    """
    s = strain
    t = temp
    p = params
    
    # Extract parameters
    E_base = p[0]
    E_temp = p[1]
    K1_base = p[2]
    K1_temp = p[3]
    K2_base = p[4]
    K2_temp = p[5]
    sy_base = p[6]
    sy_temp = p[7]
    off_base = p[8]
    off_temp = p[9]
    
    # Calculate temperature-dependent parameters
    E = E_base + E_temp * t
    K1 = K1_base + K1_temp * t
    K2 = K2_base + K2_temp * t
    sy = sy_base + sy_temp * t
    offset = off_base + off_temp * t
    
    # Sharpness of the yield transition
    # Fixed to a value that provides a smooth but distinct kink
    alpha = 100.0
    
    # Plastic strain approximation using softplus
    # softplus(x) = log(1 + exp(x))
    # We use np.logaddexp(0, x) for stability
    # s_p = softplus(s - sy) / alpha
    # This ensures s_p is small and quadratic for s < sy, and linear for s > sy
    
    x = (s - sy) * alpha
    # Clip x to avoid overflow in exp if alpha is large and s-sy is large
    # logaddexp handles this well, but let's be safe
    s_p = np.logaddexp(0, x) / alpha
    
    # Calculate stress
    stress = offset + E * s + K1 * s_p + K2 * (s_p ** 2)
    
    return stress

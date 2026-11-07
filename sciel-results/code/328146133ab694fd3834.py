import numpy as np
import time

def equation(strain, temp, params):
    """
    Strategy: Physics-Informed Smooth Yield Model (Ramberg-Osgood Variant).
    
    Unlike the current program's fixed-breakpoint piecewise linear basis, 
    this model uses a continuous, differentiable transition from elastic 
    to plastic behavior. It mimics the Ramberg-Osgood equation:
    epsilon = sigma/E + (sigma/K)^n
    
    Inverted (stress as function of strain) is hard, so we model stress directly:
    sigma = E(t) * epsilon + K(t) * (epsilon - epsilon_y(t))^n * sigmoid-like_transition
    
    To ensure BFGS stability and differentiability, we use a smooth saturation 
    term instead of a hard power law or piecewise function.
    
    Model:
    sigma = E(t) * s + H(t) * s^alpha * (1 - exp(-beta * s))
    
    Or better, a smooth yield transition:
    sigma = E(t) * s + (K(t) * (s - s_y(t))^gamma) * sigmoid(k * (s - s_y(t)))
    
    Since s_y, k, gamma are non-linear parameters, we must be careful.
    However, the inspiration suggests that physical models like Voce work well.
    Let's try a Voce-type hardening combined with elastic term:
    
    sigma = E(t) * s + Y(t) + (sigma_sat(t) - Y(t)) * (1 - exp(-H(t) * (s - s_y(t))))
    
    This has 6-8 parameters. We have 10.
    Let's define:
    E(t) = p0 + p1*t
    Y(t) = p2 + p3*t  (Yield stress)
    sigma_sat(t) = p4 + p5*t (Saturation stress)
    H(t) = p6 + p7*t (Hardening modulus)
    s_y(t) = p8 + p9*t (Yield strain)
    
    Total 10 params.
    
    The term (s - s_y(t)) can be negative. We should clamp it to 0 for the hardening part?
    If we clamp, it's non-differentiable at s_y.
    Instead, use a smooth max: softplus(s - s_y) or just (s - s_y) if we assume s > s_y for plastic part?
    No, for small s, the plastic term should be 0.
    
    Let's use a smooth transition:
    plastic_term = (sigma_sat - Y) * (1 - exp(-H * max(s - s_y, 0)))
    This is non-differentiable at s=s_y.
    
    Better: Use a smooth approximation of max(x, 0):
    softplus(x) = log(1 + exp(x)) / beta. For large beta, approx max.
    But exp can overflow.
    
    Alternative: Use a sigmoid-based transition that is smooth everywhere.
    sigma = E(t) * s + (sigma_sat(t) - Y(t)) * sigmoid(k * (s - s_y(t)))
    
    This is smooth. At s < s_y, sigmoid is small, so plastic term is small.
    At s > s_y, sigmoid is large, so plastic term approaches (sigma_sat - Y).
    This creates a smooth plateau, not a hardening curve.
    
    To get hardening, we need the plastic term to grow with s.
    sigma = E(t) * s + H(t) * (s - s_y(t))^2 * sigmoid(k * (s - s_y(t)))
    
    This is smooth.
    Let's check parameters:
    p0, p1: E(t)
    p2, p3: H(t) (Plastic hardening coefficient)
    p4, p5: s_y(t) (Yield strain)
    p6, p7: k(t) (Transition sharpness)
    p8, p9: Offset or additional term?
    
    Let's add a constant offset p8 and a temp-dependent offset p9?
    Or use p8, p9 to modify the elastic part?
    
    Let's stick to the core physics:
    sigma = (p0 + p1*t) * s + (p2 + p3*t) * (s - (p4 + p5*t))^2 * sigmoid((p6 + p7*t) * (s - (p4 + p5*t)))
    
    We have 8 params. We have 2 left.
    Let's add a saturation limit to prevent infinite growth:
    Actually, the quadratic term grows.
    Maybe use a rational form for hardening?
    
    Let's try a different structure that is known to fit metals well:
    sigma = E * s / (1 + c * s^2) + Y * (s - s_y) / (1 + h * (s - s_y)^2)
    
    This is getting complex. Let's go back to the Voce model which is simple and effective:
    sigma = Y(t) + (sigma_sat(t) - Y(t)) * (1 - exp(-H(t) * (s - s_y(t))))
    
    But this assumes elastic part is zero? No, usually:
    sigma = E(t) * s (for s < s_y)
    sigma = Y(t) + (sigma_sat(t) - Y(t)) * (1 - exp(-H(t) * (s - s_y(t)))) (for s > s_y)
    
    To make it smooth and single-expression:
    sigma = E(t) * s * (1 - sigmoid(k*(s-s_y))) + (Y(t) + (sigma_sat(t) - Y(t)) * (1 - exp(-H(t) * (s - s_y)))) * sigmoid(k*(s-s_y))
    
    This is a blend of elastic and plastic.
    Params:
    E(t): p0, p1
    Y(t): p2, p3
    sigma_sat(t): p4, p5
    H(t): p6, p7
    s_y(t): p8, p9
    k: fixed? Or param?
    
    If k is fixed (e.g., 100), we use 10 params for the physical quantities.
    This is a strong structural change: blending two physical regimes with a smooth switch.
    
    Let's implement this.
    """
    s = strain
    t = temp
    p = params
    
    # Extract parameters
    E0, E1 = p[0], p[1]
    Y0, Y1 = p[2], p[3]
    Ssat0, Ssat1 = p[4], p[5]
    H0, H1 = p[6], p[7]
    sy0, sy1 = p[8], p[9]
    
    # Temperature-dependent parameters
    E = E0 + E1 * t
    Y = Y0 + Y1 * t
    Ssat = Ssat0 + Ssat1 * t
    H = H0 + H1 * t
    sy = sy0 + sy1 * t
    
    # Fixed sharpness for the transition to keep BFGS stable
    # If k is too large, gradients vanish or explode. 50-100 is usually good for strain scale ~1.
    k = 100.0
    
    # Smooth switch: 0 in elastic, 1 in plastic
    # Use tanh for better numerical stability than sigmoid in some cases, or sigmoid is fine.
    # sigmoid(x) = 1 / (1 + exp(-x))
    # To avoid overflow in exp, we can clip x.
    x = k * (s - sy)
    x_clipped = np.clip(x, -50, 50)
    switch = 1.0 / (1.0 + np.exp(-x_clipped))
    
    # Elastic part: linear
    sigma_elastic = E * s
    
    # Plastic part: Voce hardening
    # Note: (s - sy) can be negative. exp(-H * negative) = exp(positive) -> overflow.
    # We need to handle the domain.
    # If s < sy, the plastic term should be small/zero.
    # The Voce model is typically defined for s >= sy.
    # Let's use a smooth max for the plastic strain:
    # eps_p = max(s - sy, 0)
    # Smooth max: softplus(s - sy)
    # But softplus has a slope of 1 for large positive, 0 for large negative.
    # Let's just use (s - sy) and rely on the switch to turn it off?
    # If s < sy, switch is ~0. So the plastic term is multiplied by ~0.
    # But inside the plastic term, we have exp(-H * (s - sy)).
    # If s < sy, (s - sy) is negative. -H * (s - sy) is positive.
    # If H is large and (s-sy) is large negative, exp overflows.
    # So we must clip the argument of exp.
    
    eps_p = s - sy
    # Clip eps_p to be non-negative for the hardening calculation?
    # No, let's clip the exponent.
    arg = -H * eps_p
    arg_clipped = np.clip(arg, -50, 50)
    
    sigma_plastic = Y + (Ssat - Y) * (1.0 - np.exp(arg_clipped))
    
    # Blend
    sigma = (1.0 - switch) * sigma_elastic + switch * sigma_plastic
    
    return sigma

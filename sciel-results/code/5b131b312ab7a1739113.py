import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model using a generalized polynomial basis in strain and temperature,
    augmented with a smooth softplus term to capture yield-point nonlinearity.
    
    sigma = (p0 + p1*T)*u + (p2 + p3*T)*u^2 + (p4 + p5*T)*u^3 + (p6 + p7*T)*u^4 + (p8 + p9*T)
            + softplus(p_extra) * (u - u_yield)^2 * smoothstep
    
    Parameters (10):
    p0, p1: Coefficients for linear strain term
    p2, p3: Coefficients for quadratic strain term
    p4, p5: Coefficients for cubic strain term
    p6, p7: Coefficients for quartic strain term
    p8, p9: Temperature-dependent offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    u = strain
    T = temp
    
    # Precompute powers of strain
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    
    # Temperature-dependent coefficients
    c1 = p0 + p1 * T
    c2 = p2 + p3 * T
    c3 = p4 + p5 * T
    c4 = p6 + p7 * T
    c0 = p8 + p9 * T
    
    # Base polynomial
    sigma_poly = c1 * u + c2 * u2 + c3 * u3 + c4 * u4 + c0
    
    # Add a smooth nonlinear correction term to better capture yield behavior
    # Use softplus to create a smooth, always-positive weight for a quadratic
    # correction around the yield region. This adds flexibility without
    # introducing discontinuities that could hurt BFGS convergence.
    # The term is: softplus(p9) * (u - 0.002)^2 * (1 + T)
    # We reuse p9 as part of this term to keep parameter count at 10.
    # Actually, let's use a different approach: modulate the existing terms
    # with a smooth function of strain to create an effective "stiffness"
    # that changes with strain level.
    
    # Let's instead add: p9 * softplus(u) * u * (1 - T)
    # This adds a smooth nonlinear term that's active at positive strains
    # and modulated by temperature.
    
    # A better approach: use a smooth tanh-based correction
    # correction = p9 * tanh(u * 100) * u * (1 + 0.1*T)
    # But tanh with large argument can be stiff. Let's use a milder form.
    
    # Use: p9 * u * (1 - exp(-u * 50)) * (1 + 0.05*T)
    # This smoothly transitions from 0 to p9*u as strain increases.
    # But exp can underflow for large negative u. u range is [-0.0019, 0.9636]
    # So u*50 is in [-0.095, 48.18], exp is fine.
    
    # Let's use a simpler, more robust formulation:
    # Add p9 * u * sigmoid(u * 20) * (1 + 0.1 * T)
    # sigmoid is smooth and bounded.
    
    # Actually, to stay within the polynomial family but add smooth nonlinearity,
    # let's use: p9 * u * (1 + u * u) * (1 - T * 0.1)
    # This is still polynomial but with a different structure.
    
    # Best approach: keep it simple and just ensure the polynomial is well-conditioned.
    # The current model may be struggling with optimization. Let's try a 
    # reparameterization that helps BFGS converge better.
    
    # Use centered strain to improve conditioning
    u_centered = u - 0.5  # center around middle of range
    
    uc2 = u_centered * u_centered
    uc3 = uc2 * u_centered
    uc4 = uc2 * uc2
    
    # Temperature-dependent coefficients with centered strain
    sigma = c1 * u + c2 * u2 + c3 * u3 + c4 * u4 + c0
    
    # Add a small smooth nonlinear term using softplus for yield capture
    # softplus(x) = log(1 + exp(x)) for numerical stability
    # We'll use: p9 * softplus(20 * (u - 0.005)) * 0.01 * (1 + 0.1*T)
    # This adds a smooth step-like increase around u=0.005 (yield region)
    
    x = 20.0 * (u - 0.005)
    # Numerically stable softplus
    softplus_x = np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -50, 50))))
    
    sigma += p9 * softplus_x * 0.01 * (1.0 + 0.1 * T)
    
    return sigma

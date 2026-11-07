import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Model: Piecewise-linear with smooth hinge (elastic + plastic regimes)
    
    sigma = E(T)*eps + A(T) * f(eps)
    where f(eps) is a smooth approximation of max(0, eps - eps_y)
    using f(eps) = (eps - eps_y) * (1 - exp(-k*(eps - eps_y)))
    
    This gives:
    - For eps < eps_y: small plastic contribution (elastic dominated)
    - For eps > eps_y: approximately linear hardening with slope A(T)
    
    Parameters:
    p0, p1: Elastic modulus E(T) = p0 + p1*T
    p2, p3: Yield strain eps_y(T) = p2 + p3*T
    p4, p5: Hardening modulus A(T) = p4 + p5*T
    p6, p7: Transition sharpness k(T) = p6 + p7*T (kept positive via softplus-like)
    p8, p9: Additional offset/curvature term B(T)*eps^2 for shape flexibility
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    
    # Temperature-dependent coefficients
    E = p0 + p1 * temp          # Elastic modulus
    eps_y = p2 + p3 * temp      # Yield strain threshold
    A = p4 + p5 * temp          # Hardening modulus
    k = np.abs(p6 + p7 * temp) + 1e-6  # Transition sharpness (ensure positive)
    B = p8 + p9 * temp          # Quadratic shape term
    
    # Smooth hinge: f(x) = x * (1 - exp(-k*x)) for x > 0, small for x <= 0
    # We use a numerically stable version
    x = eps - eps_y
    
    # For large positive k*x, exp(-k*x) -> 0, so f -> x
    # For large negative k*x, exp(-k*x) -> inf, but x is negative, so x*(1-exp(-kx)) 
    # Actually let's use a safer formulation:
    # f(x) = (x + 1/k) * (1 - exp(-k*x)) - 1/k  ... this gets complicated
    # Simpler: use tanh-based smooth max
    # smooth_max(x, 0) ≈ (x + sqrt(x^2 + a^2))/2 for small a
    # But let's stick with the exponential form with clipping
    
    # Clip k*x to avoid overflow
    kx = k * x
    kx_clipped = np.clip(kx, -50, 50)
    
    # f(x) = x * (1 - exp(-kx))
    # For x < 0 and kx very negative, exp(-kx) is huge, but x is negative
    # x * (1 - exp(-kx)) = x - x*exp(-kx)
    # If x < 0, -kx > 0, exp(-kx) > 1, so x*exp(-kx) is negative * positive = negative
    # So x - (negative) = x + |x*exp(-kx)| which could be large positive? No:
    # x < 0, exp(-kx) > 1, x*exp(-kx) < 0, so 1 - exp(-kx) < 0
    # x * (negative) = positive. That's not right for a hinge.
    
    # Let me use a different smooth hinge:
    # g(x) = 0.5 * (x + sqrt(x^2 + a^2)) where a controls smoothness
    # This is always >= 0 and approximates max(x, 0)
    a = 1.0 / k  # Smaller k -> smoother
    a = np.clip(a, 1e-6, 100.0)
    
    # g(x) = 0.5 * (x + sqrt(x^2 + a^2))
    x_sq = x * x
    sqrt_term = np.sqrt(x_sq + a * a)
    g = 0.5 * (x + sqrt_term)
    
    # Plastic contribution: A * g(x) + B * eps^2
    plastic = A * g + B * eps * eps
    
    # Total stress
    sigma = E * eps + plastic
    
    return sigma

import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a piecewise smooth model with explicit elastic and plastic regimes,
    separated by a smooth hinge function. Temperature modulates both regimes.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent elastic modulus
    E_e = p0 + p1 * T
    
    # Temperature-dependent yield strain (positive threshold)
    eps_y = p2 + p3 * T
    
    # Temperature-dependent plastic hardening modulus
    E_p = p4 + p5 * T
    
    # Transition width for smooth hinge
    delta = p6 + p7 * T
    # Ensure delta is positive to avoid division issues
    delta_safe = np.maximum(np.abs(delta), 1e-8)
    
    # Smooth hinge: x * (1 - exp(-k*x)) where k = 1/delta
    # This gives 0 at x=0, approaches x for large x
    # Applied to plastic strain (eps - eps_y)
    plastic_strain = eps - eps_y
    # For positive plastic strain: smooth activation
    # For negative: we want 0 (no plastic deformation in elastic regime)
    # Use: max(0, plastic_strain) * (1 - exp(-plastic_strain/delta)) for positive
    # But we need smoothness at 0. Use a smooth rectification:
    # smooth_pos(x) = x * sigmoid(x/delta) approximately, or x*(1-exp(-x/delta)) for x>0
    # Let's use: for x > 0: x * (1 - exp(-x/delta)), for x <= 0: 0
    # But to keep it smooth and differentiable, use:
    # smooth_hinge(x) = x * (1 - exp(-x/delta)) for all x, but clip negative to avoid issues
    # Actually, let's use a different smooth form that's well-behaved:
    # plastic_active = plastic_strain * (0.5 * (1 + tanh(plastic_strain / delta_safe)))
    # This smoothly transitions from 0 to plastic_strain
    
    # Use tanh-based smooth rectification
    tanh_arg = plastic_strain / delta_safe
    # Clip to avoid overflow in tanh
    tanh_arg = np.clip(tanh_arg, -50, 50)
    smooth_factor = 0.5 * (1.0 + np.tanh(tanh_arg))
    
    # Plastic strain contribution (smoothly activated)
    plastic_strain_smooth = plastic_strain * smooth_factor
    
    # Elastic stress: E_e * eps, but reduced in plastic regime
    # Use (1 - smooth_factor) to reduce elastic contribution as plastic kicks in
    elastic_stress = E_e * eps * (1.0 - smooth_factor)
    
    # Plastic stress: yield stress + hardening
    # Yield stress = E_e * eps_y
    yield_stress = E_e * eps_y
    plastic_stress = yield_stress + E_p * plastic_strain_smooth
    
    # Combine: elastic part in elastic regime, plastic part in plastic regime
    # The smooth_factor blends between them
    stress = elastic_stress + plastic_stress * smooth_factor
    
    # Additional temperature and strain interaction terms for better fit
    stress += p8 * eps * T
    stress += p9 * eps * eps
    
    return stress

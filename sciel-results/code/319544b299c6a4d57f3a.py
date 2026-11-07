import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod using a gated polynomial hardening approach.
    
    Structure:
    1. Linear elastic response: E(T) * strain
    2. Smooth transition gate: sigmoid(linear combination of strain and temp)
    3. Hardening response: Gate * (Linear + Quadratic hardening terms dependent on T)
    
    This structure avoids non-differentiable points and unstable power laws,
    offering a smoother landscape for BFGS optimization.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # 1. Temperature-dependent elastic modulus
    E = p0 + p1 * temp
    
    # 2. Smooth yield transition gate
    # z is a linear combination of strain and temperature, shifted by p4
    # The sigmoid will be near 0 for small strain (elastic regime) 
    # and approach 1 for large strain (plastic regime)
    z = p2 * strain + p3 * temp + p4
    
    # Use a numerically stable sigmoid
    # np.clip prevents overflow in exp
    z_clipped = np.clip(z, -50, 50)
    gate = 1.0 / (1.0 + np.exp(-z_clipped))
    
    # 3. Hardening terms
    # Linear hardening coefficient depends on temperature
    H_lin = p5 + p6 * temp
    # Quadratic hardening coefficient depends on temperature
    H_quad = p7 + p8 * temp
    
    # Plastic stress contribution is gated
    plastic_stress = gate * (H_lin * strain + H_quad * strain**2)
    
    # Total stress
    stress = E * strain + plastic_stress + p9
    
    return stress

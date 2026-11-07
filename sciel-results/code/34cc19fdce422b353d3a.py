import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a smooth hinge model for the elastic-plastic transition
    combined with polynomial terms for hardening and temperature effects.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are arrays for broadcasting
    strain = np.asarray(strain)
    temp = np.asarray(temp)
    
    # Smooth hinge: s * (1 - exp(-k * s))
    # This allows independent control of elastic slope (E = p0) and yield plateau (Y = p1)
    # with k = p2 placing the yield threshold.
    # For small s: ~ p0 * p2 * s^2 (but we want linear, so let's use a different formulation)
    
    # Actually, let's use: E * s + Y * (1 - exp(-k * s))
    # where E is elastic modulus, Y is yield stress plateau, k controls transition sharpness
    # For small s: E*s + Y*k*s (linear)
    # For large s: E*s + Y (linear with offset, or we can subtract to make it plateau)
    
    # Better: use a form that gives linear rise then plateau
    # stress = E * s + Y * (1 - exp(-k * s)) - E * s * exp(-k * s)
    # Hmm, let me think differently.
    
    # Use: stress = E * s * (1 - exp(-k*s)) + Y * (1 - exp(-k*s))
    # For small s: (E*s + Y) * k*s ≈ E*k*s^2 + Y*k*s (not quite linear in s for the elastic part)
    
    # Let me use the standard smooth max/hinge approach:
    # f(s) = E * s for s < s_y, and Y + H*(s - s_y) for s >= s_y
    # Smooth approximation: E * s + (Y - E*s_y) * (1 - exp(-k*(s - s_y))) * (s > s_y smoothed)
    
    # Simpler: use the form from the cell findings
    # s * (1 - exp(-k*s)) lets fitter set E and Y independently
    # So: stress = E * s + Y * (1 - exp(-k * s))
    # where E = p0, Y = p1, k = p2
    
    # But this doesn't plateau, it keeps growing linearly with E*s.
    # For aluminum, after yield, stress should plateau or grow slowly.
    
    # Let's use: stress = E * s * exp(-k*s) + Y * (1 - exp(-k*s))
    # For small s: E*s + Y*k*s (linear, slope = E + Y*k)
    # For large s: Y (plateau)
    # This gives a smooth transition from linear to plateau!
    
    # Temperature dependence: E(T) = E0 + E1*T, Y(T) = Y0 + Y1*T
    # k can also depend on T: k(T) = k0 + k1*T
    
    # Using params:
    # p0 = E0, p1 = Y0, p2 = k0
    # p3 = E1, p4 = Y1, p5 = k1
    # p6-p9: polynomial correction terms
    
    E = p0 + p3 * temp
    Y = p1 + p4 * temp
    k = p2 + p5 * temp
    
    # Ensure k is positive for numerical stability
    k_safe = np.where(k > 0.1, k, 0.1)
    
    # Core model: elastic-plastic transition
    core = E * strain * np.exp(-k_safe * strain) + Y * (1.0 - np.exp(-k_safe * strain))
    
    # Polynomial correction terms for fine-tuning
    poly_term = (
        p6 * strain + 
        p7 * temp + 
        p8 * strain * temp + 
        p9 * temp**2
    )
    
    return core + poly_term

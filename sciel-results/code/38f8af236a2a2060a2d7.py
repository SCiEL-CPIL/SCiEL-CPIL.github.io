"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a multiplicative interaction model that decouples linear elastic scaling,
    temperature-dependent stiffness modulation, and non-linear strain saturation
    into separate smooth factors. This structure allows the optimizer to find
    a balance between the different physical regimes more effectively than
    additive models.
    
    Model:
    sigma = (p0 + p1*eps) * (p2 + p3*T) * (1 + p4*eps^p5) / (1 + p6*eps^p7 + p8*T^p9) + p9_offset
    
    Wait, I have 10 params. Let's map them carefully.
    Let's use the structure suggested in the directive:
    sigma = (p0 + p1*strain) * (p2 + p3*temp) * (1 + p4*strain^p5) / (1 + p6*strain^p7 + p8*temp^p9) + p10
    
    But I only have 10 params: p0..p9.
    So I will drop the final +p10 or absorb it.
    Let's try:
    sigma = (p0 + p1*eps) * (p2 + p3*T) * (1 + p4*eps) / (1 + p5*eps + p6*T + p7*eps*T) + p8*eps^2 + p9
    
    Actually, let's stick to the directive's form but fit it into 10 params.
    Form: (A) * (B) * (C) / (D)
    A = p0 + p1*eps
    B = p2 + p3*T
    C = 1 + p4*eps
    D = 1 + p5*eps + p6*T + p7*eps*T
    Remaining params: p8, p9.
    Add p8*eps^2 + p9 to the result.
    
    This gives:
    sigma = [(p0 + p1*eps) * (p2 + p3*T) * (1 + p4*eps)] / [1 + p5*eps + p6*T + p7*eps*T] + p8*eps^2 + p9
    
    This is structurally distinct (multiplicative/rational) and has 10 params.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Numerator factors
    A = p0 + p1 * eps
    B = p2 + p3 * T
    C = 1.0 + p4 * eps
    
    # Denominator factor
    D = 1.0 + p5 * eps + p6 * T + p7 * eps * T
    
    # Safety for denominator
    safe_D = np.where(np.abs(D) < 1e-8, np.sign(D) * 1e-8 if D != 0 else 1e-8, D)
    
    # Main multiplicative term
    term1 = (A * B * C) / safe_D
    
    # Correction terms
    term2 = p8 * eps * eps
    term3 = p9
    
    sigma = term1 + term2 + term3
    
    return sigma
# EVOLVE-BLOCK-END

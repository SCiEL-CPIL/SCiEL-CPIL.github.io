# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Structural change: rational + polynomial form instead of sigmoid blend.
    Uses a hyperbolic tangent for smooth yield transition and a power-law
    hardening with temperature-dependent coefficients.
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to [0, 1] range for better conditioning
    s = (strain + 0.002) / 0.966
    t = (temp - 0.06667) / 0.93333
    
    # Temperature-dependent elastic modulus and yield point
    E = p0 * (1.0 + p1 * t)  # E can increase or decrease with T
    sigma_y = p2 * (1.0 - p3 * t)  # Yield stress typically decreases with T
    
    # Elastic strain limit
    eps_y = np.maximum(sigma_y / np.maximum(np.abs(E), 1e-8), 1e-6)
    
    # Plastic strain
    eps_p = np.maximum(s - eps_y, 0.0)
    
    # Smooth transition using tanh (sharper than sigmoid for BFGS)
    # width controls yield sharpness
    w = np.maximum(np.abs(p4), 0.005)
    x = (s - eps_y) / w
    x_clamped = np.clip(x, -15, 15)
    # Transition: 0 in elastic, 1 in fully plastic
    trans = 0.5 * (1.0 + np.tanh(x_clamped))
    
    # Elastic branch: linear with slight nonlinearity
    stress_elastic = E * s + p5 * s**2
    
    # Plastic branch: power-law hardening with saturation
    # sigma_plastic = sigma_y * (1 + a * eps_p^b) / (1 + c * eps_p^d)
    a = np.maximum(p6, 0.01)
    b = np.maximum(p7, 0.3)
    c = np.maximum(p8, 0.0)
    d = np.maximum(p9, 0.3)
    
    # Safe power for small values
    eps_p_safe = np.maximum(eps_p, 1e-8)
    hardening_num = 1.0 + a * np.power(eps_p_safe, b)
    hardening_den = 1.0 + c * np.power(eps_p_safe, d)
    stress_plastic = sigma_y * hardening_num / np.maximum(hardening_den, 1e-8)
    
    # Blend elastic and plastic
    stress = trans * stress_plastic + (1.0 - trans) * stress_elastic
    
    # Ensure finite
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END

"""Stress-strain model for aluminium rod with temperature dependence."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a piecewise-linear model with smooth hinge functions to separate
    elastic and plastic regimes. The yield point is a tunable threshold.
    
    Model:
        E(T) = p0 + p1*T + p2*T^2
        eps_y(T) = p3 + p4*T + p5*T^2  (yield strain)
        H(T) = p6 + p7*T + p8*T^2      (hardening modulus)
        k = exp(p9)                     (transition sharpness)
        
        elastic = E(T) * strain * (1 - exp(-k*strain))
        plastic = H(T) * (strain - eps_y) * (1 - exp(-k*(strain - eps_y)))
        
        stress = elastic + plastic
    
    The hinge function x*(1-exp(-k*x)) is zero for x<=0 and approximates x for x>>0,
    creating a smooth transition at the yield point.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * temp + p2 * temp * temp
    
    # Temperature-dependent yield strain
    eps_y = p3 + p4 * temp + p5 * temp * temp
    
    # Temperature-dependent hardening modulus
    H = p6 + p7 * temp + p8 * temp * temp
    
    # Transition sharpness - use exp to ensure k > 0
    # Clamp p9 to prevent overflow
    p9_clamped = np.clip(p9, -20.0, 20.0)
    k = np.exp(p9_clamped)
    
    # Hinge function: x * (1 - exp(-k*x))
    # For x <= 0: result is 0 (approximately)
    # For x >> 0: result approaches x
    def hinge(x, k_val):
        # Clamp argument to prevent overflow
        arg = k_val * x
        arg_clamped = np.clip(arg, -20.0, 20.0)
        exp_term = np.exp(arg_clamped)
        return x * (1.0 - exp_term)
    
    # Elastic contribution: active for all strain
    # The hinge ensures smooth behavior
    elastic_arg = strain
    elastic_hinge = hinge(elastic_arg, k)
    elastic_part = E * elastic_hinge
    
    # Plastic contribution: active only for strain > eps_y
    plastic_arg = strain - eps_y
    plastic_hinge = hinge(plastic_arg, k)
    plastic_part = H * plastic_hinge
    
    stress = elastic_part + plastic_part
    
    return stress
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature."""
    s = strain
    t = temp
    p = params
    
    # Build a basis of functions that are linear in params.
    # This makes the optimization problem convex (least squares), so BFGS finds the global minimum.
    # We use up to 10 parameters.
    
    # Basis functions:
    # 1. Constant
    # 2. Linear strain
    # 3. Quadratic strain
    # 4. Cubic strain
    # 5. Quartic strain
    # 6. Linear temp
    # 7. Strain * Temp
    # 8. Quadratic strain * Temp
    # 9. Cubic strain * Temp
    # 10. Quartic strain * Temp
    
    # To improve conditioning, we can normalize strain and temp.
    # Strain range: [-0.001911, 0.9636]. Center approx 0.48, scale approx 0.48.
    # Temp range: [0.06667, 1.0]. Center approx 0.53, scale approx 0.47.
    
    # However, normalization constants should ideally be fitted or fixed based on data.
    # Since we don't have data, we use reasonable fixed normalizations based on the given ranges.
    # Let's use simple scaling to [0, 1] or similar to keep numbers well-conditioned.
    
    # s_norm = (s - s_min) / (s_max - s_min)
    # t_norm = (t - t_min) / (t_max - t_min)
    
    s_min = -0.001911
    s_max = 0.9636
    t_min = 0.06667
    t_max = 1.0
    
    s_norm = (s - s_min) / (s_max - s_min)
    t_norm = (t - t_min) / (t_max - t_min)
    
    # Basis vectors
    b1 = np.ones_like(s)
    b2 = s_norm
    b3 = s_norm ** 2
    b4 = s_norm ** 3
    b5 = s_norm ** 4
    b6 = t_norm
    b7 = s_norm * t_norm
    b8 = (s_norm ** 2) * t_norm
    b9 = (s_norm ** 3) * t_norm
    b10 = (s_norm ** 4) * t_norm
    
    # Linear combination
    result = (p[0] * b1 + 
              p[1] * b2 + 
              p[2] * b3 + 
              p[3] * b4 + 
              p[4] * b5 + 
              p[5] * b6 + 
              p[6] * b7 + 
              p[7] * b8 + 
              p[8] * b9 + 
              p[9] * b10)
              
    return result
# EVOLVE-BLOCK-END

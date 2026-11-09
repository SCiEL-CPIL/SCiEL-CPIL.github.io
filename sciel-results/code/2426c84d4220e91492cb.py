"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid mechanistic-empirical model for E. coli growth rate.
    
    Structural Features:
    1. Normalized inputs for BFGS stability.
    2. Parameterized Monod kinetics for substrate (flexible Ks).
    3. Quadratic terms for Temperature, pH, and Density (capturing optimal ranges/saturation).
    4. Key interactions: T*pH and Monod_S*T.
    
    Parameters (length 10):
    p[0]: Bias
    p[1]: Linear coeff for normalized b
    p[2]: Log-Ks parameter for Monod (Ks = exp(p2 - 2.0))
    p[3]: Linear coeff for normalized temp
    p[4]: Linear coeff for normalized pH
    p[5]: Quadratic coeff for normalized temp
    p[6]: Quadratic coeff for normalized pH
    p[7]: Quadratic coeff for normalized b
    p[8]: Interaction coeff for normalized T * pH
    p[9]: Interaction coeff for Monod_S * normalized T
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Normalize inputs to approximately [0, 1]
    # Ranges from task: b[1,5], s[1,50], temp[1,40], pH[2,12]
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 48.0
    t_n = (temp - 1.0) / 39.0
    p_n = (pH - 2.0) / 10.0
    
    # Parameterized Monod kinetics
    # Using exp(p2 - 2.0) ensures Ks > 0. 
    # At init p2=1, Ks=exp(-1)~0.37, which is reasonable for normalized s in [0,1].
    Ks = np.exp(p2 - 2.0)
    s_monod = s_n / (Ks + s_n)
    
    # Construct prediction
    y = p0
    y += p1 * b_n
    y += p3 * t_n + p5 * (t_n ** 2)
    y += p4 * p_n + p6 * (p_n ** 2)
    y += p7 * (b_n ** 2)
    y += p8 * (t_n * p_n)
    y += p9 * (s_monod * t_n)
    
    return y
# EVOLVE-BLOCK-END

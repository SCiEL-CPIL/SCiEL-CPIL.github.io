import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator acceleration.
    Combines a sparse, physically-motivated polynomial basis with input scaling
    to improve numerical conditioning for the BFGS optimizer.
    """
    # 1. Scale inputs to O(1) range based on training data ranges
    # x range: [-0.5728, 0.5275] -> mid: -0.02265, half_range: 0.55015
    # v range: [-0.1754, 0.2821] -> mid: 0.05335, half_range: 0.22875
    
    x_mid = -0.02265
    x_half = 0.55015
    
    v_mid = 0.05335
    v_half = 0.22875
    
    xs = (x - x_mid) / x_half
    vs = (v - v_mid) / v_half
    
    # 2. Compute scaled basis terms
    # Using the sparse basis from Inspiration 1 which scored ~2.0
    # Terms: x, x^3, v, v^3, x^2*v, x*v^2, bias
    
    xs2 = xs * xs
    vs2 = vs * vs
    xs3 = xs2 * xs
    vs3 = vs2 * vs
    xs2_vs = xs2 * vs
    xs_vs2 = xs * vs2
    
    # 3. Linear combination of basis terms with fitted parameters
    # params[0]: x
    # params[1]: x^3
    # params[2]: v
    # params[3]: v^3
    # params[4]: x^2*v
    # params[5]: x*v^2
    # params[6]: bias
    # params[7:10] unused (or could be added if needed, but sparse is better)
    
    return (
        params[0] * xs +
        params[1] * xs3 +
        params[2] * vs +
        params[3] * vs3 +
        params[4] * xs2_vs +
        params[5] * xs_vs2 +
        params[6]
    )

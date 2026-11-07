# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity."""
    # Scale inputs to improve conditioning (inputs are already O(1) but products benefit)
    xs = x / 0.55  # ~max |x|
    vs = v / 0.25  # ~max |v|
    
    xs2 = xs * xs
    xs3 = xs2 * xs
    xs4 = xs2 * xs2
    xs5 = xs4 * xs
    vs2 = vs * vs
    vs3 = vs2 * vs
    
    # Build basis terms (10 terms using all 10 params)
    t0 = xs
    t1 = xs3
    t2 = xs5
    t3 = vs
    t4 = vs3
    t5 = xs2 * vs
    t6 = xs * vs2
    t7 = xs2
    t8 = vs2
    t9 = 1.0
    
    return (params[0] * t0 +
            params[1] * t1 +
            params[2] * t2 +
            params[3] * t3 +
            params[4] * t4 +
            params[5] * t5 +
            params[6] * t6 +
            params[7] * t7 +
            params[8] * t8 +
            params[9])
# EVOLVE-BLOCK-END

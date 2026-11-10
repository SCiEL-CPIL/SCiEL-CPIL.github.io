"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a non-linear parametric model:
    A sum of three sine waves with learnable amplitudes, frequencies, and phases,
    plus a DC offset.
    
    Parameters:
    p[0:3] : Amplitudes (a1, a2, a3)
    p[3:6] : Frequencies (w1, w2, w3)
    p[6:9] : Phases (phi1, phi2, phi3)
    p[9]   : DC Offset (c)
    
    y = c + a1*sin(w1*x + phi1) + a2*sin(w2*x + phi2) + a3*sin(w3*x + phi3)
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Extract parameters
    a1, a2, a3 = params[0], params[1], params[2]
    w1, w2, w3 = params[3], params[4], params[5]
    phi1, phi2, phi3 = params[6], params[7], params[8]
    c = params[9]
    
    # Compute the sum of sines
    # Using soft clipping on frequencies to prevent instability if optimizer goes wild?
    # The range [0.1, 6] is small. If w becomes very large, sin(wx) oscillates fast.
    # BFGS might struggle. Let's keep it simple first, but maybe constrain w to be positive?
    # Since we can't easily constrain in a simple expression without extra params, 
    # we rely on the optimizer. However, negative frequency sin(-wx + phi) is just -sin(wx - phi),
    # which is equivalent to sin(wx + phi') with different amplitude/phase. So sign doesn't matter much.
    
    term1 = a1 * np.sin(w1 * x + phi1)
    term2 = a2 * np.sin(w2 * x + phi2)
    term3 = a3 * np.sin(w3 * x + phi3)
    
    return c + term1 + term2 + term3
# EVOLVE-BLOCK-END

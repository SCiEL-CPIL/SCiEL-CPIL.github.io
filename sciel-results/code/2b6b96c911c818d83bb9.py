# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form NumPy expression for the acceleration of a nonlinear oscillator.
    
    Uses up to 10 fitted constants (params).
    Combines scaled polynomial terms with tanh nonlinearities for better
    conditioning and flexibility.
    
    Model:
    a = p0 + p1*xs + p2*vs + p3*xs^2 + p4*vs^2 + p5*xs*vs 
        + p6*xs^3 + p7*tanh(xs) + p8*tanh(vs) + p9*xs*vs^2
    
    where xs = x / 0.6 and vs = v / 0.3
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Scale inputs to roughly unit magnitude for better conditioning
    # Ranges: x ~ [-0.57, 0.53], v ~ [-0.18, 0.28]
    xs = x / 0.6
    vs = v / 0.3
    
    # Extract parameters
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]
    p4 = p[4]
    p5 = p[5]
    p6 = p[6]
    p7 = p[7]
    p8 = p[8]
    p9 = p[9]
    
    # Compute terms
    xs2 = xs * xs
    vs2 = vs * vs
    xs3 = xs2 * xs
    
    term = (p0 + 
            p1 * xs + 
            p2 * vs + 
            p3 * xs2 + 
            p4 * vs2 + 
            p5 * xs * vs + 
            p6 * xs3 + 
            p7 * np.tanh(xs) + 
            p8 * np.tanh(vs) + 
            p9 * xs * vs2)
            
    return term
# EVOLVE-BLOCK-END

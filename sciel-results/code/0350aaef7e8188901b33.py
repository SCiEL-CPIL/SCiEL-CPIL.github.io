"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Ensure we handle the fixed size of params (10)
    # If params is longer, we ignore the extra; if shorter, we pad with zeros (though task says length 10)
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    else:
        p = p[:10]

    # Scale input to [-1, 1] for better conditioning of trigonometric terms
    # Original range is [-3, 3]
    u = x / 3.0
    
    # Basis functions:
    # 0: Constant
    # 1: Linear
    # 2: Sine (1x)
    # 3: Cosine (1x)
    # 4: Sine (2x)
    # 5: Cosine (2x)
    # 6: Sine (3x)
    # 7: Cosine (3x)
    # 8: Sine (4x)
    # 9: Cosine (4x)
    
    # This basis is orthogonal-ish and stable for BFGS.
    # It can represent periodic functions and smooth curves well.
    
    term0 = np.ones_like(u)
    term1 = u
    term2 = np.sin(np.pi * u)
    term3 = np.cos(np.pi * u)
    term4 = np.sin(2.0 * np.pi * u)
    term5 = np.cos(2.0 * np.pi * u)
    term6 = np.sin(3.0 * np.pi * u)
    term7 = np.cos(3.0 * np.pi * u)
    term8 = np.sin(4.0 * np.pi * u)
    term9 = np.cos(4.0 * np.pi * u)
    
    return (p[0] * term0 + 
            p[1] * term1 + 
            p[2] * term2 + 
            p[3] * term3 + 
            p[4] * term4 + 
            p[5] * term5 + 
            p[6] * term6 + 
            p[7] * term7 + 
            p[8] * term8 + 
            p[9] * term9)
# EVOLVE-BLOCK-END

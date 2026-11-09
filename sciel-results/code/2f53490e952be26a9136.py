# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Normalized Fourier series approximation.
    
    We use a constant term, sine and cosine terms up to the 4th harmonic,
    and the 5th harmonic sine term.
    
    The basis functions are normalized to have unit L2 norm on [-2, 2]
    to improve the conditioning of the optimization problem for BFGS.
    
    Basis:
      e_0 = 1/2
      e_{2k-1} = (1/sqrt(2)) * sin(k * pi/2 * x)
      e_{2k}   = (1/sqrt(2)) * cos(k * pi/2 * x)
      
    Mapping to params (length 10):
      p0 * e_0
      p1 * e_1 (sin k=1)
      p2 * e_2 (cos k=1)
      p3 * e_3 (sin k=2)
      p4 * e_4 (cos k=2)
      p5 * e_5 (sin k=3)
      p6 * e_6 (cos k=3)
      p7 * e_7 (sin k=4)
      p8 * e_8 (cos k=4)
      p9 * e_9 (sin k=5)
    """
    # Precompute scaling factors
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    inv_2 = 0.5
    
    # Constant term: norm of 1 on [-2,2] is sqrt(4)=2. So 1/2 has norm 1.
    out = params[0] * inv_2
    
    # Harmonics k=1 to 4
    for k in range(1, 5):
        freq = k * np.pi / 2.0
        sin_term = np.sin(freq * x) * inv_sqrt2
        cos_term = np.cos(freq * x) * inv_sqrt2
        
        # params[1] is sin(1), params[2] is cos(1)
        # params[3] is sin(2), params[4] is cos(2)
        # ...
        idx_sin = 2 * k - 1
        idx_cos = 2 * k
        
        out += params[idx_sin] * sin_term
        out += params[idx_cos] * cos_term
        
    # 5th harmonic sine (params[9])
    # Note: In the previous code, params[9] was sin(5*pi/2*x).
    # idx for sin(5) would be 2*5 - 1 = 9.
    # So it fits the pattern if we extended the loop to k=5 for sine only.
    freq5 = 5 * np.pi / 2.0
    sin_term5 = np.sin(freq5 * x) * inv_sqrt2
    out += params[9] * sin_term5
    
    return out
# EVOLVE-BLOCK-END

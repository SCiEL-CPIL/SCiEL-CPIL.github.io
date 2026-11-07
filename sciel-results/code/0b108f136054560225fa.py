import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series basis model for nonlinear oscillator dynamics.
    
    Uses trigonometric basis functions to capture periodic phase-space
    dynamics: individual harmonics in x and v, and inter-harmonic couplings.
    
    Parameters (10):
      p0: sin(x)
      p1: cos(x) - 1
      p2: sin(v)
      p3: cos(v) - 1
      p4: sin(x + v)
      p5: cos(x + v) - 1
      p6: sin(x - v)
      p7: cos(x - v) - 1
      p8: sin(2x)
      p9: cos(2x) - 1
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Individual harmonics
    sx = np.sin(x)
    cx = np.cos(x) - 1.0
    sv = np.sin(v)
    cv = np.cos(v) - 1.0
    
    # Inter-harmonic couplings
    sxp = np.sin(x + v)
    cxp = np.cos(x + v) - 1.0
    sxm = np.sin(x - v)
    cxm = np.cos(x - v) - 1.0
    
    # Second harmonics in x
    s2x = np.sin(2.0 * x)
    c2x = np.cos(2.0 * x) - 1.0
    
    result = (p0 * sx + p1 * cx + p2 * sv + p3 * cv +
              p4 * sxp + p5 * cxp + p6 * sxm + p7 * cxm +
              p8 * s2x + p9 * c2x)
    
    return result

# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Trigonometric harmonic basis with mixed frequencies and cross-terms.
    
    This basis captures periodic/oscillatory dependencies that are structurally
    orthogonal to polynomial or exponential bases. All terms are linear in params,
    ensuring BFGS sees a convex (least-squares) problem.
    
    Basis functions (10 total):
      p[0]: 1                          (DC offset)
      p[1]: sin(x)                     (fundamental x harmonic)
      p[2]: cos(y)                     (fundamental y harmonic)
      p[3]: sin(z)                     (fundamental z harmonic)
      p[4]: cos(2*x)                   (2nd harmonic x)
      p[5]: sin(2*y)                   (2nd harmonic y)
      p[6]: cos(3*z)                   (3rd harmonic z)
      p[7]: sin(x + y)                 (cross-term xy)
      p[8]: cos(y + z)                 (cross-term yz)
      p[9]: sin(x + z)                 (cross-term xz)
    
    The non-integer frequency multipliers (2, 3) and phase-shifted cross-terms
    provide coverage of wave-like structures at multiple scales and directions.
    """
    p = params
    return (p[0]
            + p[1] * np.sin(x)
            + p[2] * np.cos(y)
            + p[3] * np.sin(z)
            + p[4] * np.cos(2.0 * x)
            + p[5] * np.sin(2.0 * y)
            + p[6] * np.cos(3.0 * z)
            + p[7] * np.sin(x + y)
            + p[8] * np.cos(y + z)
            + p[9] * np.sin(x + z))
# EVOLVE-BLOCK-END

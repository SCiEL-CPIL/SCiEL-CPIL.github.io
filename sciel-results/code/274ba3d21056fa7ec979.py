# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: full quadratic polynomial (10 terms) plus a radial basis
    term centered at a single point with one width parameter.
    
    The quadratic part spans the space of all degree-<=2 polynomials in 3D.
    The RBF term adds a localized, non-polynomial component that can capture
    peaks, dips, or localized structure that a pure polynomial cannot.
    
    params layout (10 total):
      params[0]:  c0  (constant)
      params[1]:  c1  (x coefficient)
      params[2]:  c2  (y coefficient)
      params[3]:  c3  (z coefficient)
      params[4]:  c4  (x^2 coefficient)
      params[5]:  c5  (xy coefficient)
      params[6]:  c6  (xz coefficient)
      params[7]:  c7  (y^2 coefficient)
      params[8]:  c8  (yz coefficient)
      params[9]:  c9  (z^2 coefficient)
    
    The RBF term is: exp(-((x-xc)^2 + (y-yc)^2 + (z-zc)^2) / sigma^2)
    with fixed center (1.25, 1.25, 1.25) and fixed width sigma = 0.5.
    Its coefficient is folded into the constant term c0 to keep exactly 10 params.
    
    Wait - that's not right. We need the RBF to have its own coefficient.
    Let me reconsider: use 9 polynomial terms + 1 RBF coefficient.
    Drop one polynomial term (e.g., z^2) and replace with RBF weight.
    
    Actually, better approach: keep all 10 as a mix. Use 7 polynomial terms
    and 3 RBF-related params (center_x, center_y, width) - no, that changes
    the semantics too much.
    
    Cleanest structural change: replace the full quadratic with a model that
    includes both polynomial AND non-polynomial basis functions, all linear
    in params, so BFGS still sees a convex problem.
    
    New basis (10 terms, all linear in params):
      1, x, y, z, x^2, y^2, z^2, xy, xz, yz  -> that's the full quadratic.
    
    To add non-polynomial capability while keeping 10 params, replace some
    polynomial terms with non-polynomial ones:
      1, x, y, z, exp(-x), exp(-y), exp(-z), sin(x), sin(y), sin(z)
    
    This is essentially inspiration 2's approach but let me think about what
    gives the best coverage.
    
    Actually, the key insight: the current program (quadratic/Legendre) scores
    ~7.39, while the mixed poly+trig+exp scores ~1.30. The mixed basis is
    clearly better. Let me use that basis but with a structural refinement:
    add interaction terms between the non-polynomial features.
    
    Refined basis (10 terms):
      p[0]: 1
      p[1]: x
      p[2]: y
      p[3]: z
      p[4]: exp(-x)
      p[5]: exp(-y)
      p[6]: exp(-z)
      p[7]: x * exp(-x)
      p[8]: y * exp(-y)
      p[9]: z * exp(-z)
    
    This adds the ability to model functions like x*exp(-x) which have a
    peak and decay, covering more functional forms than pure exp(-x).
    """
    p = params
    return (p[0]
            + p[1] * x
            + p[2] * y
            + p[3] * z
            + p[4] * np.exp(-x)
            + p[5] * np.exp(-y)
            + p[6] * np.exp(-z)
            + p[7] * x * np.exp(-x)
            + p[8] * y * np.exp(-y)
            + p[9] * z * np.exp(-z))
# EVOLVE-BLOCK-END

import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Quadratic Polynomial Basis in Log-Space.
    
    This model class assumes the target function can be well-approximated by 
    a polynomial in the logarithms of the inputs. This is equivalent to a 
    generalized power-law model:
    Y ~ c0 + c1*ln(x) + ... + c8*ln(x)ln(y) + ...
    
    This is often superior to raw polynomials for positive inputs because:
    1. It naturally captures multiplicative interactions (x*y becomes ln(x)+ln(y)).
    2. It handles scale invariance better.
    3. The domain [0.5, 2] maps to a bounded, well-behaved log domain.
    
    Basis (10 terms):
    0: 1
    1: ln(x)
    2: ln(y)
    3: ln(z)
    4: (ln(x))^2
    5: (ln(y))^2
    6: (ln(z))^2
    7: ln(x)*ln(y)
    8: ln(x)*ln(z)
    9: ln(y)*ln(z)
    """
    p = params
    
    # Compute logs
    # Inputs are guaranteed to be in [0.5, 2], so log is safe and real.
    lx = np.log(x)
    ly = np.log(y)
    lz = np.log(z)
    
    # Compute squares and products
    lx2 = lx * lx
    ly2 = ly * ly
    lz2 = lz * lz
    
    lxy = lx * ly
    lxz = lx * lz
    lyz = ly * lz
    
    # Linear combination
    return (p[0] + 
            p[1] * lx + 
            p[2] * ly + 
            p[3] * lz + 
            p[4] * lx2 + 
            p[5] * ly2 + 
            p[6] * lz2 + 
            p[7] * lxy + 
            p[8] * lxz + 
            p[9] * lyz)

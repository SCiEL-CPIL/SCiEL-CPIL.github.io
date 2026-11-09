import numpy as np
from scipy.special import chebyt

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a separable Chebyshev polynomial basis.
    
    Strategy:
    - Uses Chebyshev polynomials of the first kind, which are orthogonal on [-1, 1].
    - This provides superior numerical conditioning compared to monomial bases,
      helping BFGS converge more reliably and accurately.
    - The basis consists of a product of Chebyshev polynomials in x and y.
    - We select 10 terms: 
      x-degrees: 0, 1, 2, 3
      y-degrees: 0, 1, 2
      Excluding the highest-order mixed terms to fit the 10-parameter limit.
      Specifically:
      (0,0), (1,0), (2,0), (3,0)
      (0,1), (1,1), (2,1)
      (0,2), (1,2), (2,2)
    """
    p = params
    
    # Precompute Chebyshev polynomials for x and y
    # Degrees needed for x: 0, 1, 2, 3
    # Degrees needed for y: 0, 1, 2
    
    T_x0 = np.ones_like(x)
    T_x1 = x
    T_x2 = 2 * x**2 - 1
    T_x3 = 4 * x**3 - 3 * x
    
    T_y0 = np.ones_like(y)
    T_y1 = y
    T_y2 = 2 * y**2 - 1
    
    # Construct the 10 basis terms
    # Indexing:
    # 0: T_x0 * T_y0
    # 1: T_x1 * T_y0
    # 2: T_x2 * T_y0
    # 3: T_x3 * T_y0
    # 4: T_x0 * T_y1
    # 5: T_x1 * T_y1
    # 6: T_x2 * T_y1
    # 7: T_x0 * T_y2
    # 8: T_x1 * T_y2
    # 9: T_x2 * T_y2
    
    term_0 = T_x0 * T_y0
    term_1 = T_x1 * T_y0
    term_2 = T_x2 * T_y0
    term_3 = T_x3 * T_y0
    term_4 = T_x0 * T_y1
    term_5 = T_x1 * T_y1
    term_6 = T_x2 * T_y1
    term_7 = T_x0 * T_y2
    term_8 = T_x1 * T_y2
    term_9 = T_x2 * T_y2
    
    return (p[0] * term_0 + 
            p[1] * term_1 + 
            p[2] * term_2 + 
            p[3] * term_3 + 
            p[4] * term_4 + 
            p[5] * term_5 + 
            p[6] * term_6 + 
            p[7] * term_7 + 
            p[8] * term_8 + 
            p[9] * term_9)

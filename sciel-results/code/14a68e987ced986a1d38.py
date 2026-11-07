import numpy as np

def equation(x, y, params):
    # Use a cubic polynomial basis (monomials)
    # This structural change moves from Chebyshev to monomials,
    # which showed higher score in inspiration 2.
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x**2
    b4 = x * y
    b5 = y**2
    b6 = x**3
    b7 = x**2 * y
    b8 = x * y**2
    b9 = y**3

    return (params[0]*b0 + params[1]*b1 + params[2]*b2 + params[3]*b3 +
            params[4]*b4 + params[5]*b5 + params[6]*b6 + params[7]*b7 +
            params[8]*b8 + params[9]*b9)

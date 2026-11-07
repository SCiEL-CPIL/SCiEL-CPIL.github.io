"""
Rewritten program: 2D Chebyshev polynomial basis for symbolic regression.

Strategy:
Use a basis of Chebyshev polynomials of the first kind in x and y.
Chebyshev polynomials are orthogonal on [-1, 1] with respect to the weight
1/sqrt(1-x^2), but even without weighting, they provide much better
numerical conditioning than monomials (x^n) or trigonometric functions
for the BFGS optimizer.

Basis functions (10 total):
1. T0(x)T0(y) = 1
2. T1(x)T0(y) = x
3. T0(x)T1(y) = y
4. T2(x)T0(y) = 2x^2 - 1
5. T0(x)T2(y) = 2y^2 - 1
6. T1(x)T1(y) = xy
7. T3(x)T0(y) = 4x^3 - 3x
8. T0(x)T3(y) = 4y^3 - 3y
9. T2(x)T1(y) = (2x^2 - 1)y = 2x^2y - y
10. T1(x)T2(y) = x(2y^2 - 1) = 2xy^2 - x

This spans the space of polynomials up to degree 3 in x and y, plus
some cross terms, providing a rich, well-conditioned feature space.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a 2D Chebyshev polynomial basis.
    
    The 10 parameters are weights for the following basis functions:
    p0: 1
    p1: x
    p2: y
    p3: 2x^2 - 1
    p4: 2y^2 - 1
    p5: x * y
    p6: 4x^3 - 3x
    p7: 4y^3 - 3y
    p8: (2x^2 - 1) * y
    p9: x * (2y^2 - 1)
    """
    # Compute Chebyshev polynomials
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x**2 - 1.0
    T3_x = 4.0 * x**3 - 3.0 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y**2 - 1.0
    T3_y = 4.0 * y**3 - 3.0 * y
    
    # Combine parameters with basis functions
    # p0: T0_x * T0_y
    # p1: T1_x * T0_y
    # p2: T0_x * T1_y
    # p3: T2_x * T0_y
    # p4: T0_x * T2_y
    # p5: T1_x * T1_y
    # p6: T3_x * T0_y
    # p7: T0_x * T3_y
    # p8: T2_x * T1_y
    # p9: T1_x * T2_y
    
    f = (params[0] * T0_x * T0_y +
         params[1] * T1_x * T0_y +
         params[2] * T0_x * T1_y +
         params[3] * T2_x * T0_y +
         params[4] * T0_x * T2_y +
         params[5] * T1_x * T1_y +
         params[6] * T3_x * T0_y +
         params[7] * T0_x * T3_y +
         params[8] * T2_x * T1_y +
         params[9] * T1_x * T2_y)
         
    return f
# EVOLVE-BLOCK-END

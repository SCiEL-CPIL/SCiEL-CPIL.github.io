"""
Strategy: Use a rich basis of Chebyshev polynomials (up to order 6) in x and y.
This provides a stable, global approximation space that can capture smooth
bivariate functions. The evaluator will fit the 10 parameters using BFGS.
Since we have 10 params and a potentially richer basis, we need to be careful.
Wait, the params array has length 10.
The seed uses 3 params: p0*x + p1*y + p2.
We have 10 params available.
We can use a basis of 10 terms.
Let's pick a basis that spans a good range of functions.
Terms:
1. 1
2. x
3. y
4. x^2
5. y^2
6. xy
7. x^3
8. y^3
9. x^2*y
10. x*y^2

This is a polynomial of degree 3 (mostly).
Actually, Chebyshev polynomials are better conditioned.
T0=1, T1=x, T2=2x^2-1, T3=4x^3-3x, etc.
Let's use Chebyshev products up to total degree 3 or 4.
Degree 0: 1
Degree 1: T1(x), T1(y)
Degree 2: T2(x), T2(y), T1(x)T1(y)
Degree 3: T3(x), T3(y), T2(x)T1(y), T1(x)T2(y)
Total 10 terms. This fits exactly 10 params.

Basis:
b0 = 1
b1 = T1(x) = x
b2 = T1(y) = y
b3 = T2(x) = 2x^2 - 1
b4 = T2(y) = 2y^2 - 1
b5 = T1(x)T1(y) = xy
b6 = T3(x) = 4x^3 - 3x
b7 = T3(y) = 4y^3 - 3y
b8 = T2(x)T1(y) = (2x^2-1)y = 2x^2y - y
b9 = T1(x)T2(y) = x(2y^2-1) = 2xy^2 - x

This basis is orthogonal-ish on [-1,1] if x,y are independent uniform.
It should be well-conditioned for BFGS.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Compute Chebyshev polynomials
    T1x = x
    T2x = 2 * x**2 - 1
    T3x = 4 * x**3 - 3 * x
    
    T1y = y
    T2y = 2 * y**2 - 1
    T3y = 4 * y**3 - 3 * y
    
    # Basis functions
    b0 = np.ones_like(x)
    b1 = T1x
    b2 = T1y
    b3 = T2x
    b4 = T2y
    b5 = T1x * T1y
    b6 = T3x
    b7 = T3y
    b8 = T2x * T1y
    b9 = T1x * T2y
    
    # Linear combination
    pred = (params[0] * b0 + params[1] * b1 + params[2] * b2 + 
            params[3] * b3 + params[4] * b4 + params[5] * b5 + 
            params[6] * b6 + params[7] * b7 + params[8] * b8 + 
            params[9] * b9)
    
    return pred
# EVOLVE-BLOCK-END

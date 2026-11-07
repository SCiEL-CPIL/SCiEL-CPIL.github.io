"""
Strategy: Rich polynomial basis with internal L-BFGS optimization.
The evaluator calls equation(x, y, params) and uses BFGS to fit params.
We ignore params for the model structure but use them as a seed for deterministic
initialization if needed. We build a basis of monomials up to degree 6 in (x, y)
and use scipy.optimize.minimize (L-BFGS-B) internally to fit the weights
to the training data (x, y) provided.

Key insight: The 'params' argument is what the outer BFGS fits. However, we can
treat the entire model as a fixed-basis linear model where the *weights* are
what we need to fit. Since the outer BFGS only fits 10 params, we cannot
directly fit a high-dimensional basis.

Alternative Strategy:
The outer BFGS fits 10 parameters. We need a function f(x, y, params) that is
flexible enough to approximate a wide class of functions using only 10 free
parameters.
A simple linear model (score 1.0) is too weak.
A quadratic model has 6 terms: 1, x, y, x^2, xy, y^2. That's 6 params.
A cubic model has 10 terms: 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3.
This matches the 10 parameters exactly!

Let's try a full cubic polynomial in x and y.
Basis:
1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3

This is a linear model in the parameters. The outer BFGS will fit these 10
coefficients. Since the model is linear in params, the loss is convex and
BFGS should converge quickly to the global minimum (which is the same as
least squares). This should be robust and fast.

Why this might beat 1.0:
If the target function is a low-degree polynomial (degree <= 3), this will
recover it exactly (or very closely). If it's higher degree, it will still
approximate it better than a linear model for smooth functions on [-1, 1].

We need to be careful about scaling. x, y are in [-1, 1]. Powers up to 3 are
well-conditioned.

Let's verify the number of terms:
Degree 0: 1
Degree 1: x, y
Degree 2: x^2, xy, y^2
Degree 3: x^3, x^2y, xy^2, y^3
Total: 1 + 2 + 3 + 4 = 10. Perfect.

Implementation:
def equation(x, y, params):
    # x, y are arrays of shape (N,)
    # params is an array of shape (10,)
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    return (p0 * 1 + 
            p1 * x + 
            p2 * y + 
            p3 * x**2 + 
            p4 * x * y + 
            p5 * y**2 + 
            p6 * x**3 + 
            p7 * x**2 * y + 
            p8 * x * y**2 + 
            p9 * y**3)

This is a closed-form expression. It is linear in params.
The outer BFGS will minimize MSE. Since it's linear, we could even use
np.linalg.lstsq, but the evaluator uses BFGS. BFGS on a quadratic loss
function (MSE of a linear model) is very efficient.

Wait, is there any risk?
If the target function is not a polynomial, this is the best cubic approximation.
Is there a better 10-parameter model?
Maybe a neural network with 1 hidden layer?
But the evaluator uses BFGS starting from all ones. Deep networks are hard to
optimize with BFGS from a bad start.
A radial basis function (RBF) network with a few centers?
We have 10 params. We could use 2 centers (4 params for centers) and 6 weights?
But the centers would need to be optimized too.
The cubic polynomial is the most "standard" and robust 10-parameter model
for symbolic regression on a square domain. It is complete for all polynomials
of degree <= 3.

Let's double check if we can do better with a different basis.
Chebyshev polynomials are better conditioned.
T0, T1, T2, T3 for x and y.
Product terms:
T0(x)T0(y), T1(x)T0(y), T0(x)T1(y), ...
The space of polynomials of degree <= 3 is the same dimension (10).
Chebyshev basis is just a linear transformation of the monomial basis.
Since the outer optimizer fits the coefficients, the span of the model is
identical. The conditioning might be slightly better for Chebyshev, but
monomials up to degree 3 on [-1,1] are already well-conditioned.
So, monomials are fine and simpler.

One potential issue: The evaluator minimizes training MSE.
If the target is e.g. sin(pi*x), the cubic approximation is good.
If the target is 1/(1+x^2), the cubic approximation is decent.

Let's stick with the full cubic polynomial.

Code:
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Unpack the 10 parameters
    p = params
    
    # Compute the full cubic polynomial in x and y
    # Terms:
    # 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3
    
    x2 = x * x
    x3 = x2 * x
    y2 = y * y
    y3 = y2 * y
    xy = x * y
    
    pred = (
        p[0] * 1.0 +
        p[1] * x +
        p[2] * y +
        p[3] * x2 +
        p[4] * xy +
        p[5] * y2 +
        p[6] * x3 +
        p[7] * x2 * y +
        p[8] * x * y2 +
        p[9] * y3
    )
    
    return pred
# EVOLVE-BLOCK-END

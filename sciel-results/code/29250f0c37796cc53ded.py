"""
Strategy: Separable Tensor-Product Chebyshev Basis.

The current program uses a Radial Basis Function (RBF) basis with 10 fixed centers.
While RBFs are universal approximators, with only 10 parameters and a fixed grid,
the optimization landscape can be difficult for BFGS, especially if the centers are
not well-chosen or if the target function has significant global structure that
requires careful weighting of distant centers.

This new strategy uses a separable tensor-product basis of Chebyshev polynomials.
We use Chebyshev polynomials of the first kind $T_n(t)$ for $n=0,1,2$ in both $x$ and $y$.
The basis functions are:
$B_{ij}(x,y) = T_i(x) T_j(y)$ for $i \in \{0,1,2\}$ and $j \in \{0,1,2\}$.
This gives $3 \times 3 = 9$ basis functions. We have 10 parameters.
We can add one more term to make it 10, e.g., $T_3(x)T_0(y) = (4x^3-3x) \cdot 1$ or simply use a 10th basis function like $T_2(x)T_2(y)$ is already there.
Let's list them:
0: T0(x)T0(y) = 1
1: T1(x)T0(y) = x
2: T0(x)T1(y) = y
3: T2(x)T0(y) = 2x^2 - 1
4: T1(x)T1(y) = xy
5: T0(x)T2(y) = 2y^2 - 1
6: T2(x)T1(y) = (2x^2-1)y
7: T1(x)T2(y) = x(2y^2-1)
8: T2(x)T2(y) = (2x^2-1)(2y^2-1)
9: We need a 10th. Let's add T3(x)T0(y) = (4x^3-3x) or T0(x)T3(y) or just repeat a useful one?
Actually, a 4x4 grid would be 16, too many.
A 3x3 grid is 9.
Maybe use a 2x2 grid of higher degree? No.
Let's stick to the 9 Chebyshev terms and add a 10th term that is orthogonal-ish or just a higher degree term.
However, the prompt asks for a structural change. The key is the *separable tensor-product* nature and the *Chebyshev* basis.
With 10 params, we can include $T_3(x)T_0(y)$ as the 10th term to capture cubic behavior in x.
Or better, just use the 9 terms and let the 10th param be unused? No, BFGS will optimize it.
If we define 10 basis functions, the 10th should be distinct.
Let's use $T_3(x)T_0(y) = 4x^3 - 3x$.

Why this is different:
1. **Well-conditioned**: Chebyshev polynomials are orthogonal on [-1,1] (with weight), leading to a much better-conditioned Gram matrix than monomials or RBFs with close centers.
2. **Global Structure**: Unlike RBFs which are local, Chebyshev polynomials capture global trends. If the function is smooth, a low-degree polynomial is an excellent approximation.
3. **Separability**: The tensor product structure allows the model to capture interactions between x and y (e.g., xy, x^2y) explicitly and efficiently.

Implementation:
We compute the Chebyshev values for x and y up to degree 3 (to have enough for the 10th term).
T0(t) = 1
T1(t) = t
T2(t) = 2t^2 - 1
T3(t) = 4t^3 - 3t

Basis matrix construction:
For each point (x_i, y_i):
b_0 = T0(x_i)T0(y_i)
b_1 = T1(x_i)T0(y_i)
b_2 = T0(x_i)T1(y_i)
b_3 = T2(x_i)T0(y_i)
b_4 = T1(x_i)T1(y_i)
b_5 = T0(x_i)T2(y_i)
b_6 = T2(x_i)T1(y_i)
b_7 = T1(x_i)T2(y_i)
b_8 = T2(x_i)T2(y_i)
b_9 = T3(x_i)T0(y_i)

Prediction = sum_{k=0}^9 params[k] * b_k
"""

# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a separable tensor-product Chebyshev basis.
    """
    # Deadline guard: return current best (which is just the linear combination) if we run out of time?
    # The computation is O(N), so it's very fast. No real need for a complex deadline guard, 
    # but we can add a simple check if N is huge, though N=200 is small.
    
    # Compute Chebyshev polynomials for x and y
    # T0 = 1
    # T1 = t
    # T2 = 2t^2 - 1
    # T3 = 4t^3 - 3t
    
    t0_x = np.ones_like(x)
    t1_x = x
    t2_x = 2.0 * x**2 - 1.0
    t3_x = 4.0 * x**3 - 3.0 * x
    
    t0_y = np.ones_like(y)
    t1_y = y
    t2_y = 2.0 * y**2 - 1.0
    # t3_y not needed for the 10th term chosen, but let's keep it simple.
    
    # Construct the 10 basis functions
    # b_0 = T0(x)T0(y)
    # b_1 = T1(x)T0(y)
    # b_2 = T0(x)T1(y)
    # b_3 = T2(x)T0(y)
    # b_4 = T1(x)T1(y)
    # b_5 = T0(x)T2(y)
    # b_6 = T2(x)T1(y)
    # b_7 = T1(x)T2(y)
    # b_8 = T2(x)T2(y)
    # b_9 = T3(x)T0(y)
    
    # We can compute this efficiently without creating a huge intermediate array if we just sum.
    # But N=200, so creating a (N, 10) array is trivial.
    
    basis = np.column_stack((
        t0_x * t0_y,
        t1_x * t0_y,
        t0_x * t1_y,
        t2_x * t0_y,
        t1_x * t1_y,
        t0_x * t2_y,
        t2_x * t1_y,
        t1_x * t2_y,
        t2_x * t2_y,
        t3_x * t0_y
    ))
    
    # Linear combination
    f = basis @ params
    
    return f
# EVOLVE-BLOCK-END

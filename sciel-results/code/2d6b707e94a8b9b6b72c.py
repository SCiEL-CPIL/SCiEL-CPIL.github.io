"""
Strategy: Self-contained local optimization.
The 'equation' function ignores the external BFGS optimization of 'params' for the final prediction.
Instead, it treats the call as an opportunity to perform its own high-quality fitting.
It uses a small, fixed-architecture neural network (2 hidden layers, tanh activation) 
with a parameter count <= 10 (or mapped to 10). 
Wait, the evaluator fits 'params' (length 10). 
If I do my own fitting, I need access to y. 
The task says: "You do not see the data, only the score of each candidate." 
AND "The evaluator fits params... minimizing the training MSE." 
This implies the evaluator has x_train, y_train. 
But 'equation' only receives x and params. It does NOT receive y. 
Therefore, I CANNOT fit the model inside 'equation' because I don't have y.

CRITICAL REALIZATION: 
The current program (and most in this lineage) are linear-in-params. 
The evaluator uses BFGS to optimize the 10 parameters to minimize MSE on training data. 
If the function is non-linear in params, BFGS might get stuck. 
If the function is linear in params, BFGS (or any least-squares solver) will find the global optimum for that fixed basis. 

The current score is ~88 million, which is terrible (var(y) is likely ~1, so MSE is huge). 
This suggests the current basis (Chebyshev + 2 RBFs) is a poor approximation of the target, OR the fit failed. 
Given "noiseless" and "uniform [-3,3]", a polynomial of degree 9 (10 params) should be able to fit many smooth functions well. 
The current program uses 8 Chebyshev + 2 RBFs. 

Why is the score so bad? 
1. The target might be highly non-smooth (e.g., step, abs, sin with high frequency). 
2. The RBFs at fixed centers -1, 1 might be useless. 
3. Chebyshev polynomials of degree 0-7 might not be enough. 

A STRUCTURAL CHANGE: 
Use a **Pure Polynomial Basis of Higher Degree** or **Different Basis Functions**. 
Since we have exactly 10 parameters, we can use a polynomial of degree 9. 
Chebyshev polynomials are the best-conditioned orthogonal polynomials for least-squares fitting on [-1,1]. 
The current program uses T_0 to T_7 (8 terms) and 2 RBFs. 
This wastes 2 parameters on potentially useless RBFs and limits the polynomial degree to 7. 

New Strategy: 
Use **10 Chebyshev Polynomials** T_0 to T_9. 
This maximizes the ability to capture smooth global trends with the given 10 parameters. 
It is strictly linear in params, ensuring BFGS converges to the global optimal projection onto this subspace. 
If the target is a polynomial of degree <= 9, this will recover it exactly (score ~1.0). 
If the target is smooth, this will be a much better approximation than mixing in RBFs which might not align with the target's features.

To be even more robust, we can use **Scaled Legendre Polynomials** or just **Monomials**? 
No, Chebyshev is best for conditioning. 
However, if the target has local features, a pure global polynomial might struggle. 
But with only 10 params, we can't do much local fitting anyway. 
The "Inspiration 3" with score 641 billion is even worse. 
The "Inspiration 1" with score 1.0 is the seed program? No, seed is 88M. 
Wait, Inspiration 1 says "score 1.0000". That is a perfect score. 
The seed program in the prompt is the one with score 88M. 
The prompt says: "CURRENT PROGRAM (score 88287292.4400)". 
But Inspiration 1 says "score 1.0000". 
This implies there exists a solution that scores 1.0. 
If the target is a polynomial of degree <= 9, then a degree 9 polynomial basis will score 1.0. 
The current program uses degree 7 Chebyshev + RBFs. It fails to reach 1.0. 
This strongly suggests the target is a polynomial of degree <= 9, or at least well-approximated by one, and the current basis is suboptimal due to the RBFs or lower degree. 

Therefore, the structural change is: **Replace the hybrid basis with a pure Chebyshev basis of degree 0-9.**
"""

import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a pure Chebyshev polynomial basis of degree 0 to 9.
    
    Rationale:
    - The problem has 10 parameters.
    - Chebyshev polynomials are orthogonal on [-1, 1] and provide the best-conditioned basis for polynomial approximation.
    - Using all 10 parameters for a degree 9 polynomial maximizes the flexibility to capture smooth functions, including polynomials up to degree 9 exactly.
    - The current program wastes 2 parameters on fixed RBFs and only uses degree 7, which is likely why it fails to achieve a high score (assuming the target is smooth/polynomial).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
    
    # Normalize x to [-1, 1] for Chebyshev stability
    x_norm = x / 3.0
    
    # Build basis matrix for T_0 to T_9
    # T[0, :] = 1
    # T[1, :] = x_norm
    # T[k, :] = 2 * x_norm * T[k-1, :] - T[k-2, :]
    
    N = len(x_norm)
    T = np.zeros((10, N), dtype=np.float64)
    
    if N == 0:
        return np.array([])
        
    T[0, :] = 1.0
    T[1, :] = x_norm
    
    for k in range(2, 10):
        T[k, :] = 2 * x_norm * T[k-1, :] - T[k-2, :]
        
    # Linear combination: y = params @ T
    # params is shape (10,), T is shape (10, N)
    # result is shape (N,)
    return T.T @ params

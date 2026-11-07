"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Simple linear-in-parameters model.
    The evaluator uses BFGS to optimize `params` to minimize training MSE.
    Since the model is linear in params, the optimal solution is the least squares solution.
    BFGS starting from all ones will converge to this solution (or close to it) 
    because the loss landscape is convex (a paraboloid).
    
    We use a basis of common functions that might appear in symbolic regression tasks.
    Keeping the basis simple avoids the optimization difficulties seen in the previous 
    complex basis (score ~4390). The seed program (score 1.0) was likely simple.
    
    Basis:
    0: 1 (bias)
    1: x
    2: x^2
    3: x^3
    4: sin(x)
    5: cos(x)
    6: tanh(x)
    7: x^4
    8: x^5
    9: x^6
    
    Note: Higher degree polynomials can be ill-conditioned, but with only 200 points 
    and x in [-3,3], they should be manageable. If BFGS struggles, it might just 
    converge to a local minimum if the problem were non-linear, but here it is linear.
    Actually, for linear models, the gradient is linear, so BFGS is efficient.
    """
    x = np.asarray(x, dtype=float)
    p = params
    
    # Ensure x is a 1D array for consistent broadcasting, though x is already 1D
    # f0 = 1.0
    # f1 = x
    # f2 = x**2
    # f3 = x**3
    # f4 = np.sin(x)
    # f5 = np.cos(x)
    # f6 = np.tanh(x)
    # f7 = x**4
    # f8 = x**5
    # f9 = x**6
    
    # To be safer against ill-conditioning and optimization issues, 
    # let's stick to a very robust set. 
    # The seed program (score 1.0) was `p0*x + p1`. 
    # Let's try a slightly richer but still linear basis.
    
    result = (
        p[0] * 1.0 +
        p[1] * x +
        p[2] * x**2 +
        p[3] * x**3 +
        p[4] * np.sin(x) +
        p[5] * np.cos(x) +
        p[6] * np.tanh(x)
    )
    
    # We have 10 params. Let's use the remaining 3 for higher order terms if needed, 
    # but let's keep it simple to ensure convergence. 
    # Actually, unused params are fine, BFGS will just optimize them to 0 or whatever 
    # minimizes the loss (which is 0 contribution if multiplied by 0? No, they multiply terms).
    # If I don't use p[7], p[8], p[9], they will be optimized but have no effect on output.
    # This is wasteful but safe.
    
    # Let's use all 10 to be expressive, but with well-scaled terms.
    # x is in [-3, 3].
    # x^2 in [0, 9]
    # x^3 in [-27, 27]
    # x^4 in [0, 81]
    # x^5 in [-243, 243]
    # x^6 in [0, 729]
    
    # The variance of x^6 is high. This might make the Hessian ill-conditioned.
    # Let's scale the higher order terms or just use lower orders.
    
    # Let's try:
    # 0: 1
    # 1: x
    # 2: x^2
    # 3: x^3
    # 4: sin(x)
    # 5: cos(x)
    # 6: tanh(x)
    # 7: x^4 / 10  (scaled down)
    # 8: x^5 / 100 (scaled down)
    # 9: x^6 / 1000 (scaled down)
    
    # This scaling helps with numerical stability.
    
    f0 = 1.0
    f1 = x
    f2 = x**2
    f3 = x**3
    f4 = np.sin(x)
    f5 = np.cos(x)
    f6 = np.tanh(x)
    f7 = x**4 / 10.0
    f8 = x**5 / 100.0
    f9 = x**6 / 1000.0
    
    result = (
        p[0] * f0 +
        p[1] * f1 +
        p[2] * f2 +
        p[3] * f3 +
        p[4] * f4 +
        p[5] * f5 +
        p[6] * f6 +
        p[7] * f7 +
        p[8] * f8 +
        p[9] * f9
    )
    
    return result
# EVOLVE-BLOCK-END

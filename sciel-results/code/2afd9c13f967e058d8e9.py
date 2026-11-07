"""Hybrid model: Chebyshev polynomial basis with bounded rational modulation for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    
    # Ensure x and y are 1D arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    
    # Extract parameters
    p = params
    
    # We have 10 parameters.
    # Strategy:
    # 1. Construct a rich basis of functions of x and y.
    # 2. Use a linear combination of these basis functions, but with a "soft" denominator
    #    that allows for rational-like behavior without poles.
    #
    # Basis functions (Chebyshev-like for stability, plus trig for periodicity):
    # T0(x) = 1
    # T1(x) = x
    # T2(x) = 2x^2 - 1
    # T0(y) = 1
    # T1(y) = y
    # T2(y) = 2y^2 - 1
    # Interactions: x*y, x^2*y, x*y^2, sin(x), sin(y), cos(x), cos(y)
    #
    # With only 10 params, we must be selective.
    # Let's try a structure:
    # f(x,y) = (A + B*x + C*y + D*x^2 + E*y^2 + F*x*y) / (1 + (G*sin(x) + H*sin(y) + I*cos(x) + J*cos(y))^2)
    # This is similar to the current program but let's refine the basis.
    #
    # Alternative: A pure linear model in a rich feature space is often more robust for BFGS.
    # Features:
    # f1 = 1
    # f2 = x
    # f3 = y
    # f4 = x^2
    # f5 = y^2
    # f6 = x*y
    # f7 = sin(x)
    # f8 = sin(y)
    # f9 = cos(x)
    # f10 = cos(y)
    #
    # This uses all 10 params as coefficients for a linear combination of these 10 features.
    # This is a very standard and robust approach for symbolic regression when the function
    # is a sum of simple terms. It avoids the non-convexity of rational functions.
    #
    # Let's check if the hidden function might be rational. The inspiration mentions rational
    # functions. However, the current score (29.5) is worse than the parent (17.8).
    # The parent used: numerator (quad poly) / (1 + trig^2).
    #
    # Let's try a hybrid:
    # Use 6 params for a quadratic polynomial numerator.
    # Use 4 params for a trigonometric denominator.
    # This is exactly the current program's structure. Why did it score worse?
    # Maybe the squared denominator made it too flat or hard to fit?
    # Or maybe the polynomial part is insufficient?
    #
    # Let's try a different structure:
    # f(x,y) = p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*x*y 
    #          + p6*sin(x) + p7*sin(y) + p8*cos(x) + p9*cos(y)
    #
    # This is a linear model in a 10-dim feature space.
    # Advantages:
    # - Convex optimization problem (if we view it as linear regression).
    # - BFGS should converge quickly and reliably.
    # - Can represent many functions (polynomial + periodic).
    # Disadvantages:
    # - Cannot represent products of features (e.g., x*sin(x)).
    #
    # Let's consider if the function could be x*sin(y) or similar.
    # If so, the linear model will fail.
    #
    # Let's try a model that includes some interaction terms but stays linear in params.
    # Features:
    # 1, x, y, x^2, y^2, x*y, x*sin(y), y*sin(x), sin(x), sin(y)
    #
    # This covers:
    # - Polynomial up to degree 2
    # - Pure trig
    # - Some cross-terms with trig
    #
    # Let's evaluate the computational cost.
    # 200 points * 10 features = 2000 operations. Very fast.
    
    # Feature 1: Constant
    f1 = np.ones_like(x)
    # Feature 2: x
    f2 = x
    # Feature 3: y
    f3 = y
    # Feature 4: x^2
    f4 = x**2
    # Feature 5: y^2
    f5 = y**2
    # Feature 6: x*y
    f6 = x * y
    # Feature 7: sin(x)
    f7 = np.sin(x)
    # Feature 8: sin(y)
    f9 = np.sin(y)  # Note: using index 9 for sin(y) to keep order intuitive? No, let's stick to 0-9.
    
    # Let's map params p0..p9 to these features:
    # p0: 1
    # p1: x
    # p2: y
    # p3: x^2
    # p4: y^2
    # p5: x*y
    # p6: sin(x)
    # p7: sin(y)
    # p8: cos(x)
    # p9: cos(y)
    
    # Wait, I missed cos in the feature list above. Let's define them all.
    f1 = np.ones_like(x)
    f2 = x
    f3 = y
    f4 = x**2
    f5 = y**2
    f6 = x * y
    f7 = np.sin(x)
    f8 = np.sin(y)
    f9 = np.cos(x)
    f10 = np.cos(y)
    
    # Result is a linear combination
    result = (p[0] * f1 + p[1] * f2 + p[2] * f3 + p[3] * f4 + 
              p[4] * f5 + p[5] * f6 + p[6] * f7 + p[7] * f8 + 
              p[8] * f9 + p[9] * f10)
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
# EVOLVE-BLOCK-END

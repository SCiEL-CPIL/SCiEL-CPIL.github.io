import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Hybrid basis covering polynomial interactions AND multiplicative non-polynomial terms.
    
    The current island best (1.30) is an additive non-polynomial basis: 1, x, y, z, exp(x), exp(y), exp(z), log(x), log(y), log(z).
    The previous mutation (4.52) tried strong interactions like exp(x+y+z) and log(x*y*z), which likely hurt conditioning or overfit.
    
    Key insight from inspirations:
    - Inspiration 2 (score 7.37) and findings suggest normalizing inputs improves conditioning.
    - The target function is hidden, but inputs are in [0.5, 2].
    - A good balance is needed between flexibility (interactions) and stability (conditioning).
    
    New Basis Design (10 terms):
    We want to capture:
    1. Constant
    2. Linear terms (x, y, z)
    3. Pairwise polynomial interactions (x*y, y*z, z*x) - these are fundamental in many physical laws
    4. One multiplicative exponential: exp(x*y) - captures strong non-linear coupling
    5. One multiplicative log: log(x*y) - captures soft non-linear coupling
    
    Wait, that's 1+3+3+1+1 = 9 terms. We have 10 params.
    Let's add x*y*z as the 10th? Or maybe exp(x+y) is safer than exp(x*y)?
    exp(x*y) for x,y in [0.5,2] -> x*y in [0.25, 4], exp in [1.28, 54.6]. Reasonable.
    log(x*y) -> log in [-1.38, 1.38]. Very safe.
    
    Let's stick to:
    b0 = 1
    b1 = x
    b2 = y
    b3 = z
    b4 = x*y
    b5 = y*z
    b6 = z*x
    b7 = exp(x*y)
    b8 = log(x*y)  # or maybe log(x*z) or log(y*z)? Let's use log(x*y) as representative
    b9 = x*y*z
    
    Actually, x*y*z might be redundant with the other terms if the function is simple, but it's a distinct monomial.
    
    However, the conditioning of exp(x*y) might be tricky for BFGS if the scale is off.
    Let's normalize the exponential term slightly? No, keep it simple first.
    
    Alternative: Use a Gaussian RBF term?
    b9 = exp(-(x-1.25)^2 - (y-1.25)^2 - (z-1.25)^2)  # Centered bump
    
    Let's try a mix:
    b0 = 1
    b1 = x
    b2 = y
    b3 = z
    b4 = x*y
    b5 = y*z
    b6 = z*x
    b7 = exp(x + y + z)  # This failed before? The 4.52 score had exp(x+y+z).
    
    Let's look at the 4.52 score again. It had exp(x+y+z) and log(x*y*z).
    The 1.30 score had exp(x), exp(y), exp(z), log(x), log(y), log(z).
    
    Why is 1.30 better than 4.52?
    Perhaps exp(x+y+z) is too large and causes numerical issues or overfitting?
    exp(x+y+z) range: exp(1.5) to exp(6) ~ 4.5 to 403.
    exp(x) range: 1.6 to 7.3.
    
    The magnitude difference is huge. BFGS might struggle.
    
    Let's try a basis that includes pairwise products but avoids the huge dynamic range.
    
    Basis:
    1. 1
    2. x
    3. y
    4. z
    5. x*y
    6. y*z
    7. z*x
    8. sin(x)  # Oscillatory? Unlikely, but cheap.
    9. cos(y)
    10. x^2  # Quadratic in one dimension
    
    This feels arbitrary.
    
    Let's go with a more principled approach:
    Normalized inputs to improve conditioning.
    x_n = (x - 1.25) / 0.75  # maps [0.5, 2] to [-1, 1]
    y_n = (y - 1.25) / 0.75
    z_n = (z - 1.25) / 0.75
    
    Basis on normalized inputs:
    1. 1
    2. x_n
    3. y_n
    4. z_n
    5. x_n * y_n
    6. y_n * z_n
    7. z_n * x_n
    8. x_n * y_n * z_n  # Full interaction
    9. exp(x_n)  # Non-linear, bounded range exp([-1,1]) ~ [0.37, 2.71]
    10. exp(y_n)  # Another non-linear
    
    This seems balanced. The normalization helps BFGS. The interactions cover multiplicative behavior. The exp terms cover non-polynomial curvature.
    
    Let's implement this.
    """
    try:
        n = len(x)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        
        # Normalize to [-1, 1] for better conditioning
        center = 1.25
        half_range = 0.75
        x_n = (x - center) / half_range
        y_n = (y - center) / half_range
        z_n = (z - center) / half_range
        
        p = params
        
        # Basis functions
        b0 = np.ones_like(x)
        b1 = x_n
        b2 = y_n
        b3 = z_n
        
        # Pairwise interactions
        b4 = x_n * y_n
        b5 = y_n * z_n
        b6 = z_n * x_n
        
        # Triple interaction
        b7 = x_n * y_n * z_n
        
        # Non-polynomial terms (normalized to avoid huge values)
        b8 = np.exp(x_n)
        b9 = np.exp(y_n)
        
        output = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
                  p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
                  p[8] * b8 + p[9] * b9)
        
        if not np.all(np.isfinite(output)):
            return np.zeros(n)
            
        return output

    except Exception:
        try:
            return np.zeros(len(x))
        except:
            return np.array([0.0])

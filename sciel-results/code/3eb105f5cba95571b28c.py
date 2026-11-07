import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Local Linear Regression with Multi-Start Grid Search.
    
    The 10 parameters are interpreted as:
    - p[0]: Intercept (bias)
    - p[1]: Coefficient for x
    - p[2]: Coefficient for y
    - p[3]: Coefficient for z
    - p[4]: Coefficient for x*y
    - p[5]: Coefficient for x*z
    - p[6]: Coefficient for y*z
    - p[7]: Coefficient for x^2
    - p[8]: Coefficient for y^2
    - p[9]: Coefficient for z^2
    
    This is a quadratic polynomial basis. However, to improve flexibility, 
    we use a "local" weighting scheme where the model is a weighted average 
    of predictions from multiple local quadratic models centered at different 
    points in the domain. But wait, we only have 10 parameters.
    
    Alternative Strategy: 
    Use the 10 parameters to define a flexible basis that is locally adaptive.
    Let's try a different approach: 
    Interpret the 10 parameters as defining a neural network with a single hidden layer 
    but with a specific activation that allows for better fitting.
    
    Actually, let's look at the inspirations. 
    Inspiration 1: Neural Network with single hidden layer.
    Inspiration 3: Hybrid basis.
    
    Let's try a Neural Network with 1 hidden layer.
    We have 3 inputs, 1 output.
    Let h be the number of hidden units.
    Weights: 3*h + h (biases) for first layer.
    Weights: h*1 + 1 (bias) for second layer.
    Total params: 4h + 1.
    If h=2, total = 9. We have 10.
    If h=2, we can use:
    w1 (3x2), b1 (2x1), w2 (2x1), b2 (1x1). Total = 6+2+2+1 = 11. Too many.
    If h=2, we can drop one bias? Or use h=2 and share some weights?
    
    Let's try h=2 with no bias in hidden layer?
    w1 (3x2), b1 (2x1) -> 6+2=8.
    w2 (2x1), b2 (1x1) -> 2+1=3. Total 11.
    
    Let's try h=1?
    w1 (3x1), b1 (1x1) -> 4.
    w2 (1x1), b2 (1x1) -> 2. Total 6. Too few.
    
    Let's try a different architecture.
    What if we use a basis function approach but with more terms?
    The current RBF approach scored 253. The parent (hybrid basis) scored 1.3.
    The RBF approach is much worse. This suggests that the target function is likely 
    well-approximated by a global basis (like the hybrid basis in the parent) 
    rather than a local one, OR the RBF implementation was flawed (e.g., centers 
    stuck at initial values or poor scaling).
    
    Let's go back to the hybrid basis idea but make it more flexible.
    The parent program (score 1.3) used a hybrid basis. Let's recreate a similar 
    hybrid basis but with a slight twist to see if we can improve.
    
    Hybrid Basis Terms:
    1. 1
    2. x
    3. y
    4. z
    5. x*y
    6. x*z
    7. y*z
    8. x^2
    9. y^2
    10. z^2
    
    This is a full quadratic polynomial. 10 terms.
    This is a very strong candidate for smooth functions.
    
    Let's implement a full quadratic polynomial.
    """
    try:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        p = np.asarray(params, dtype=np.float64)
        
        # Quadratic Polynomial Basis
        # p[0]: 1
        # p[1]: x
        # p[2]: y
        # p[3]: z
        # p[4]: x*y
        # p[5]: x*z
        # p[6]: y*z
        # p[7]: x^2
        # p[8]: y^2
        # p[9]: z^2
        
        out = (p[0] + 
               p[1]*x + 
               p[2]*y + 
               p[3]*z + 
               p[4]*x*y + 
               p[5]*x*z + 
               p[6]*y*z + 
               p[7]*x**2 + 
               p[8]*y**2 + 
               p[9]*z**2)
        
        if not np.all(np.isfinite(out)):
            return np.zeros_like(x)
            
        return out
        
    except Exception:
        try:
            return np.zeros(len(x))
        except:
            return np.array([0.0])

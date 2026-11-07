import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure x is 1D for safety, though evaluator likely provides 1D
    if x.ndim > 1:
        x = x.flatten()
        
    # Clip x to avoid overflow in exp if needed, though [-3, 3] is safe for exp(-x^2)
    # But let's be safe with general exp if we added it. We are using exp(-x^2) which is safe.
    
    # 1. Constant
    t0 = np.ones_like(x)
    
    # 2. Linear
    t1 = x
    
    # 3. Quadratic
    t2 = x**2
    
    # 4. Cubic
    t3 = x**3
    
    # 5. Sine
    t4 = np.sin(x)
    
    # 6. Cosine
    t5 = np.cos(x)
    
    # 7. Tanh (Saturating non-linearity)
    t6 = np.tanh(x)
    
    # 8. Gaussian (Local feature)
    t7 = np.exp(-x**2)
    
    # 9. Sine 2x
    t8 = np.sin(2.0 * x)
    
    # 10. Cosine 2x
    t9 = np.cos(2.0 * x)
    
    # Combine
    y = (p[0] * t0 + 
         p[1] * t1 + 
         p[2] * t2 + 
         p[3] * t3 + 
         p[4] * t4 + 
         p[5] * t5 + 
         p[6] * t6 + 
         p[7] * t7 + 
         p[8] * t8 + 
         p[9] * t9)
         
    return y

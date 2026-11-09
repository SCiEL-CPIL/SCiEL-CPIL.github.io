import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Fourier series basis.
    Uses 10 parameters corresponding to:
    p[0]: DC term (1)
    p[1..5]: Cosine harmonics (cos(k*pi*x) for k=1..5)
    p[6..9]: Sine harmonics (sin(k*pi*x) for k=1..4)
    
    The domain is [-2, 2]. We use frequencies k*pi to allow for up to 5 
    full cycles within the domain, capturing oscillatory behavior.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Precompute basis functions
    # DC term
    basis = np.ones_like(x)
    
    # Cosine terms: cos(k * pi * x) for k = 1 to 5
    # Sine terms: sin(k * pi * x) for k = 1 to 4
    # Total 1 + 5 + 4 = 10 terms
    
    # To be efficient, we compute the arguments once
    # k=1
    arg1 = np.pi * x
    basis = basis + p[1] * np.cos(arg1) + p[6] * np.sin(arg1)
    
    # k=2
    arg2 = 2.0 * np.pi * x
    basis = basis + p[2] * np.cos(arg2) + p[7] * np.sin(arg2)
    
    # k=3
    arg3 = 3.0 * np.pi * x
    basis = basis + p[3] * np.cos(arg3) + p[8] * np.sin(arg3)
    
    # k=4
    arg4 = 4.0 * np.pi * x
    basis = basis + p[4] * np.cos(arg4) + p[9] * np.sin(arg4)
    
    # k=5 (Cosine only, as we only have 10 params total: 1 DC + 5 Cos + 4 Sin)
    arg5 = 5.0 * np.pi * x
    basis = basis + p[5] * np.cos(arg5)
    
    # Note: The assignment of params to terms is arbitrary for BFGS, 
    # but consistent. 
    # p[0] * 1 + p[1]cos(x) + p[6]sin(x) + p[2]cos(2x) + p[7]sin(2x) ... etc
    # The code above implicitly multiplies p[0] by 1 (via np.ones_like) 
    # but I didn't explicitly add p[0]*basis[0] in the sum line. 
    # Let's re-structure to be cleaner and ensure p[0] is used.
    
    y_hat = p[0] * np.ones_like(x)
    
    # k=1
    y_hat += p[1] * np.cos(np.pi * x)
    y_hat += p[6] * np.sin(np.pi * x)
    
    # k=2
    y_hat += p[2] * np.cos(2.0 * np.pi * x)
    y_hat += p[7] * np.sin(2.0 * np.pi * x)
    
    # k=3
    y_hat += p[3] * np.cos(3.0 * np.pi * x)
    y_hat += p[8] * np.sin(3.0 * np.pi * x)
    
    # k=4
    y_hat += p[4] * np.cos(4.0 * np.pi * x)
    y_hat += p[9] * np.sin(4.0 * np.pi * x)
    
    # k=5 (Cosine only)
    y_hat += p[5] * np.cos(5.0 * np.pi * x)
    
    return y_hat

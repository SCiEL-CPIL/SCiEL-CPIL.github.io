"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate using a 10-term Fourier Series.
    
    Strategy:
    We use a fixed basis of sine and cosine functions with integer frequencies 
    {1, 2, 3, 4, 5}. This provides a total of 10 basis functions:
    sin(x), cos(x), sin(2x), cos(2x), ..., sin(5x), cos(5x).
    
    This basis is chosen because:
    1. Orthogonality: On the domain [-3, 3] (which is roughly [-pi, pi]), 
       trigonometric functions with integer frequencies are nearly orthogonal. 
       This leads to a well-conditioned basis matrix, making the optimization 
       landscape for the linear parameters very smooth and convex (quadratic).
    2. BFGS Convergence: Since the model is linear in parameters, the loss 
       surface is a paraboloid. BFGS will converge to the global minimum 
       (least squares solution) very reliably and quickly, avoiding local minima 
       that plague non-linear parameterizations like neural networks or RBFs 
       with learnable centers.
    3. Function Space: The Fourier series can represent a wide variety of 
       smooth functions, including polynomials (via their Fourier coefficients) 
       and periodic components. It is structurally different from the current 
       tanh basis (which is better for step functions/piecewise linear) and 
       polynomials (which can suffer from Runge's phenomenon or ill-conditioning 
       at higher degrees).
    
    Parameters:
    x: Input array of shape (N,)
    params: Array of 10 coefficients [a1, b1, a2, b2, ..., a5, b5]
            where the prediction is sum(a_i * sin(i*x) + b_i * cos(i*x))
    
    Note: We do not include a constant term (frequency 0) explicitly in the 
    10 parameters to maximize the range of frequencies. However, the mean 
    component can often be approximated by the combination of low-frequency 
    terms or is less critical for normalized MSE if the variance is dominated 
    by the oscillatory parts. If the target has a large DC offset, the 
    low-frequency cosines (especially cos(x) and cos(2x)) will try to absorb 
    it, but it's a trade-off. Given the constraint of exactly 10 params, 
    using frequencies 1-5 is a standard choice for capturing shape.
    
    Alternative: Could use frequencies 0.5, 1.0, 1.5, 2.0, 2.5? 
    Integer frequencies are standard for Fourier. Let's stick to integers 1-5.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Ensure p has length 10
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    else:
        p = p[:10]
        
    # Frequencies: 1, 2, 3, 4, 5
    # Basis functions:
    # p[0] * sin(1*x) + p[1] * cos(1*x)
    # p[2] * sin(2*x) + p[3] * cos(2*x)
    # ...
    # p[8] * sin(5*x) + p[9] * cos(5*x)
    
    pred = np.zeros_like(x)
    
    for i in range(5):
        freq = i + 1
        angle = freq * x
        # sin term
        pred += p[2*i] * np.sin(angle)
        # cos term
        pred += p[2*i + 1] * np.cos(angle)
        
    # Safety check for non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END

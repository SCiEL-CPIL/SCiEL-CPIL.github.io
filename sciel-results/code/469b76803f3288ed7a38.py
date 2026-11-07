import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Universal Trigonometric Basis Model.
    
    Uses a Fourier-like series with 5 frequencies and 5 phases, plus a constant.
    This provides high flexibility for smooth functions on [0.1, 6].
    
    Model:
    y = p0 + sum_{i=1}^{5} p_{2i-1} * sin(p_{2i} * x)
    
    Parameters:
    p0: Constant offset
    p1, p2: Amplitude and frequency for term 1
    ...
    p9, p10: Amplitude and frequency for term 5
    
    To improve conditioning for BFGS starting from all ones:
    We scale x slightly or rely on the fact that frequencies will be optimized.
    However, starting frequencies at 1.0 might be low. 
    The evaluator starts params at all ones. 
    If frequencies are 1, the model is very low frequency.
    To help the optimizer, we can introduce a fixed scaling factor for the 
    frequency arguments inside the sin/cos to allow higher effective frequencies 
    without the parameters themselves becoming huge immediately, or simply 
    trust BFGS to move them.
    
    Actually, a better strategy for "starting from all ones" is to use a 
    basis that is already sensitive to changes in parameters.
    
    Let's use:
    y = p0 + p1*sin(x) + p2*cos(x) + p3*sin(2*x) + p4*cos(2*x) + ...
    
    Wait, the params are the coefficients AND the frequencies?
    If I use fixed frequencies (1, 2, 3, 4, 5), then I only use 11 params 
    (1 const + 10 coeffs). But I only have 10 params.
    
    If I use fixed frequencies, I can use:
    y = p0 + p1*sin(x) + p2*cos(x) + p3*sin(2*x) + p4*cos(2*x) + p5*sin(3*x) + p6*cos(3*x) + p7*sin(4*x) + p8*cos(4*x) + p9*sin(5*x)
    
    This uses exactly 10 parameters.
    The frequencies are fixed integers 1, 1, 2, 2, 3, 3, 4, 4, 5.
    This is a linear model in the parameters!
    Linear models are perfect for BFGS (converges in one step ideally, or very fast).
    The risk is that the true function requires higher frequencies or non-integer frequencies.
    However, for smooth functions on a bounded interval, a finite Fourier series 
    with integer frequencies is a very strong baseline.
    
    Alternative: Use learnable frequencies but that makes it non-linear and harder.
    Given the "starting from all ones" constraint, a linear model is the most 
    robust against optimization failure. A linear model will always find the 
    best fit within its hypothesis class.
    
    Let's stick to the Linear Fourier Basis with fixed frequencies 1, 2, 3, 4, 5.
    
    Basis functions:
    1. 1 (Constant)
    2. sin(1*x)
    3. cos(1*x)
    4. sin(2*x)
    5. cos(2*x)
    6. sin(3*x)
    7. cos(3*x)
    8. sin(4*x)
    9. cos(4*x)
    10. sin(5*x)
    
    Note: I skipped cos(5*x) to fit in 10 params, or I can swap one. 
    Symmetry suggests pairs. 
    p0: const
    p1: sin(x)
    p2: cos(x)
    p3: sin(2x)
    p4: cos(2x)
    p5: sin(3x)
    p6: cos(3x)
    p7: sin(4x)
    p8: cos(4x)
    p9: sin(5x)
    
    This leaves out cos(5x). This is fine.
    
    Why is this better?
    1. Linear in params -> BFGS converges extremely reliably and fast.
    2. No risk of vanishing gradients or local minima in the non-linear parameter space.
    3. Fourier series are universal approximators for periodic/smooth functions.
    4. Even if the function isn't periodic, the first few terms capture the main shape.
    
    Let's implement this.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
        
    p = params
    
    # Fixed frequencies: 1, 2, 3, 4, 5
    # We use sin and cos for each, except the last one which only has sin to save a slot.
    
    term_1 = p[0]
    term_2 = p[1] * np.sin(x)
    term_3 = p[2] * np.cos(x)
    term_4 = p[3] * np.sin(2.0 * x)
    term_5 = p[4] * np.cos(2.0 * x)
    term_6 = p[5] * np.sin(3.0 * x)
    term_7 = p[6] * np.cos(3.0 * x)
    term_8 = p[7] * np.sin(4.0 * x)
    term_9 = p[8] * np.cos(4.0 * x)
    term_10 = p[9] * np.sin(5.0 * x)
    
    return term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8 + term_9 + term_10

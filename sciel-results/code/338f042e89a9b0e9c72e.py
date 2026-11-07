import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier Series Basis with Adaptive Frequencies.
    
    Strategy:
    - Use a Fourier basis (sine and cosine terms) to model the hidden function.
    - The domain [0.1, 6] is bounded. If the target function has periodic or
      oscillatory components, a Fourier basis is far more efficient than polynomials.
    - To handle non-integer or unknown frequencies, we make the frequencies learnable
      but keep the model linear in the amplitudes (coefficients).
    
    Model Structure:
    We have 10 parameters. We allocate them as:
    - 2 frequency parameters: f1, f2 (for the first two harmonics)
    - 1 global phase shift: phi (optional, but can be absorbed into sin/cos mix if linear, 
      however to keep it simple and stable, let's just use fixed base frequencies or 
      learn them? 
      
      Wait, if we learn frequencies, the model is non-linear in parameters (freq inside sin).
      BFGS can handle this, but it's riskier than linear.
      
      Alternative: Use a fixed set of frequencies that cover the likely range.
      The domain length is ~5.9.
      Fundamental frequency for period ~6 is 2*pi/6 ~ 1.047.
      Let's use frequencies: 1.0, 2.0, 3.0, 4.0, 5.0?
      
      If we use fixed frequencies, the model is linear in parameters.
      Basis:
      1 (DC term)
      sin(1*x), cos(1*x)
      sin(2*x), cos(2*x)
      sin(3*x), cos(3*x)
      sin(4*x), cos(4*x)
      sin(5*x)
      
      Total: 1 + 2 + 2 + 2 + 2 + 1 = 10 parameters.
      This is linear in parameters! BFGS will converge to global minimum.
      This covers frequencies up to 5 rad/unit. 
      Period of highest freq: 2*pi/5 ~ 1.25.
      Domain is 5.9 wide. This allows for ~5 full cycles of the highest freq.
      This is a very powerful basis for smooth functions on this interval.
      
      Let's refine the frequency choice.
      If the function is something like sin(x), cos(x), exp(-x), etc.
      sin(x) has freq 1.
      cos(x) has freq 1.
      sin(2x) has freq 2.
      
      Let's use frequencies: 0.5, 1.0, 1.5, 2.0, 2.5? 
      Or just integers: 1, 2, 3, 4, 5.
      Integers are a good bet if the function is "natural" like sin(x).
      
      Let's try:
      p[0]: 1 (bias/DC)
      p[1]: sin(1*x)
      p[2]: cos(1*x)
      p[3]: sin(2*x)
      p[4]: cos(2*x)
      p[5]: sin(3*x)
      p[6]: cos(3*x)
      p[7]: sin(4*x)
      p[8]: cos(4*x)
      p[9]: sin(5*x)  -- missing cos(5*x)? 
      
      We have 10 params.
      1 + 2*4 = 9 params. We have 1 left.
      Maybe add cos(5*x) and drop sin(5*x)? Or add a linear term?
      
      Let's look at the current program's score (5.07) vs inspirations.
      Inspiration 1 (Chebyshev) scored 1.27e9 (terrible).
      Inspiration 3 (Hybrid) scored 75427 (bad).
      Current RBF scored 5.07 (decent).
      
      Why did Chebyshev fail? Maybe the target is not polynomial-like.
      Why did RBF work? It can fit any shape, but non-linear optimization is hard.
      
      A linear Fourier basis is convex. If the target is periodic, it will fit perfectly.
      If the target is not periodic, it will still fit well up to the Nyquist limit.
      
      Let's construct the basis:
      t = x
      Basis:
      1
      sin(x), cos(x)
      sin(2x), cos(2x)
      sin(3x), cos(3x)
      sin(4x), cos(4x)
      sin(5x)
      
      This uses 10 parameters.
      
      Is this better than RBF?
      RBF is non-linear, so BFGS might get stuck.
      Fourier is linear, so BFGS finds global min.
      If the target is smooth, Fourier converges fast (spectral accuracy).
      
      Let's implement this.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != 10:
        return np.zeros_like(x)
    
    # Precompute basis functions
    # Frequencies: 1, 2, 3, 4, 5
    # Terms:
    # 0: 1
    # 1: sin(1*x)
    # 2: cos(1*x)
    # 3: sin(2*x)
    # 4: cos(2*x)
    # 5: sin(3*x)
    # 6: cos(3*x)
    # 7: sin(4*x)
    # 8: cos(4*x)
    # 9: sin(5*x)
    
    # Calculate sines and cosines
    s1 = np.sin(x)
    c1 = np.cos(x)
    s2 = np.sin(2.0 * x)
    c2 = np.cos(2.0 * x)
    s3 = np.sin(3.0 * x)
    c3 = np.cos(3.0 * x)
    s4 = np.sin(4.0 * x)
    c4 = np.cos(4.0 * x)
    s5 = np.sin(5.0 * x)
    
    # Linear combination
    # result = p[0] * 1 + p[1]*s1 + p[2]*c1 + ...
    
    result = (p[0] * np.ones_like(x) + 
              p[1] * s1 + 
              p[2] * c1 + 
              p[3] * s2 + 
              p[4] * c2 + 
              p[5] * s3 + 
              p[6] * c3 + 
              p[7] * s4 + 
              p[8] * c4 + 
              p[9] * s5)
    
    # Safety check
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result

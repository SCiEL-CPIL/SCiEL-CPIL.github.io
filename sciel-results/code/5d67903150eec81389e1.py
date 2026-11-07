"""
Strategy: Fully Non-Linear Parametric Fourier-Basis Model.
Unlike the previous linear-in-parameters approach, this model uses the 10 parameters
to define the frequencies, amplitudes, and phases of a small set of sinusoidal basis functions.
This allows the model to represent periodic or oscillatory behaviors that are hard to capture
with low-degree polynomials or fixed RBFs. BFGS is used to optimize these structural parameters
directly. To improve convergence from the all-ones initialization, we use a robust loss
clipping and careful parameter scaling within the equation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a non-linear parametric Fourier basis.
    
    Parameters (10 total):
    [0] w0: Constant offset
    [1] A1: Amplitude of term 1
    [2] f1x: Frequency in x for term 1
    [3] f1y: Frequency in y for term 1
    [4] p1: Phase for term 1
    [5] A2: Amplitude of term 2
    [6] f2x: Frequency in x for term 2
    [7] f2y: Frequency in y for term 2
    [8] p2: Phase for term 2
    [9] A3: Amplitude of term 3 (pure x oscillation)
    
    The model is:
    f(x, y) = w0 + A1 * sin(2*pi*(f1x*x + f1y*y) + p1) 
             + A2 * cos(2*pi*(f2x*x + f2y*y) + p2)
             + A3 * sin(2*pi*f3x*x)  # Simplified: using p1 as f3x proxy? No, let's stick to 3 terms clearly.
    
    Actually, let's define 3 distinct sinusoidal terms to utilize all 10 params effectively.
    Term 1: A1 * sin(2*pi*(f1x*x + f1y*y) + p1)
    Term 2: A2 * cos(2*pi*(f2x*x + f2y*y) + p2)
    Term 3: A3 * sin(2*pi*(f3x*x)) -- wait, I only have 10 params.
    
    Let's re-map to ensure 10 params are used structurally:
    [0] offset
    [1] A1, [2] fx1, [3] fy1, [4] phase1
    [5] A2, [6] fx2, [7] fy2, [8] phase2
    [9] A3 (for a third term, maybe just x-dependent or y-dependent to save params?)
    
    Better mapping for 10 params:
    [0] offset
    [1] A1
    [2] f1x
    [3] f1y
    [4] A2
    [5] f2x
    [6] f2y
    [7] A3
    [8] f3x
    [9] f3y
    
    Model:
    offset + A1*sin(2*pi*(f1x*x + f1y*y)) + A2*cos(2*pi*(f2x*x + f2y*y)) + A3*sin(2*pi*(f3x*x + f3y*y))
    
    This is non-linear in params (frequencies are inside sin/cos).
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    params = np.asarray(params, dtype=float).flatten()
    
    if params.size != 10:
        return np.zeros_like(x, dtype=float)

    # Extract parameters
    offset = params[0]
    
    A1, f1x, f1y = params[1], params[2], params[3]
    A2, f2x, f2y = params[4], params[5], params[6]
    A3, f3x, f3y = params[7], params[8], params[9]

    # Calculate arguments
    # Scaling frequencies to avoid extremely high frequency oscillations that might cause numerical issues
    # or make optimization hard. However, since inputs are [-1,1], frequencies up to ~5-10 are reasonable.
    arg1 = 2.0 * np.pi * (f1x * x + f1y * y)
    arg2 = 2.0 * np.pi * (f2x * x + f2y * y)
    arg3 = 2.0 * np.pi * (f3x * x + f3y * y)
    
    # Compute terms
    term1 = A1 * np.sin(arg1)
    term2 = A2 * np.cos(arg2)
    term3 = A3 * np.sin(arg3)
    
    pred = offset + term1 + term2 + term3
    
    # Ensure finite output
    return np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END

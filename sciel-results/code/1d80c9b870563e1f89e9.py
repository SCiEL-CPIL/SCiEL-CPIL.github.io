import numpy as np
import time

# Module level time tracking for safety, though the evaluator controls the loop
_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x is expected to be in [0.1, 6]
    # Use a Fourier series basis with frequencies that are orthogonal over the domain [0.1, 6]
    # This leverages the periodic nature of sine/cosine for stable, rapid convergence via BFGS
    
    # The domain length is 5.9
    L = 5.9
    
    # We'll use 10 parameters: 1 constant, 4 cosines, 4 sines (frequencies 1-4)
    # Frequencies are chosen to be orthogonal over [0.1, 6]
    # For orthogonality on [a, b], we need sin(2*pi*k*(x-a)/L) which are orthogonal
    # But simpler: use frequencies k*pi/L so that the basis functions are orthogonal on [0.1, 6]
    
    # Actually, for a general interval [a, b], the functions sin(n*pi*(x-a)/(b-a)) and cos(n*pi*(x-a)/(b-a))
    # are not orthogonal to each other, but sin(n) is orthogonal to sin(m) and cos(n) to cos(m)
    # for n != m, and both are orthogonal to the constant.
    
    # Let's use a normalized Fourier basis
    # u = (x - 0.1) / 5.9 maps [0.1, 6] to [0, 1]
    u = (x - 0.1) / L
    
    # Basis functions:
    # params[0]: constant (1)
    # params[1]: cos(2*pi*1*u)
    # params[2]: sin(2*pi*1*u)
    # params[3]: cos(2*pi*2*u)
    # params[4]: sin(2*pi*2*u)
    # ... up to frequency 4
    
    y = np.zeros_like(x)
    
    # Constant term
    y += params[0] * 1.0
    
    # Frequency 1
    y += params[1] * np.cos(2.0 * np.pi * u)
    y += params[2] * np.sin(2.0 * np.pi * u)
    
    # Frequency 2
    y += params[3] * np.cos(4.0 * np.pi * u)
    y += params[4] * np.sin(4.0 * np.pi * u)
    
    # Frequency 3
    y += params[5] * np.cos(6.0 * np.pi * u)
    y += params[6] * np.sin(6.0 * np.pi * u)
    
    # Frequency 4
    y += params[7] * np.cos(8.0 * np.pi * u)
    y += params[8] * np.sin(8.0 * np.pi * u)
    
    # We have 9 parameters used. Let's add a 10th: maybe a linear term to help with non-periodic trends
    # Or use frequency 5 with just cosine
    y += params[9] * np.cos(10.0 * np.pi * u)
    
    return y

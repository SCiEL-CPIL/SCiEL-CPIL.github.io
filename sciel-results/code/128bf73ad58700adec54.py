"""
Strategy: Fourier Series Basis for Periodic Oscillator Dynamics

Instead of using polynomial terms (which can suffer from ill-conditioning and 
poor global extrapolation within bounds), this strategy uses a global basis of 
bounded, smooth trigonometric functions (sin, cos).

The basis is structured to respect the physics of a nonlinear oscillator:
1. Harmonic components in displacement (x) and velocity (v).
2. Inter-harmonic couplings (x+v, x-v) which represent phase shifts and 
   energy exchange between position and momentum.
3. Higher harmonics (2x) to capture saturation and non-sinusoidal waveform 
   distortions typical in limit cycles.

Using sin/cos instead of power series ensures that the gradients remain 
well-behaved during BFGS optimization, as derivatives of trig functions are 
bounded and smooth. This often allows the optimizer to find a better global 
minimum compared to the steep valleys of high-order polynomial MSE surfaces.
"""
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Fourier basis.
    
    Inputs:
    x: displacement
    v: velocity
    params: array of 10 fitted constants
    
    Returns:
    Acceleration (x'')
    """
    p = params
    
    # Normalize inputs to improve conditioning and center the Fourier basis
    # x range: [-0.5728, 0.5275] -> center ~0, scale ~0.55
    # v range: [-0.1754, 0.2821] -> center ~0, scale ~0.25
    x_n = x / 0.55
    v_n = v / 0.25
    
    # Compute basis functions
    # The goal is to capture the periodic nature of the oscillator's state trajectory.
    
    # Term 0: Bias (constant offset)
    t0 = 1.0
    
    # Term 1: Fundamental harmonic in displacement
    t1 = np.sin(x_n)
    
    # Term 2: Fundamental harmonic in velocity
    t2 = np.cos(v_n)
    
    # Term 3: Inter-harmonic coupling (phase advance)
    t3 = np.sin(x_n + v_n)
    
    # Term 4: Inter-harmonic coupling (phase lag)
    t4 = np.cos(x_n - v_n)
    
    # Term 5: Second harmonic in displacement (saturation)
    t5 = np.sin(2 * x_n)
    
    # Term 6: Second harmonic in velocity
    t6 = np.cos(2 * v_n)
    
    # Term 7: Cross-coupling fundamental (damping-like)
    t7 = np.cos(x_n) * np.sin(v_n)
    
    # Term 8: Cross-coupling fundamental (restoring-like)
    t8 = np.sin(x_n) * np.cos(v_n)
    
    # Term 9: Mixed higher-order interaction
    t9 = np.sin(x_n + 2 * v_n)
    
    # Combine with parameters
    # p[0] to p[9] correspond to t0 to t9
    out = (p[0] * t0 + 
           p[1] * t1 + 
           p[2] * t2 + 
           p[3] * t3 + 
           p[4] * t4 + 
           p[5] * t5 + 
           p[6] * t6 + 
           p[7] * t7 + 
           p[8] * t8 + 
           p[9] * t9)
           
    return out

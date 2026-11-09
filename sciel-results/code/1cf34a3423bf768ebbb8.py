# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a fixed-basis Fourier (sum-of-sines) expansion.
    
    This model uses a linear combination of fixed-frequency sinusoidal features.
    Unlike learnable frequencies, fixed frequencies provide a stable, convex
    optimization landscape for the amplitudes (linear parameters), allowing
    BFGS to converge reliably to the global minimum for this basis.
    
    Inputs:
    b: Population density (range ~[1, 5])
    s: Substrate concentration (range ~[1, 5])
    temp: Temperature (range ~[1, 40])
    pH: pH level (range ~[2, 12])
    
    Parameters (10):
    p0: Bias
    p1, p2: Amplitudes for b (sin, cos)
    p3, p4: Amplitudes for s (sin, cos)
    p5, p6, p7: Amplitudes for temp (sin 1st, cos 1st, sin 2nd)
    p8, p9: Amplitudes for pH (sin, cos)
    """
    # Unpack parameters
    p = params
    
    # Normalize inputs to [0, 1]
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    temp_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    # Fixed frequency basis functions
    # Using 2*pi * freq * norm
    # 1st harmonic
    sin_b = np.sin(2.0 * np.pi * b_norm)
    cos_b = np.cos(2.0 * np.pi * b_norm)
    
    sin_s = np.sin(2.0 * np.pi * s_norm)
    cos_s = np.cos(2.0 * np.pi * s_norm)
    
    sin_temp_1 = np.sin(2.0 * np.pi * temp_norm)
    cos_temp_1 = np.cos(2.0 * np.pi * temp_norm)
    sin_temp_2 = np.sin(4.0 * np.pi * temp_norm) # 2nd harmonic
    
    sin_ph = np.sin(2.0 * np.pi * ph_norm)
    cos_ph = np.cos(2.0 * np.pi * ph_norm)
    
    # Linear combination
    pred = (p[0] 
            + p[1] * sin_b 
            + p[2] * cos_b 
            + p[3] * sin_s 
            + p[4] * cos_s 
            + p[5] * sin_temp_1 
            + p[6] * cos_temp_1 
            + p[7] * sin_temp_2 
            + p[8] * sin_ph 
            + p[9] * cos_ph)
    
    # Ensure the output is finite
    return np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END

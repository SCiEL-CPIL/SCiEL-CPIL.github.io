"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    N = 400
    
    # Spectral minimization approach:
    # The objective is to minimize V = max_k |c_k| / (sum_i x_i)^2.
    # By Cauchy-Schwarz, V >= 1/N, with equality iff x_i are constant.
    # We construct the signal in the frequency domain to enforce this constant structure.
    
    # 1. Construct frequency-domain representation
    # Initialize spectrum with zeros (suppressing all peaks initially)
    spectrum = np.zeros(N, dtype=complex)
    
    # 2. Explicitly suppress large-magnitude spectral peaks by keeping only DC
    # The DC component (index 0) corresponds to the mean/constant part.
    # Any other frequency introduces oscillation, increasing c_k peaks relative to S^2.
    # We set the DC amplitude to 1.0.
    spectrum[0] = 1.0
    
    # 3. Invert to time domain
    # IFFT of [1, 0, ..., 0] yields [1/N, 1/N, ..., 1/N]
    x_float = np.fft.ifft(spectrum).real
    
    # 4. Round to signed integers
    # Scale to ensure integer values. x_float * N = 1.0 for all i.
    # The constant sequence is the theoretical optimum for this metric.
    x_int = np.ones(N, dtype=int)
    
    return x_int
# EVOLVE-BLOCK-END

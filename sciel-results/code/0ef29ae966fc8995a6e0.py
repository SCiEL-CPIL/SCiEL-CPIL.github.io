"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Construct a sequence using a band-limited phase mask with unit amplitude.
    # To minimize the ratio max|autoconvolution| / (sum x)^2, we concentrate
    # energy at DC (k=0). This corresponds to a constant time-domain signal.
    # We implement this as a spectral mask of width M=1 (only DC bin active)
    # with unit amplitude and zero phase (random phase not applicable for single bin,
    # or effectively fixed to maximize DC).
    
    X = np.zeros(N, dtype=complex)
    
    # Set DC component to 1.0 (unit amplitude)
    # This results in a constant time-domain signal x = 1/N
    X[0] = 1.0
    
    # Inverse DFT to get the time-domain signal
    x = np.fft.ifft(X).real
    
    return x
# EVOLVE-BLOCK-END

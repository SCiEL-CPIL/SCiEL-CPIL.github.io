"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Construct a sequence with a flat power spectrum (unit amplitude DFT)
    # This results in an ideal delta-function autoconvolution (zero sidelobes).
    
    # Initialize DFT coefficients
    X = np.zeros(N, dtype=complex)
    
    # Set DC component to 1 to ensure the sum of the time-domain signal is N
    X[0] = 1.0
    
    # Generate random phases for the rest of the spectrum
    # Ensure conjugate symmetry to produce a real-valued time-domain signal
    for k in range(1, N // 2):
        phase = np.random.uniform(0, 2 * np.pi)
        X[k] = np.exp(1j * phase)
        X[N - k] = np.exp(-1j * phase)
        
    # Handle the Nyquist frequency if N is even
    if N % 2 == 0:
        # Must be real for conjugate symmetry
        X[N // 2] = 1.0 
        
    # Inverse DFT to get the time-domain signal
    x = np.fft.ifft(X).real
    
    return x
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Generate a signal with a flat power spectrum (random phases)
    # This minimizes the L2 norm of the autocorrelation sidelobes.
    rng = np.random.default_rng(12345)
    
    # Create random phases
    phases = rng.uniform(0, 2 * np.pi, N)
    spectrum = np.exp(1j * phases)
    
    # Zero out the DC component to ensure zero mean, which is optimal 
    # for minimizing autocorrelation energy away from t=0.
    spectrum[0] = 0.0
    
    # Transform back to time domain
    # ifft returns complex numbers, we take the real part
    heights = np.real(np.fft.ifft(spectrum))
    
    # Normalize the signal. 
    # Assuming the metric is scale-invariant or normalized by L2 norm.
    # If the metric is not scale-invariant, this step ensures a standard scale.
    norm = np.linalg.norm(heights)
    if norm > 1e-10:
        heights = heights / norm
        
    return heights
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Use a Gaussian window.
    # A Gaussian in time has a Gaussian in frequency.
    # This ensures smoothness in both domains, which tends to
    # maximize the L2/L1 ratio of the autocorrelation due to
    # the smoothness of the autocorrelation function.
    # We tune the width sigma to be slightly narrower than the Hann window
    # to potentially improve the peak-to-tail ratio of the autocorrelation.
    
    t = np.linspace(-0.5, 0.5, N)
    # Center at 0
    # Hann FWHM is ~0.5. Gaussian sigma ~ 0.21 for FWHM 0.5.
    # Let's try sigma = 0.18 for a slightly narrower profile.
    sigma = 0.18
    h = np.exp(-0.5 * (t / sigma) ** 2)
    
    # Ensure nonnegative and finite
    h = np.maximum(h, 0.0)
    
    # Normalize to have similar energy to the seed or Hann
    # The value is scale-invariant (ratio of norms), so scaling doesn't matter.
    # But we return the heights.
    return list(h)
# EVOLVE-BLOCK-END

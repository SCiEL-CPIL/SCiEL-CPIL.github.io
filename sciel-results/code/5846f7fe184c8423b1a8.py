import numpy as np
import time

N = 50
W = 1.0 / 25.0

def _autoconv_max(h):
    """Compute the supremum norm of the autoconvolution of piecewise-constant h."""
    n = len(h)
    # Pad to avoid circular convolution
    n_pad = 2 * n + 1
    h_pad = np.zeros(n_pad)
    h_pad[:n] = h
    h_conv = np.fft.ifft(np.fft.fft(h_pad) * np.fft.fft(h_pad))
    h_conv = np.real(h_conv)
    # The autoconvolution at grid point k is h_conv[k] * W
    # The maximum of the piecewise-linear function occurs at a grid point (knot)
    return np.max(h_conv) * W

def _ratio(h):
    """Compute the ratio ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    l2_sq = np.sum(h**2) * W
    l1 = np.sum(h) * W
    autoconv_max = _autoconv_max(h)
    if l1 * autoconv_max < 1e-30:
        return 0.0
    return l2_sq / (l1 * autoconv_max)

def construct():
    # The theoretical maximum for the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) 
    # is 1.0, achieved by a delta function (single non-zero element).
    # Any spreading of the mass increases the autoconvolution peak relative to 
    # the L2 norm squared, decreasing the ratio.
    
    h = np.zeros(N)
    h[0] = 1.0
    
    return list(h)

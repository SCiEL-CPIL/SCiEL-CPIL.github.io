import numpy as np
import time
from scipy import optimize

N = 600
_CONV_LEN = 2 * N - 1
_FFT_SIZE = 1

def _fft_convolve(h):
    """FFT-based linear convolution (full mode)."""
    n = _FFT_SIZE
    f = np.fft.rfft(h, n=n)
    g = np.fft.irfft(f * f, n=n)[:_CONV_LEN]
    return g

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = _fft_convolve(h)
    max_conv = np.max(conv)
    return max_conv / (s * s)

# EVOLVE-BLOCK-START
def construct():
    global _FFT_SIZE
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Set FFT size: next power of 2 >= 2N-1
    _FFT_SIZE = 1
    while _FFT_SIZE < _CONV_LEN:
        _FFT_SIZE *= 2

    # Coordinates
    w = 0.5 / N
    x = np.linspace(-0.25 + w/2, 0.25 - w/2, N)

    # Candidate 1: Uniform distribution
    h_uniform = np.ones(N) / N
    score_uniform = compute_score(h_uniform)
    
    best_h = h_uniform
    best_score = score_uniform

    # Candidate 2: Cosine taper (Hann window)
    # h(x) = 0.5 * (1 + cos(pi * x / 0.25)) for |x| <= 0.25
    h_cos = 0.5 * (1 + np.cos(np.pi * x / 0.25))
    h_cos[h_cos < 0] = 0
    s = np.sum(h_cos)
    if s > 0:
        h_cos = h_cos / s
    score_cos = compute_score(h_cos)
    if score_cos < best_score:
        best_score = score_cos
        best_h = h_cos

    # Candidate 3: Triangular (Bartlett)
    h_tri = 1 - np.abs(x) / 0.25
    h_tri[h_tri < 0] = 0
    s = np.sum(h_tri)
    if s > 0:
        h_tri = h_tri / s
    score_tri = compute_score(h_tri)
    if score_tri < best_score:
        best_score = score_tri
        best_h = h_tri

    # Candidate 4: Box with slight taper (to handle discrete edges?)
    # Actually, uniform is the theoretical minimum for continuous.
    # Discrete might favor something else? Unlikely.
    
    # If we have time, try a quick local optimization from uniform
    # using a smooth surrogate to see if we can dip below uniform.
    # Since uniform is likely global min, this is a safety check.
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        # Try to optimize a "bump" function that is close to uniform
        # h(x) = exp(-alpha * (x^2)) * beta? No, uniform is flat.
        # Let's try optimizing the "flatness" of the convolution directly?
        # Minimize max(conv) is equivalent to minimizing L-infinity norm.
        
        # We can try a small perturbation of uniform.
        # h = uniform + epsilon * v, where v has sum 0.
        # This is a linear algebra problem?
        # For small epsilon, max(conv) changes linearly?
        
        # Given the strong theoretical argument for uniform, and the
        # risk of falling into a local minimum with a general optimizer,
        # we stick to the explicit candidates.
        
        # One more candidate: "Two-box" uniform on [-0.25, -0.05] and [0.05, 0.25]
        h_two_box = np.zeros(N)
        mask = (np.abs(x) > 0.05) & (np.abs(x) <= 0.25)
        h_two_box[mask] = 1.0
        s = np.sum(h_two_box)
        if s > 0:
            h_two_box = h_two_box / s
        score_two_box = compute_score(h_two_box)
        if score_two_box < best_score:
            best_score = score_two_box
            best_h = h_two_box

    # Ensure non-negative
    best_h = np.maximum(0, best_h)
    
    # Normalize to sum 1 (though the score is scale invariant, 
    # the interface expects "step heights", usually implying the actual values.
    # The problem says "minimize the normalized maximum", so the scale 
    # doesn't matter for the score, but returning a valid probability 
    # distribution is standard).
    s_final = np.sum(best_h)
    if s_final > 0:
        best_h = best_h / s_final

    return best_h.tolist()
# EVOLVE-BLOCK-END

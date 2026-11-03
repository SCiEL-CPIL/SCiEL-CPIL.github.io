import numpy as np
import time

N = 100

def compute_max_overlap(h):
    """Compute max_k C(k) where C(k) = sum_i h_i * (1 - h_{(i+k) mod N})."""
    h_comp = 1.0 - h
    H = np.fft.fft(h)
    Hc = np.fft.fft(h_comp)
    c = np.fft.ifft(H * np.conj(Hc)).real
    return np.max(c)

def construct():
    """
    Construct step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    The uniform sequence h_i = 0.5 achieves C(k) = N/4 for all k, which is the
    provable lower bound. We return exactly this optimal solution.
    """
    # The uniform sequence is the unique minimizer (up to measure-zero exceptions)
    # achieving the theoretical lower bound of N/4 for max overlap.
    h = np.full(N, 0.5)
    
    # Verify the result satisfies all constraints
    assert np.all(h >= 0.0) and np.all(h <= 1.0)
    assert abs(np.sum(h) - N / 2.0) < 1e-12
    
    return h

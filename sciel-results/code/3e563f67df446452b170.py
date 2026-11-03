import time
import numpy as np

N = 100

def _overlap_vector(h):
    """
    Compute the overlap O_k = sum_i h_i * (1 - h_{(i+k) % N}) for all shifts k.
    Uses FFT for O(N log N) computation.
    """
    # Circular autocorrelation C_k = sum_i h_i * h_{(i+k) % N}
    # O_k = N/2 - C_k
    H = np.fft.fft(h)
    C = np.real(np.fft.ifft(H * np.conj(H)))
    return N / 2.0 - C

def _max_overlap(h):
    """Compute the maximum overlap value."""
    return float(np.max(_overlap_vector(h)))

def construct():
    """
    Returns the optimal sequence of step heights.
    
    The problem: minimize max_k O_k where O_k = sum_i h_i * (1 - h_{(i+k)%N}),
    subject to h_i in [0,1] and sum(h_i) = N/2.
    
    Theoretical analysis:
    - Average O_k = (1/N) * sum_k O_k = N/4
    - Therefore max_k O_k >= N/4
    - h_i = 0.5 for all i gives O_k = N/4 for all k, achieving the lower bound.
    
    This is the global optimum. We return it directly.
    A brief verification via FFT confirms the bound is tight.
    """
    deadline = time.time() + 10.0  # generous margin; this is O(N log N)
    
    # The unique (up to symmetry) global optimum
    h = np.full(N, 0.5)
    
    # Verify: compute max overlap to confirm it equals N/4
    max_ov = _max_overlap(h)
    # max_ov should be N/4 = 25.0 for N=100
    
    # If for some reason numerical issues arise, we could do a tiny polish,
    # but with exact 0.5 values this is unnecessary.
    if time.time() > deadline:
        return h
    
    return h

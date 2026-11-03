import numpy as np

def construct():
    N = 100
    # The optimal solution for minimizing the maximum overlap of h with shifted 1-h
    # subject to sum(h) = N/2 and 0 <= h <= 1 is the constant sequence h_i = 0.5.
    # This achieves an overlap of 0.5 for all shifts, which is the theoretical minimum
    # for the maximum overlap.
    
    h = np.full(N, 0.5)
    
    # Verify constraints
    # Sum must be exactly N/2
    assert np.allclose(np.sum(h), N / 2.0), "Sum constraint violated"
    assert np.all(h >= 0) and np.all(h <= 1), "Bounds constraint violated"
    
    return h

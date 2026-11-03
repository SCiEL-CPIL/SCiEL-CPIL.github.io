import numpy as np

def construct():
    N = 100
    # The optimal solution for minimizing the maximum overlap of h with shifted 1-h
    # subject to sum(h) = N/2 and 0 <= h <= 1 is the constant sequence h[i] = 0.5.
    # Proof: For C(k) = sum_i h[i]*(1-h[i+k]), sum_k C(k) = (sum_i h[i])*(N - sum_j h[j]) = (N/2)*(N/2) = N^2/4.
    # Thus max_k C(k) >= N^2/4 / N = N/4.
    # For h[i] = 0.5: C(k) = sum_i 0.5*0.5 = N/4 for all k, achieving the lower bound.
    h = np.full(N, 0.5)
    
    # Verify constraint: sum must be exactly N/2
    assert np.isclose(h.sum(), N/2), f"Sum {h.sum()} != {N/2}"
    assert np.all(h >= 0) and np.all(h <= 1)
    
    return h

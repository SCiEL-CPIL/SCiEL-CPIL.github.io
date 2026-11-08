import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # The theoretical minimum for the maximum overlap is achieved by the constant function h(x) = 0.5.
    # Proof:
    # Let O(k) = sum(h[i] * (1 - h[(i+k) % N]))
    # Average O(k) over all k is sum(h[i] * (1 - mean(h))) = sum(h[i] * 0.5) = 0.5 * (N/2) = N/4.
    # Since max(O(k)) >= average(O(k)), max(O(k)) >= N/4.
    # For h[i] = 0.5, O(k) = sum(0.5 * 0.5) = N/4 for all k.
    # Thus, h=0.5 is a global minimum.
    
    h = np.full(N, 0.5)
    
    # Verify constraints
    if time.time() < deadline:
        assert np.all(h >= 0) and np.all(h <= 1)
        assert abs(h.sum() - N / 2) < 1e-10
        
    return h

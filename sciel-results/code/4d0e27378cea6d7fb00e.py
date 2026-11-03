import numpy as np
import time
import random

N = 600
# Domain is [-0.25, 0.25].
# We restrict to [0, 0.25] to make f one-sided (lag 0 peak = 0).
# This corresponds to indices 300 to 599.
# Let's use indices 300..599 (300 slots).

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Number of pulses
    K = 200
    
    # Available slots (one-sided half of the domain)
    slots = np.arange(300, 600)
    
    # Initial random positions
    positions = np.random.choice(slots, K, replace=False)
    positions.sort()
    
    def get_heights(pos):
        h = np.zeros(N)
        h[pos] = 1.0
        return h
    
    def calc_score(pos):
        # Calculate auto-correlation peak (excluding lag 0)
        # Use FFT for speed
        h = get_heights(pos)
        # Zero-pad to avoid circular convolution issues
        # Max lag is N. Size 2*N is safe.
        f = np.fft.fft(h, n=2*N)
        ac = np.fft.ifft(f * np.conj(f)).real
        # ac[k] is correlation at lag k.
        # We want max of ac[1:] and ac[-1:-N-1] (negative lags)
        # Since f is one-sided, negative lags are 0?
        # f is 0 for x < 0.
        # f*f(x) for x < 0 is 0.
        # So we only care about positive lags.
        # ac[0] is sum of squares = K.
        # We want max(ac[1:N+1])
        peak = np.max(ac[1:N+1])
        # Score is proportional to peak / K
        # We minimize peak
        return peak
        
    current_pos = positions.copy()
    current_peak = calc_score(current_pos)
    best_pos = current_pos.copy()
    best_peak = current_peak
    
    # Local Search
    while time.time() < deadline:
        # Pick a random pulse
        idx = np.random.randint(K)
        old_pos = current_pos[idx]
        
        # Try to move to a random empty slot
        # Get empty slots
        occupied = set(current_pos)
        empty = [s for s in slots if s not in occupied]
        if not empty:
            break
        new_pos = random.choice(empty)
        
        # Update
        current_pos[idx] = new_pos
        
        # Evaluate
        new_peak = calc_score(current_pos)
        
        if new_peak < current_peak:
            current_peak = new_peak
            if new_peak < best_peak:
                best_peak = new_peak
                best_pos = current_pos.copy()
        else:
            # Revert
            current_pos[idx] = old_pos
            
    # Construct final heights
    final_h = get_heights(best_pos)
    return final_h.tolist()

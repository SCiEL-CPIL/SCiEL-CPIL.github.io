import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initialize
    h = np.full(N, 0.5)
    
    # Helper to calculate value and gradients
    def compute_metrics(h):
        # h is 1D array
        # Calculate C_k for all k
        # Use FFT for speed? N=100 is small, direct is fine.
        # C_k = sum h[i] h[i+k]
        # We need min C_k
        
        # Efficient way:
        # corr = np.correlate(h, h, mode='full')
        # This gives C_k for k from -(N-1) to N-1.
        # Note: np.correlate(h, h) computes sum h[i] h[i+k]?
        # Actually np.correlate(a, v) is sum a[n+k] * conj(v[n]).
        # So correlate(h, h) gives sum h[n+k] h[n].
        # This is C_{-k} if we define C_k = sum h[i+k]h[i].
        # Symmetric for real h.
        
        c = np.correlate(h, h, mode='full')
        # c[0] is C_{-(N-1)}? No.
        # Index 0 corresponds to lag -(N-1).
        # Index N-1 corresponds to lag 0.
        # Index 2N-2 corresponds to lag N-1.
        
        # We want min of c.
        min_val = np.min(c)
        
        # Find the lag(s) with min value
        min_indices = np.where(c == min_val)[0]
        
        return min_val, min_indices, c

    # Gradient step
    eta = 0.01
    best_h = h.copy()
    best_v = 1.0 # Initial V is approx 1.0
    
    for step in range(10000):
        if time.time() > deadline:
            break
            
        min_val, min_indices, c = compute_metrics(h)
        
        # Calculate V
        S = np.sum(h)
        # V = (S - min_val) / (N/2)
        # Actually V = 2 * (S - min_val) / N
        current_v = 2.0 * (S - min_val) / N
        
        if current_v < best_v:
            best_v = current_v
            best_h = h.copy()
            
        # Gradient of min_val w.r.t h
        # d(min C_k) / d h_j
        # If multiple min indices, sum gradients? Or pick one?
        # Let's pick the one with largest magnitude gradient or just average.
        
        grad = np.zeros(N)
        for idx in min_indices:
            # idx is index in 'full' correlation array
            # lag = idx - (N - 1)
            lag = idx - (N - 1)
            
            # d C_lag / d h_j = h_{j-lag} + h_{j+lag}
            # Need to handle bounds
            
            # For each j, if j-lag is in [0, N-1], add h[j-lag]
            # if j+lag is in [0, N-1], add h[j+lag]
            
            # Vectorized approach:
            # h_shifted_1 = shift h by -lag
            # h_shifted_2 = shift h by +lag
            
            # Simpler:
            # Create indices
            js = np.arange(N)
            
            # j - lag
            idx1 = js - lag
            mask1 = (idx1 >= 0) & (idx1 < N)
            grad[mask1] += h[idx1[mask1]]
            
            # j + lag
            idx2 = js + lag
            mask2 = (idx2 >= 0) & (idx2 < N)
            grad[mask2] += h[idx2[mask2]]
            
        # We want to MAXIMIZE min_val.
        # So h += eta * grad
        
        h += eta * grad
        
        # Project to [0, 1]
        h = np.clip(h, 0.0, 1.0)
        
        # Enforce sum constraint
        # Current sum
        S_curr = np.sum(h)
        target_sum = N / 2.0
        
        if S_curr > 0:
            # Scale? No, clipping breaks scaling.
            # Simple adjustment: add (target - S_curr) / N to all, then clip again.
            # This might oscillate.
            # Better: find elements with room to grow/shrink.
            
            diff = target_sum - S_curr
            if abs(diff) > 1e-6:
                # Distribute diff
                # If diff > 0, add to elements < 1
                # If diff < 0, subtract from elements > 0
                
                # Simple iterative projection
                for _ in range(10):
                    S_curr = np.sum(h)
                    diff = target_sum - S_curr
                    if abs(diff) < 1e-9:
                        break
                    
                    if diff > 0:
                        # Add to elements not at 1
                        room = 1.0 - h
                        total_room = np.sum(room)
                        if total_room > 1e-9:
                            h += diff * (room / total_room)
                    else:
                        # Subtract from elements not at 0
                        room = h
                        total_room = np.sum(room)
                        if total_room > 1e-9:
                            h -= (-diff) * (room / total_room)
                    
                    h = np.clip(h, 0.0, 1.0)
        
        # Decay learning rate
        eta *= 0.999

    return best_h

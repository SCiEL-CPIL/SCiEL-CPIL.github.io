import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    rng = np.random.default_rng(42)
    h = rng.choice([-1, 1], size=N)
    
    # Compute initial autocorrelation
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    R_full = np.fft.irfft(power, n=n_fft)
    R = R_full[1:N].copy()
    
    current_peak = np.max(np.abs(R))
    best_h = h.copy()
    best_peak = current_peak
    
    # Simulated Annealing
    T = 5.0
    cooling_rate = 0.999
    # We will run until time limit or T is very small
    # With N=400, each step is O(N) for update and O(N) for max.
    # 400 * 2 = 800 ops per step.
    # 1e6 steps is 8e8 ops. Might be too slow in Python loops.
    # We need to vectorize the update or accept fewer steps.
    # Let's try to do as many as possible.
    
    # Pre-compute indices? No, random i each time.
    # The update is:
    # delta_R[k] = -2 * h[i] * (h[(i+k)%N] + h[(i-k)%N])
    # This requires indexing h with (i+k)%N and (i-k)%N.
    
    # To speed up, we can use a rolling window or just rely on NumPy's speed for the vector ops.
    # np.roll is slow. Direct indexing is better.
    
    steps = 0
    while time.time() < deadline:
        # Pick random index to flip
        i = rng.integers(N)
        
        # Calculate new R
        # We need h_new = h.copy() ? No, we can calculate delta.
        # But we need to update R in place or create new R.
        # Creating new R array of size 400 is cheap.
        
        # Term: h[(i+k)%N] + h[(i-k)%N] for k=1..N-1
        # This is effectively a circular convolution of h with a delta-like vector?
        # Actually, let's just compute the vector.
        # k = 1..N-1
        # idx_plus = (i + np.arange(1, N)) % N
        # idx_minus = (i - np.arange(1, N)) % N
        
        # To avoid allocating arange every time, pre-compute?
        # The indices depend on i.
        # However, we can just do:
        # shifted_plus = np.roll(h, -i) # h[i+k] is at index k in rolled array?
        # np.roll(h, -i)[k] = h[(i+k)%N]. Yes.
        # np.roll(h, i)[k] = h[(i-k)%N]. Yes.
        
        # np.roll is implemented in C and is fast for N=400.
        h_plus = np.roll(h, -i)
        h_minus = np.roll(h, i)
        
        # term = h_plus[1:N] + h_minus[1:N]
        # Note: R[0] is not tracked in R (R is lags 1..N-1).
        # The update formula for lag k is -2*h[i]*(h[i+k] + h[i-k]).
        
        term = h_plus[1:N] + h_minus[1:N]
        delta_R = -2.0 * h[i] * term
        
        new_R = R + delta_R
        new_peak = np.max(np.abs(new_R))
        
        # Acceptance
        if new_peak < current_peak or (T > 0.01 and rng.random() < np.exp(-(new_peak - current_peak) / T)):
            # Accept move
            h[i] = -h[i]
            R = new_R
            current_peak = new_peak
            
            if current_peak < best_peak:
                best_peak = current_peak
                best_h = h.copy()
                
        T *= cooling_rate
        steps += 1
        
        # Check time more frequently if steps are fast
        if steps % 100 == 0 and time.time() > deadline:
            break
            
    return best_h

import numpy as np
import time

N = 400

def _max_sidelobe(h):
    """
    Compute the maximum absolute value of the aperiodic autocorrelation sidelobes.
    h: numpy array of length N containing 1s and -1s.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    
    # Linear aperiodic autocorrelation:
    # Lag k (1 <= k < N) is at ac[k]
    # Lag -k (-(N-1) <= k <= -1) is at ac[N+1 : 2*N] (specifically ac[N+1] is lag -(N-1), ac[2*N-1] is lag -1)
    # Due to symmetry, max |ac[1:N]| is sufficient, but we take both for safety against numerical noise.
    pos_lags = ac[1:N]
    neg_lags = ac[N+1:2*N]
    
    return np.max(np.concatenate([np.abs(pos_lags), np.abs(neg_lags)]))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_h = None
    best_val = float('inf')
    
    # Random initialization
    h = np.random.choice([-1.0, 1.0], size=N)
    current_val = _max_sidelobe(h)
    best_val = current_val
    best_h = h.copy()
    
    # Local search with occasional restarts if stuck
    no_improve_count = 0
    max_no_improve = 10000
    
    while time.time() < deadline:
        # Pick a random index to flip
        i = np.random.randint(N)
        
        # Flip the bit
        h[i] = -h[i]
        
        # Calculate new objective
        new_val = _max_sidelobe(h)
        
        if new_val < current_val:
            current_val = new_val
            no_improve_count = 0
            # Update global best
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
        else:
            # Revert the flip
            h[i] = -h[i]
            no_improve_count += 1
        
        # If stuck for too long, restart from a random sequence
        if no_improve_count >= max_no_improve:
            h = np.random.choice([-1.0, 1.0], size=N)
            current_val = _max_sidelobe(h)
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
            no_improve_count = 0
            
    return best_h

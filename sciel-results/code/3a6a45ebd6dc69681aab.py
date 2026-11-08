import numpy as np
import time

N = 400

def _compute_score(h):
    """Compute the maximum absolute autocorrelation at non-zero lags, normalized."""
    H = np.fft.fft(h)
    R = np.fft.ifft(np.abs(H) ** 2).real
    # R[0] = N, we want max of |R[k]| for k != 0
    # R is circular autocorrelation; for a real sequence, R[k] = sum h[i]*h[(i+k)%N]
    # We want the max over non-zero lags
    max_abs = np.max(np.abs(R[1:]))
    return max_abs / N

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    np.random.seed(12345)
    
    # Start with a random ±1 sequence
    h = np.random.choice([-1, 1], size=N)
    best_h = h.copy()
    best_score = _compute_score(best_h)
    
    # Local search: flip bits to reduce max autocorrelation
    for iteration in range(20000):
        if time.time() > deadline:
            break
        
        # Compute current autocorrelation
        H = np.fft.fft(best_h)
        R = np.fft.ifft(np.abs(H) ** 2).real
        abs_R = np.abs(R).copy()
        abs_R[0] = 0
        current_max = np.max(abs_R)
        
        # Find the worst lag
        worst_lag = np.argmax(abs_R)
        
        # Try flipping each bit, see which one most reduces the worst autocorrelation
        best_delta = 0
        best_flip = -1
        
        # Sample a subset of positions to try (for speed)
        n_tries = min(50, N)
        positions = np.random.choice(N, size=n_tries, replace=False)
        
        for pos in positions:
            # Flip bit at pos
            h_trial = best_h.copy()
            h_trial[pos] *= -1
            
            # Compute new autocorrelation efficiently
            H_trial = np.fft.fft(h_trial)
            R_trial = np.fft.ifft(np.abs(H_trial) ** 2).real
            abs_R_trial = np.abs(R_trial).copy()
            abs_R_trial[0] = 0
            new_max = np.max(abs_R_trial)
            
            delta = new_max - current_max
            if delta < best_delta:
                best_delta = delta
                best_flip = pos
        
        if best_flip >= 0:
            best_h[best_flip] *= -1
            best_score = _compute_score(best_h)
        else:
            # No improvement found, try a random restart or accept a random move
            if iteration % 500 == 0:
                # Occasionally do a random perturbation
                n_perturb = np.random.randint(1, 5)
                perturb_pos = np.random.choice(N, size=n_perturb, replace=False)
                best_h[perturb_pos] *= -1
                best_score = _compute_score(best_h)
    
    return best_h

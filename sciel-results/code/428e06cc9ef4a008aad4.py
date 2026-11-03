import numpy as np
import time

N = 400
_DEADLINE = 145.0
_T0 = time.monotonic()

def _time_left():
    return _DEADLINE - (time.monotonic() - _T0)

def _score(x):
    """
    Calculate the metric: max_{k != 0} |c_k| / (sum x)^2
    Assumes x is a 1D numpy array.
    """
    S = np.sum(x)
    if S == 0:
        return np.inf
    # Compute autoconvolution
    # Using FFT for speed
    n_fft = 2 * N
    F = np.fft.fft(x, n=n_fft)
    P = F * np.conj(F)
    c = np.fft.ifft(P).real
    # c has length n_fft, but we only care about lags 0 to N-1
    # Linear autoconvolution lags are 0 to N-1
    # c[0] is lag 0. c[1] is lag 1... c[N-1] is lag N-1.
    # Note: np.fft.ifft of P gives circular conv, but with zero-padding it's linear.
    # The lags are 0, 1, ..., N-1.
    sidelobes = c[1:N]
    max_sid = np.max(np.abs(sidelobes))
    return max_sid / (S * S)

def construct():
    # Start with the constant vector of 1s
    x = np.ones(N, dtype=np.float64)
    best_x = x.copy()
    best_score = _score(best_x)
    
    # If time is low, return immediately
    if _time_left() < 5.0:
        return best_x
    
    # Local search / Simulated Annealing
    # We want to minimize the score.
    # Try flipping random bits.
    
    current_x = x.copy()
    current_score = best_score
    
    # Simple greedy hill climbing with random restarts
    # Since N=400, FFT is fast.
    
    steps = 0
    max_steps = 20000  # Limit to ensure we finish in time
    
    while _time_left() > 1.0 and steps < max_steps:
        # Pick a random index to flip
        idx = np.random.randint(N)
        current_x[idx] *= -1
        
        new_score = _score(current_x)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = current_x.copy()
        else:
            # Revert if not improved (Greedy)
            # Or accept with probability p (Annealing)
            # Given the small improvement expected, greedy might get stuck.
            # But let's try a bit of annealing.
            T = 0.01 * (1.0 - steps / max_steps)
            delta = new_score - current_score
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_score = new_score
            else:
                current_x[idx] *= -1
                
        steps += 1
        
    return best_x

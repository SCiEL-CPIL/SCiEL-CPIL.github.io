import numpy as np
import time

N = 400

def _autoconv(x):
    """Compute full autoconvolution of x."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    ac = np.fft.irfft(X * np.conj(X), size)[:2 * n - 1]
    return ac

def _score(x):
    """Compute max|autoconv| / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv(x)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1."""
    tm = np.zeros(n, dtype=np.int8)
    tm[0] = 1
    for i in range(1, n):
        # Count trailing 1s in binary representation of i
        j = i
        parity = 0
        while j:
            parity ^= (j & 1)
            j >>= 1
        tm[i] = 1 if parity == 0 else -1
    return tm.astype(np.float64)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    np.random.seed(42)

    # Start with Thue-Morse sequence
    x_tm = _thue_morse(N)
    score_tm = _score(x_tm)
    
    # Also try a few random initializations
    best_x = x_tm.copy()
    best_score = score_tm
    
    for init_trial in range(10):
        if time.time() > deadline - 60:
            break
        x = np.random.choice([-1.0, 1.0], size=N)
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()

    # Local sign-flip optimization with simulated annealing
    x = best_x.copy()
    current_score = best_score
    
    T = 0.5
    T_min = 1e-7
    cooling = 0.9998
    
    # Track number of evaluations for progress
    n_evals = 0
    
    while time.time() < deadline - 10.0:
        if T < T_min:
            break
        
        # Single flip
        idx = np.random.randint(N)
        x_new = x.copy()
        x_new[idx] *= -1.0
        
        new_score = _score(x_new)
        n_evals += 1
        
        if new_score < current_score or np.random.random() < np.exp(-(new_score - current_score) / max(T, 1e-10)):
            x = x_new
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
        
        T *= cooling
        
        # Occasionally restart from best
        if np.random.random() < 0.005:
            x = best_x.copy()
            current_score = best_score
    
    # Final local refinement: single-flip search until no improvement
    x = best_x.copy()
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        for i in range(N):
            if time.time() > deadline - 3.0:
                break
            x_old = x[i]
            x[i] = -x[i]
            sc = _score(x)
            if sc < best_score - 1e-12:
                best_score = sc
                best_x = x.copy()
                improved = True
            else:
                x[i] = x_old
    
    return best_x

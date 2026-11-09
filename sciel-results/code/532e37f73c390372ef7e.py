import numpy as np
import time

N = 400

def _compute_sos_metric(h):
    """Compute sum of squared autocorrelation at non-zero lags."""
    H = np.fft.fft(h)
    R = np.fft.ifft(np.abs(H) ** 2).real
    return np.sum(R[1:]**2)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial random sequence
    rng = np.random.RandomState(42)
    h = rng.choice([-1.0, 1.0], N)
    
    current_score = _compute_sos_metric(h)
    best_h = h.copy()
    best_score = current_score
    
    # Simulated Annealing parameters
    # Estimate initial temperature based on score scale
    # For N=400, SOS is ~160,000.
    T = 1000.0
    cooling_rate = 0.9995
    min_T = 1e-6
    
    steps = 0
    while time.time() < deadline - 5:
        # Choose a random index to flip
        idx = rng.randint(0, N)
        
        h_trial = h.copy()
        h_trial[idx] = -h_trial[idx]
        
        trial_score = _compute_sos_metric(h_trial)
        delta = trial_score - current_score
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or (T > min_T and rng.random() < np.exp(-delta / T)):
            h = h_trial
            current_score = trial_score
            
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
        
        T *= cooling_rate
        steps += 1
        
    # Final polish: Greedy search on the best found
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for idx in range(N):
            if time.time() > deadline - 2:
                break
            h_trial = best_h.copy()
            h_trial[idx] = -h_trial[idx]
            trial_score = _compute_sos_metric(h_trial)
            if trial_score < best_score - 1e-12:
                best_h = h_trial
                best_score = trial_score
                improved = True
                
    return best_h

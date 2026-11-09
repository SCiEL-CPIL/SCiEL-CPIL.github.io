"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Deadline: 110 seconds (safety margin for 120s limit)
    start_time = time.time()
    time_limit = 110.0
    
    rng = np.random.default_rng(42)
    
    # Structural Change: Initialize with Thue-Morse sequence
    # Thue-Morse has low autocorrelation sidelobes compared to random.
    a = np.array([1.0 if bin(i).count('1') % 2 == 0 else -1.0 for i in range(N)])
    
    # Compute initial circular autocorrelation R
    def compute_R(a):
        fa = np.fft.fft(a)
        power = fa * np.conj(fa)
        R = np.fft.ifft(power).real
        return R

    current_R = compute_R(a)
    # Metric: Peak Sidelobe Level (Max absolute value of non-zero lags)
    current_peak = np.max(np.abs(current_R[1:]))
    
    best_a = a.copy()
    best_peak = current_peak
    current_a = a.copy()
    
    # Simulated Annealing with Incremental Updates for Peak Sidelobe
    # We use a simple time-based temperature decay
    # T is scaled to the magnitude of the peak
    T_base = 0.5
    
    while time.time() - start_time < time_limit:
        # Simple cooling: T decreases as time passes
        elapsed = time.time() - start_time
        T = T_base * np.exp(-elapsed / 20.0) # Cools down over 20s
        
        # Pick random index
        idx = rng.integers(0, N)
        
        # Calculate change in R if we flip a[idx]
        delta_val = -2.0 * current_a[idx]
        
        # Incremental update of R
        # R_new[k] = R_old[k] + delta_val * (a[(idx+k)%N] + a[(idx-k)%N])
        # Using np.roll for efficiency
        term = np.roll(current_a, idx) + np.roll(current_a, -idx)
        delta_R = delta_val * term
        
        candidate_R = current_R + delta_R
        candidate_peak = np.max(np.abs(candidate_R[1:]))
        
        # Acceptance criteria
        if candidate_peak < current_peak:
            current_R = candidate_R
            current_a[idx] *= -1.0
            current_peak = candidate_peak
            
            if current_peak < best_peak:
                best_peak = current_peak
                best_a = current_a.copy()
        else:
            # Accept worse moves with probability exp(-delta/T)
            # Delta is in units of Peak Level
            delta = candidate_peak - current_peak
            if T > 1e-9:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_R = candidate_R
                    current_a[idx] *= -1.0
                    current_peak = candidate_peak

    return best_a
# EVOLVE-BLOCK-END

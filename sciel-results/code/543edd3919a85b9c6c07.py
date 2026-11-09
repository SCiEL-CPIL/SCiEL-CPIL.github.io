"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Objective: Minimize the upper bound on C3.
    # Based on inspirations, the metric is the peak sidelobe magnitude
    # of the aperiodic autocorrelation: max(|R(k)|) for k != 0.
    # The task mentions "Paper and published result notebook place the absolute value differently; execution must explicitly select a metric."
    # We select the peak sidelobe metric as it is standard for "autoconvolution upper bound" in this context (related to C3 inequality).
    
    def get_peak_sidelobe(x):
        # Use FFT for efficient computation of aperiodic autocorrelation
        # Zero-pad to 2*N to ensure aperiodic autocorrelation for lags 0..N-1
        n_fft = 2 * N
        F = np.fft.rfft(x, n=n_fft)
        P = F * np.conj(F)
        R = np.fft.irfft(P, n=n_fft)
        # R[0] is lag 0 (sum of squares = N).
        # We want max |R[k]| for k = 1, ..., N-1
        return np.max(np.abs(R[1:N]))
    
    best_x = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Simulated Annealing with multiple restarts
    # We use a structural change: instead of pure greedy local search (current)
    # or simple SA (inspiration 2/3), we combine SA for global exploration
    # with a fast greedy polish.
    
    sa_deadline = start_time + (115.0 - 10.0)  # Leave 10s for final polish
    
    # Try a few random restarts
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() > sa_deadline:
            break
            
        # Initialize with random signs
        x = rng.choice([-1.0, 1.0], size=N)
        current_score = get_peak_sidelobe(x)
        
        # Track best in this run
        run_best_x = x.copy()
        run_best_score = current_score
        
        # Simulated Annealing parameters
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.995
        
        # Inner SA loop
        while T > T_min and time.time() < sa_deadline:
            # Pick a random position to flip
            i = rng.integers(0, N)
            
            # Try flipping
            old_val = x[i]
            x[i] = -old_val
            
            new_score = get_peak_sidelobe(x)
            delta = new_score - current_score
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
                current_score = new_score
                if new_score < run_best_score:
                    run_best_score = new_score
                    run_best_x = x.copy()
            else:
                # Revert
                x[i] = old_val
            
            T *= cooling_rate
            
        # Update global best
        if run_best_score < best_score:
            best_score = run_best_score
            best_x = run_best_x.copy()
            
        # Early termination if we've found a very good solution
        # Theoretical lower bound for peak sidelobe is ~sqrt(N) = 20.
        # If best_score < 22, it's quite good.
        if best_score < 22.0:
            break
    
    # Phase 2: Greedy local search polish on the best solution
    if best_x is not None and time.time() < deadline - 2.0:
        x = best_x.copy()
        current_score = best_score
        polish_deadline = deadline - 2.0
        
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            for i in range(N):
                if time.time() > polish_deadline:
                    break
                
                # Try flipping position i
                old_val = x[i]
                x[i] = -old_val
                
                new_score = get_peak_sidelobe(x)
                
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_x = x.copy()
                else:
                    # Revert
                    x[i] = old_val
                    
    return best_x
# EVOLVE-BLOCK-END

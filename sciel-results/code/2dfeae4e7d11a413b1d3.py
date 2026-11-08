"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _c3_upper_bound(x):
    """
    Compute the upper bound on C3 for a signed step function.
    Metric: h * max(|autoconvolution|)
    """
    n_fft = 2 * N
    f_x = np.fft.rfft(x, n=n_fft)
    f_conv = f_x * f_x
    conv = np.fft.irfft(f_conv, n=n_fft)[:2*N-1]
    h = 0.5 / N
    return h * np.max(np.abs(conv))

def _local_search(x, time_limit):
    """
    Fast local search using single flips.
    """
    x = x.copy()
    best_score = _c3_upper_bound(x)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(N):
            old_val = x[i]
            x[i] = -old_val
            new_score = _c3_upper_bound(x)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                x[i] = old_val
    return x, best_score

def _simulated_annealing(deadline):
    """
    Simulated Annealing for optimizing the step function.
    Uses probabilistic uphill moves to escape local minima.
    """
    # Start with random initial solution
    x = np.random.choice([-1.0, 1.0], size=N)
    current_score = _c3_upper_bound(x)
    best_x = x.copy()
    best_score = current_score
    
    # SA parameters
    T_initial = 0.1  # Initial temperature (scale relative to typical score changes)
    T_final = 0.001  # Final temperature
    cooling_rate = 0.999  # Geometric cooling per step
    
    T = T_initial
    start_time = time.time()
    steps = 0
    
    while T > T_final and (time.time() - start_time) < (deadline - start_time) * 0.85:
        # Generate neighbor: flip one or a few random positions
        # Flip a small random number of positions (1-5)
        n_flips = np.random.randint(1, 6)
        indices = np.random.choice(N, size=n_flips, replace=False)
        
        x_old = x[indices].copy()
        x[indices] *= -1.0
        
        new_score = _c3_upper_bound(x)
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            # Probabilistic acceptance
            if delta < 1e-12:  # Numerical tolerance
                current_score = new_score
            else:
                accept_prob = np.exp(-delta / T)
                if np.random.random() < accept_prob:
                    current_score = new_score
                else:
                    # Reject: revert
                    x[indices] = x_old
        
        # Cool down
        T *= cooling_rate
        steps += 1
        
        # Time check with safety margin
        if time.time() > deadline - 15.0:
            break
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Run simulated annealing
    x_best, score_best = _simulated_annealing(deadline)
    
    # Final local search refinement if time remains
    if time.time() < deadline - 5.0:
        time_remaining = deadline - 5.0 - time.time()
        if time_remaining > 0.1:
            x_best, score_best = _local_search(x_best, time_remaining)
    
    return x_best
# EVOLVE-BLOCK-END

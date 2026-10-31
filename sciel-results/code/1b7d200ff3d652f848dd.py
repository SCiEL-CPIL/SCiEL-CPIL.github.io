import numpy as np
import time

N = 400

def _compute_c3_bound(x):
    """Compute the upper bound on C3 for a step function.
    
    The C3 upper bound is computed as the maximum absolute value of the
    autocorrelation sidelobes (excluding lag 0), which provides an upper
    bound in the third autocorrelation inequality.
    """
    # Compute full autocorrelation
    # For a step function, autocorrelation at lag k is sum_i x[i]*x[i+k]
    ac = np.correlate(x, x, mode='full')
    # The peak is at index N-1 (lag 0)
    center = N - 1
    # Sidelobes are all other lags
    sidelobes = np.delete(ac, center)
    return np.max(np.abs(sidelobes))

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Start with a random {-1, 1} signed step function
    rng = np.random.default_rng(42)
    best_x = rng.choice([-1.0, 1.0], size=N)
    best_val = _compute_c3_bound(best_x)
    
    # Simulated annealing to minimize the C3 bound
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    iteration = 0
    while time.time() < deadline and T > min_T:
        # Random perturbation: flip a small number of elements
        num_flips = max(1, int(5 * T))
        indices = rng.choice(N, size=num_flips, replace=False)
        x_new = best_x.copy()
        x_new[indices] *= -1
        
        val_new = _compute_c3_bound(x_new)
        
        # Accept if better, or with probability exp(-delta/T)
        if val_new < best_val:
            best_x = x_new
            best_val = val_new
        else:
            delta = val_new - best_val
            if delta < 0:
                best_x = x_new
                best_val = val_new
            else:
                if rng.random() < np.exp(-delta / max(T, 1e-10)):
                    best_x = x_new
                    best_val = val_new
        
        T *= cooling_rate
        iteration += 1
    
    # Final local search: try single flips
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                x_trial = best_x.copy()
                x_trial[i] *= -1
                val_trial = _compute_c3_bound(x_trial)
                if val_trial < best_val:
                    best_x = x_trial
                    best_val = val_trial
                    improved = True
    
    return best_x

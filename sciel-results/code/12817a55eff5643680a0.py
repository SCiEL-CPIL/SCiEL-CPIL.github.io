import numpy as np
import time

N = 400

def _compute_psl(x):
    """
    Compute the peak sidelobe level (PSL) of the autocorrelation of x.
    x is a sequence of +/- 1.
    """
    n = len(x)
    size = 1
    while size < 2 * n:
        size *= 2
    X = np.fft.rfft(x, size)
    acf = np.fft.irfft(X * np.conj(X), size)[:n]
    # Sidelobes are lags 1 to n-1
    if n > 1:
        return np.max(np.abs(acf[1:]))
    return 0.0

def _local_search(x_init, time_limit):
    """
    Perform local search to minimize PSL.
    Uses first-improvement strategy with random perturbations if stuck.
    """
    start_time = time.time()
    x = x_init.copy()
    current_psl = _compute_psl(x)
    best_psl = current_psl
    best_x = x.copy()
    
    n = len(x)
    stall_counter = 0
    max_stall = 50
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Iterate through all positions
        for i in range(n):
            # Flip bit i
            x[i] *= -1
            new_psl = _compute_psl(x)
            
            if new_psl < current_psl:
                current_psl = new_psl
                improved = True
                if current_psl < best_psl:
                    best_psl = current_psl
                    best_x = x.copy()
            else:
                # Revert
                x[i] *= -1
            
            # Check time periodically to avoid long loops
            if (i + 1) % 50 == 0:
                if time.time() - start_time > time_limit - 0.5:
                    return best_x, best_psl
        
        if not improved:
            stall_counter += 1
            if stall_counter > max_stall:
                # Perturb to escape local minimum
                # Flip 5 random bits
                for _ in range(5):
                    idx = np.random.randint(n)
                    x[idx] *= -1
                current_psl = _compute_psl(x)
                stall_counter = 0
        else:
            stall_counter = 0
            
    return best_x, best_psl

def construct():
    """
    Construct a signed step function (±1 sequence of length N) that minimizes
    the C3 autoconvolution upper bound (Peak Sidelobe Level).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    np.random.seed(42)
    
    best_x = None
    best_val = float('inf')
    
    # Phase 1: Multiple random restarts with local search
    # Each restart gets a portion of the time.
    # We estimate how many restarts we can do.
    # Local search for N=400 is fast.
    
    while time.time() - start_time < deadline - 5.0:
        # Initialize with random sequence
        x_init = np.random.choice([-1.0, 1.0], size=N)
        
        # Allocate time for this restart
        # If we have plenty of time, give it 10s. If little, give it the rest.
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        
        # Adaptive time allocation: start with small chunks, increase if improving
        # For simplicity, use fixed 5s chunks, but ensure we finish at least one.
        t_alloc = min(5.0, remaining - 2.0)
        
        x_res, val = _local_search(x_init, t_alloc)
        
        if val < best_val:
            best_val = val
            best_x = x_res.copy()
            
        # Optional: If we found a very good solution, we might want to spend more time polishing it?
        # For now, just keep restarting to explore different basins.
        
    # Fallback if no valid result was found (should not happen)
    if best_x is None:
        return np.ones(N)
        
    return best_x

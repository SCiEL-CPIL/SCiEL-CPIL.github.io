import numpy as np
import time

N = 400

def _get_cost(h):
    """
    Compute the upper bound on C3: max(|autocorr_sidelobe|) / N
    """
    size = 2 * N
    H = np.fft.rfft(h, n=size)
    C = np.fft.irfft(H * H, n=size)[:N]
    # C[0] is the DC component (sum of squares = N for +/- 1)
    # We want max absolute value of sidelobes C[1:]
    if N > 1:
        max_side = np.max(np.abs(C[1:]))
    else:
        max_side = 0.0
    return max_side / N

def _local_search(h, max_iters=100, temperature=0.0, cooling_rate=0.99):
    """
    Simulated Annealing / Local Search hybrid.
    """
    current_cost = _get_cost(h)
    best_h = h.copy()
    best_cost = current_cost
    T = temperature if temperature > 0 else 1.0
    
    for i in range(max_iters):
        # Pick a random index to flip
        idx = np.random.randint(N)
        old_val = h[idx]
        h[idx] = -old_val
        
        new_cost = _get_cost(h)
        delta = new_cost - current_cost
        
        if delta < 0:
            # Improvement
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_h = h.copy()
        else:
            # Worse, accept with probability exp(-delta/T)
            if np.random.rand() < np.exp(-delta / T):
                current_cost = new_cost
            else:
                # Reject, revert
                h[idx] = old_val
        
        T *= cooling_rate
        
        # Early stop if T is very small and no improvement recently?
        # For simplicity, just run fixed iters or until T is negligible.
        if T < 1e-6:
            break
            
    return best_h, best_cost

def construct():
    deadline = time.time() + 110.0 # Safety margin
    np.random.seed(42)
    
    best_overall_h = None
    best_overall_cost = float('inf')
    
    # Multiple restarts to escape local minima
    num_restarts = 0
    
    while time.time() < deadline:
        # 1. Initialize
        # Try a mix of random and structured initializations
        if num_restarts % 3 == 0:
            h = np.random.choice([-1.0, 1.0], size=N)
        elif num_restarts % 3 == 1:
            # Random block structure
            h = np.zeros(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(1, 10)
                val = np.random.choice([-1.0, 1.0])
                end = min(pos + block_len, N)
                h[pos:end] = val
                pos = end
        else:
            # Random
            h = np.random.choice([-1.0, 1.0], size=N)
            
        # 2. Local Search (Simulated Annealing)
        # Use a higher initial temperature for exploration
        h_opt, cost_opt = _local_search(h, max_iters=200, temperature=5.0, cooling_rate=0.95)
        
        if cost_opt < best_overall_cost:
            best_overall_cost = cost_opt
            best_overall_h = h_opt.copy()
            
        # 3. If we have time, do a second pass with lower temperature on the best found so far
        if time.time() < deadline - 5.0 and best_overall_h is not None:
             # Polish the best found
             h_polish, cost_polish = _local_search(best_overall_h.copy(), max_iters=100, temperature=0.5, cooling_rate=0.9)
             if cost_polish < best_overall_cost:
                 best_overall_cost = cost_polish
                 best_overall_h = h_polish.copy()
                 
        num_restarts += 1
        # If we are getting close to deadline, stop
        if num_restarts > 50:
             break

    return best_overall_h

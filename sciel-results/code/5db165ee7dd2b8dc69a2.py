import numpy as np
import time

N = 400

def _objective(h):
    """
    Calculate the objective function value: max |autoconvolution| / (sum h)^2.
    If sum h is 0, return a large penalty.
    """
    S = np.sum(h)
    if S == 0:
        return 1e9
    C = np.convolve(h, h, mode='full')
    max_abs_C = np.max(np.abs(C))
    return max_abs_C / (S * S)

def _simulated_annealing(h_init, max_iter=2000, start_temp=1.0, end_temp=0.01, deadline=140.0):
    """
    Run simulated annealing to minimize the objective.
    """
    h = h_init.copy()
    current_val = _objective(h)
    best_h = h.copy()
    best_val = current_val
    
    T = start_temp
    cooling_rate = (end_temp / start_temp) ** (1.0 / max_iter)
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random index to flip
        idx = np.random.randint(N)
        
        # Flip
        h[idx] *= -1
        
        # Calculate new objective
        new_val = _objective(h)
        
        # Accept or reject
        if new_val < current_val:
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Probabilistic acceptance
            delta = new_val - current_val
            if delta < 0:
                accept = True
            else:
                # delta is positive, so -delta/T is negative
                # If T is very small, exp is ~0
                if T > 1e-6:
                    accept = np.random.rand() < np.exp(-delta / T)
                else:
                    accept = False
            
            if accept:
                current_val = new_val
            else:
                # Revert
                h[idx] *= -1
        
        T *= cooling_rate
        
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = None
    best_val = float('inf')
    
    # Set random seed for reproducibility in the sense of deterministic behavior per run?
    # The task doesn't require deterministic output, but consistent improvement.
    # Using a fixed seed for the RNG inside the loop helps debugging but might limit search space exploration across runs if the evaluator runs multiple times.
    # However, usually these tasks are single-shot. We will use a time-based seed or fixed seed.
    # Let's use a fixed seed for the initializations but vary the SA process.
    
    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. All 1s
    candidates.append(np.ones(N))
    
    # 2. Biased random sequences
    for p in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        h = rng.choice([-1.0, 1.0], size=N, p=[1-p, p])
        # Ensure sum is not zero
        s = np.sum(h)
        if s == 0:
             h[0] = 1.0 # Flip one to break symmetry
        candidates.append(h)
        
    # 3. Rudin-Shapiro like (low peak, but low sum)
    # We will just let SA fix the sum if needed, but RS is a good starting point for peak reduction.
    def rudin_shapiro(n):
        r = np.array([1.0])
        s = np.array([0.0]) # Not standard, let's use standard RS
        # Standard RS:
        # a_1 = 1
        # a_{n+1} = a_n b_n
        # b_{n+1} = a_n (-1)^{k} b_n ... this is getting complicated to implement correctly from scratch quickly.
        # Let's just use random starts. RS is balanced, so S~0, which is bad for our objective.
        # We want high S. So biased random is better.
        pass
        
    # Run SA on each candidate
    for h_init in candidates:
        if time.time() > deadline:
            break
            
        # Run SA for a limited number of iterations
        # We can run multiple SA restarts for the best candidate
        h_opt, val_opt = _simulated_annealing(h_init, max_iter=3000, start_temp=0.5, end_temp=0.001, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
            
    # If we have time, polish the best solution
    if best_h is not None and time.time() < deadline - 10.0:
        # Run a final hill climbing pass (greedy) to ensure local optimum
        # Or run SA again with lower temp
        h_polished, val_polished = _simulated_annealing(best_h, max_iter=5000, start_temp=0.1, end_temp=0.0001, deadline=deadline)
        if val_polished < best_val:
            best_val = val_polished
            best_h = h_polished.copy()
            
    if best_h is None:
        return np.ones(N)
        
    return best_h

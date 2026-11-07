import numpy as np
import time

N = 100

def compute_cost(h):
    # h is a numpy array
    # 1-h is the other sequence
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr)

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # Initial solution
    # Try a few initializations
    best_h = None
    best_cost = float('inf')
    
    # Initialization 1: Uniform 0.5
    h = np.full(N, 0.5)
    cost = compute_cost(h)
    if cost < best_cost:
        best_cost = cost
        best_h = h.copy()
        
    # Initialization 2: Random uniform, scaled to sum 50
    for _ in range(5):
        h_init = np.random.rand(N)
        h_init = (h_init - h_init.mean()) * 10 + 0.5 # Just to make it varied
        # Actually, let's just do random and then adjust sum
        h_init = np.random.rand(N)
        h_init = h_init / h_init.sum() * 50
        # Clip to [0, 1] and renormalize? This is tricky.
        # Better: start with 0.5 and add random noise, then project to [0,1] and sum 50?
        # Simpler: just use random valid points.
        # Generate random points in [0,1] and adjust.
        
        # Let's stick to a simple random start that is valid
        h_rand = np.random.rand(N) * 0.2 + 0.4 # in [0.4, 0.6]
        # This sums to approx 50.
        # Adjust to sum exactly 50
        diff = 50 - np.sum(h_rand)
        h_rand += diff / N
        # Clip
        h_rand = np.clip(h_rand, 0, 1)
        # Re-adjust sum (might break [0,1] if diff is large, but diff is small here)
        # If clipping changed sum, we need to fix it.
        # For simplicity, let's just use the uniform 0.5 start or a step function.
        
        # Let's try a "block" start
        h_block = np.zeros(N)
        # Two blocks of 50?
        # h[0:50] = 1, h[50:100] = 0? Sum = 50.
        # Let's test this.
        h_test = np.zeros(N)
        h_test[:50] = 1.0
        cost_test = compute_cost(h_test)
        if cost_test < best_cost:
            best_cost = cost_test
            best_h = h_test.copy()

    # Run Simulated Annealing
    h = best_h.copy()
    current_cost = best_cost
    T = 1.0
    T_min = 1e-6
    cooling_rate = 0.99
    
    # Pre-compute indices for moves?
    # Moves: pick i, j. delta.
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
            
        # Perform a move
        i, j = np.random.choice(N, 2, replace=False)
        
        # Choose delta
        # Adaptive delta? Or fixed small delta?
        # Let's use a delta that allows exploration.
        # Max change in h is 1.
        # Let's try delta ~ U(-0.1, 0.1)
        delta = np.random.uniform(-0.1, 0.1)
        
        new_hi = h[i] - delta
        new_hj = h[j] + delta
        
        if new_hi < 0 or new_hi > 1 or new_hj < 0 or new_hj > 1:
            # Try to adjust delta to be valid
            # If invalid, skip or shrink delta
            # Shrink delta
            max_d = min(h[i], 1 - h[j])
            min_d = max(-h[i], h[j] - 1) # -delta < h[i] => delta > -h[i], delta < 1-h[j]
            # Actually:
            # h[i] - delta >= 0 => delta <= h[i]
            # h[i] - delta <= 1 => delta >= h[i] - 1
            # h[j] + delta >= 0 => delta >= -h[j]
            # h[j] + delta <= 1 => delta <= 1 - h[j]
            
            lo = max(0 - h[j], h[i] - 1) # Wait.
            # delta >= -h[j]
            # delta >= h[i] - 1
            # delta <= h[i]
            # delta <= 1 - h[j]
            
            lo = max(-h[j], h[i] - 1)
            hi = min(h[i], 1 - h[j])
            
            if lo >= hi:
                continue # No valid delta
            delta = np.random.uniform(lo, hi)
            if abs(delta) < 1e-9:
                continue
            
            new_hi = h[i] - delta
            new_hj = h[j] + delta

        # Apply move
        h[i] = new_hi
        h[j] = new_hj
        
        new_cost = compute_cost(h)
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_h = h.copy()
        else:
            # Revert
            h[i] += delta
            h[j] -= delta
            
        T *= cooling_rate
        
    return best_h

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 110  # safety margin
    
    def overlap_cost(h):
        # h has length N, values in [0,1], sum = N/2
        # Compute max overlap with shifted 1-h
        max_ov = 0.0
        for k in range(1, N):
            # overlap at shift k: sum_i h[i] * (1 - h[(i-k) % N])
            # = sum h[i] - sum h[i]*h[(i-k)%N]
            # = N/2 - autocorr[k]
            autocorr = np.dot(h, np.roll(h, k))
            ov = N / 2.0 - autocorr
            if ov > max_ov:
                max_ov = ov
        return max_ov
    
    # Start with alternating 0,1 pattern
    h = np.zeros(N)
    h[0::2] = 1.0
    h[1::2] = 0.0
    best_h = h.copy()
    best_cost = overlap_cost(h)
    
    # Simulated annealing
    temp = 0.1
    cooling = 0.995
    np.random.seed(42)
    
    while time.time() < deadline and temp > 1e-6:
        # Perturb: pick a random index and adjust
        idx = np.random.randint(N)
        delta = np.random.uniform(-0.2, 0.2)
        new_h = h.copy()
        new_h[idx] = np.clip(h[idx] + delta, 0.0, 1.0)
        
        # Adjust another index to maintain sum = N/2
        if np.abs(np.sum(new_h) - N / 2.0) > 1e-10:
            # Find another index to compensate
            others = np.arange(N)
            others = others[others != idx]
            # Distribute the correction
            diff = N / 2.0 - np.sum(new_h)
            # Pick a random other index
            jdx = others[np.random.randint(len(others))]
            new_h[jdx] = np.clip(new_h[jdx] + diff, 0.0, 1.0)
            # If still not exact, fix by adjusting idx
            remaining = N / 2.0 - np.sum(new_h)
            new_h[idx] = np.clip(new_h[idx] + remaining, 0.0, 1.0)
        
        if np.abs(np.sum(new_h) - N / 2.0) > 1e-8:
            temp *= cooling
            continue
        
        new_cost = overlap_cost(new_h)
        
        if new_cost < best_cost:
            best_h = new_h.copy()
            best_cost = new_cost
        
        if new_cost < best_cost or np.random.random() < np.exp(-(new_cost - best_cost) / max(temp, 1e-10)):
            h = new_h.copy()
        
        temp *= cooling
    
    # Final refinement: project to exact sum
    best_h = np.clip(best_h, 0.0, 1.0)
    # Ensure exact sum
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-10:
        # Distribute evenly
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        # Final correction
        diff2 = N / 2.0 - np.sum(best_h)
        if abs(diff2) > 1e-12:
            # Add to first element
            best_h[0] = np.clip(best_h[0] + diff2, 0.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_overlap(h):
    """Compute the maximum circular overlap of h with shifted (1-h)."""
    N = len(h)
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for k in range(N):
        # Circular shift of one_minus_h by k
        shifted = np.roll(one_minus_h, k)
        ov = np.sum(h * shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _local_search(h, n_steps=500, deadline=None):
    """Simple local search: randomly perturb and keep if better."""
    N = len(h)
    current = h.copy()
    current_score = _max_overlap(current)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        # Random perturbation: pick a random index and adjust
        i = np.random.randint(N)
        delta = np.random.uniform(-0.1, 0.1)
        new_h = current.copy()
        new_h[i] = np.clip(current[i] + delta, 0.0, 1.0)
        
        # Adjust to maintain sum = N/2
        diff = N/2.0 - np.sum(new_h)
        if abs(diff) > 1e-10:
            # Distribute the difference to a random other index
            j = np.random.randint(N)
            if j == i:
                j = (i + 1) % N
            new_h[j] = np.clip(new_h[j] + diff, 0.0, 1.0)
            # If clipping caused issues, try to fix
            if not (0.0 <= new_h[j] <= 1.0):
                continue
        
        new_score = _max_overlap(new_h)
        if new_score < current_score:
            current = new_h
            current_score = new_score
    
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Start with a sinusoidal pattern
    h = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(N) / N)
    # Ensure exact sum
    h = h - np.mean(h) + 0.5
    h = np.clip(h, 0.0, 1.0)
    # Fix sum exactly
    h = h - (np.sum(h) - N/2.0) / N
    
    # Local search
    h, score = _local_search(h, n_steps=1000, deadline=deadline)
    
    # Final verification
    h = np.clip(h, 0.0, 1.0)
    h = h - (np.sum(h) - N/2.0) / N
    h = np.clip(h, 0.0, 1.0)
    
    return h
# EVOLVE-BLOCK-END

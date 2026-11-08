import numpy as np
import time

N = 100

def compute_overlap(h):
    """Compute overlap at all shifts. h is array of length N."""
    B = 1.0 - h
    overlaps = np.zeros(N)
    for k in range(N):
        shifted_B = np.roll(B, -k)
        overlaps[k] = np.sum(h * shifted_B)
    return overlaps

def score(h):
    """Return the max overlap."""
    return np.max(compute_overlap(h))

def construct():
    start = time.time()
    deadline = 140.0  # safety margin
    
    # Start with uniform
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = score(h)
    
    # Try some structured patterns
    # Pattern 1: alternating blocks
    # Pattern 2: sinusoidal perturbation
    
    # Let's try gradient-free optimization with time budget
    import random
    random.seed(42)
    
    # Try small perturbations
    current_h = h.copy()
    current_score = best_score
    
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        # Random perturbation
        idx = random.randint(0, N-1)
        delta = random.uniform(-0.1, 0.1)
        new_h = current_h.copy()
        new_h[idx] = np.clip(new_h[idx] + delta, 0, 1)
        
        # Adjust another index to maintain sum
        diff = 50.0 - np.sum(new_h)
        if abs(diff) > 1e-9:
            # Find another index to adjust
            other_idx = random.randint(0, N-1)
            if other_idx == idx:
                other_idx = (other_idx + 1) % N
            adj = np.clip(new_h[other_idx] - diff, 0, 1)
            new_h[other_idx] = adj
            # Re-check sum
            diff2 = 50.0 - np.sum(new_h)
            if abs(diff2) > 1e-6:
                continue
        
        s = score(new_h)
        if s < current_score - 1e-9:
            current_h = new_h
            current_score = s
            if s < best_score:
                best_score = s
                best_h = new_h.copy()
    
    return best_h

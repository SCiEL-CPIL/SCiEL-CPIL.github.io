"""Erdos minimum overlap: minimize max(correlate(h, 1-h, full))/N*2 via binary local search."""
import numpy as np
import time

N = 100
DEADLINE = 145.0  # Safety margin below 150s kill
_START = time.time()


def remaining_time():
    return DEADLINE - (time.time() - _START)


def objective(h):
    """Compute the raw value (lower is better)."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def construct():
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)  # = 1.0
    
    # 1. Try structured binary candidates
    candidates = []
    
    # Alternating 1, 0, 1, 0...
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    candidates.append(h_alt)
    
    # Blocks of 1, 0, 1, 0... (e.g., 5 ones, 5 zeros)
    for block_size in [2, 3, 4, 5, 7, 10]:
        h_block = np.tile([1.0] * block_size + [0.0] * block_size, N // (2 * block_size) + 1)[:N]
        # Adjust to ensure sum is N/2 if possible, otherwise keep as is for evaluation
        # The sum constraint is strict. If we use binary, sum must be exactly N/2.
        # If N is even, we can have N/2 ones and N/2 zeros.
        if h_block.sum() != N / 2.0:
            # Adjust: flip bits to match sum
            # This is a bit complex for a simple candidate, so let's just evaluate 
            # the "closest" valid binary vector or skip if sum is far off.
            # Actually, let's just generate random binary vectors with exactly N/2 ones.
            pass
        candidates.append(h_block)
        
    # 2. Random binary vectors with exactly N/2 ones
    rng = np.random.default_rng(42)
    for _ in range(50):
        h_rand = np.zeros(N)
        ones_indices = rng.choice(N, N // 2, replace=False)
        h_rand[ones_indices] = 1.0
        candidates.append(h_rand)

    # Evaluate all candidates
    for h_cand in candidates:
        if not np.all(np.isfinite(h_cand)):
            continue
        if h_cand.sum() != N / 2.0:
            continue
        val = objective(h_cand)
        if val < best_val:
            best_val = val
            best_h = h_cand.copy()

    # 3. Local Search (Hill Climbing / Simulated Annealing) on binary states
    # Start from the best candidate found so far
    current_h = best_h.copy()
    current_val = best_val
    
    # We will perform a random walk on the hypercube of binary vectors with fixed weight N/2.
    # Move: pick a random 1 and a random 0, swap them.
    # This preserves the sum constraint and bounds.
    
    # Simulated Annealing parameters
    T_start = 0.01
    T_end = 1e-6
    alpha = 0.999
    
    # Estimate number of steps we can afford
    # Each step is O(N) for correlation. N=100.
    # np.correlate is fast. Let's do as many as possible.
    
    h_curr = current_h.copy()
    val_curr = current_val
    
    T = T_start
    # We don't know exactly how many steps, so we loop until time is up.
    # To avoid infinite loops if time check is slow, we can batch steps.
    
    while remaining_time() > 1.0:
        # Batch of 1000 steps
        for _ in range(1000):
            if remaining_time() < 1.0:
                break
            
            # Pick random indices
            idx_1 = rng.integers(0, N)
            idx_0 = rng.integers(0, N)
            
            # Ensure they are distinct and have different values
            # Since we have exactly N/2 ones and N/2 zeros, 
            # we can just pick from the sets of indices where h=1 and h=0.
            # But maintaining sorted lists is overhead. 
            # Simple approach: pick random i, j. If h[i] != h[j], swap.
            # Probability of h[i] != h[j] is 1/2. So we might waste steps.
            # Better: maintain lists of ones and zeros? 
            # For N=100, just picking random i, j and checking is fine.
            
            if h_curr[idx_1] == h_curr[idx_0]:
                continue
                
            # Propose swap
            h_prop = h_curr.copy()
            h_prop[idx_1], h_prop[idx_0] = h_prop[idx_0], h_prop[idx_1]
            
            val_prop = objective(h_prop)
            
            # Accept if better or with probability exp(-(val_prop - val_curr)/T)
            if val_prop < val_curr:
                h_curr = h_prop
                val_curr = val_prop
                if val_curr < best_val:
                    best_val = val_curr
                    best_h = h_curr.copy()
            else:
                delta = val_prop - val_curr
                if delta < 1e-9:
                    continue
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    h_curr = h_prop
                    val_curr = val_prop
            
            T *= alpha
            if T < T_end:
                T = T_end
        
        # If T is very low, we might want to restart from best_h to escape local minima
        # But for now, simple SA is good.
        
    # Final sanity check
    # The problem requires h in [0,1] and sum = N/2.
    # Our binary search ensures this.
    assert np.all(np.isfinite(best_h)), "Non-finite values in result"
    assert np.all(best_h >= -1e-10) and np.all(best_h <= 1.0 + 1e-10), "Out of bounds"
    assert abs(best_h.sum() - N / 2.0) < 1e-9 * N, "Sum constraint violated"
    
    return best_h


# EVOLVE-BLOCK-START
# (construct is already defined above)
# EVOLVE-BLOCK-END

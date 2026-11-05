"""Low discrepancy on all arithmetic progressions - Simulated Annealing with Vectorized Evaluation."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: list or array of +1/-1 indexed 0..n-1.
    Returns the max absolute sum.
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int16)
    
    max_abs = 0
    # We iterate over step sizes d.
    # For a given d, the progressions are determined by start index i (0 <= i < d) 
    # and length L.
    # An AP starting at i with step d is s[i], s[i+d], s[i+2d], ...
    # We want max |sum_{k=0}^{L-1} s[i + k*d]| for all valid i, L.
    
    # Optimization:
    # For a fixed d, we can process all starting indices i in [0, d) simultaneously.
    # The maximum length for a given i is floor((n - 1 - i) / d) + 1.
    # Let's create a matrix M where M[i, k] = s[i + k*d] if valid, else 0.
    # Then prefix sums along axis 1 give us sums of APs starting at i.
    # Specifically, prefix_sum[i, j] = sum_{k=0}^j M[i, k].
    # The max absolute sum for start i is max_j |prefix_sum[i, j]|.
    # The overall max is max_i max_j |prefix_sum[i, j]|.
    
    # We can vectorize this for a given d.
    # Max possible length for any start is ceil(n/d).
    # Let max_len = (n - 1) // d + 1.
    
    for d in range(1, n + 1):
        max_len = (n - 1) // d + 1
        # Number of valid starts is d (0 to d-1).
        # However, not all starts have length max_len.
        # We can create indices for a "strided" view.
        # Indices: 
        # Row i (0..d-1), Col k (0..max_len-1)
        # Index = i + k*d
        # This is valid if i + k*d < n.
        
        # Create index matrix
        # i = np.arange(d)[:, None]
        # k = np.arange(max_len)[None, :]
        # idx = i + k*d
        
        # Instead of explicit index matrix creation which might be slow for large d,
        # we can just loop over d and use slicing if possible, but d varies.
        # For small d, vectorization is key. For large d, number of APs is small.
        
        if d > 64:
            # For large d, the number of progressions is small.
            # Just loop through starts.
            for i in range(d):
                if i >= n: break
                # Extract sequence
                seq = s[i:n:d]
                if len(seq) == 0: continue
                # Cumulative sum
                cumsum = np.cumsum(seq)
                current_max = np.max(np.abs(cumsum))
                if current_max > max_abs:
                    max_abs = current_max
        else:
            # Vectorized approach for small/medium d
            # Construct indices
            i = np.arange(d)[:, None]
            k = np.arange(max_len)[None, :]
            idx = i + k * d
            
            # Mask invalid indices
            valid = idx < n
            
            # We need to handle the fact that some rows are shorter.
            # We can set invalid entries to 0, but we must be careful with cumsum.
            # If we set invalid to 0, cumsum will include 0s, which is fine for sum,
            # BUT we only care about prefixes that are valid.
            # Actually, if we set invalid to 0, the cumsum for a valid prefix is correct.
            # The cumsum for an invalid position (beyond the end of the sequence for that start)
            # will be the sum of the valid part + 0s, so it equals the sum of the valid part.
            # Thus, max(abs(cumsum)) over the whole row will correctly capture the max
            # absolute sum of any valid prefix, because the "tail" values are just repeats
            # of the final sum (or sum of valid part), which is a valid prefix sum (the full length).
            
            # However, we must ensure we don't access out of bounds.
            # np.take or fancy indexing with out-of-bounds will fail or wrap.
            # We need to clamp indices or use a safe gather.
            
            # Safe gather:
            # Create a flat array of 0s? No, just use np.where or similar?
            # Actually, we can just create a zero-padded array and use advanced indexing.
            # Or, simpler:
            # s_flat = s
            # We can't directly index with out-of-bounds.
            
            # Let's use a masked approach or just clip indices to n-1 and then mask the result?
            # If we clip, we add s[n-1] multiple times? No, that's wrong.
            
            # Alternative:
            # For each d, we can create a 2D array by slicing if d is small?
            # No, slicing s[i:n:d] creates a view.
            # We can stack these views? They have different lengths.
            # np.stack requires same shape.
            
            # Let's stick to the loop for simplicity and correctness, but optimize the inner loop with NumPy.
            # The previous "if d > 64" branch already does this.
            # Let's just use the loop for all d, but it's fast enough?
            # n=256.
            # d=1: 1 start, len 256. 1 cumsum.
            # d=2: 2 starts, len 128. 2 cumsums.
            # ...
            # d=256: 256 starts, len 1. 256 cumsums (trivial).
            # Total operations ~ sum_{d=1}^{256} d * (n/d) = sum n = n^2 = 65536.
            # This is very fast in Python/NumPy.
            
            # So we can just use the loop for all d.
            for i in range(d):
                if i >= n: break
                seq = s[i:n:d]
                if len(seq) == 0: continue
                cumsum = np.cumsum(seq)
                current_max = np.max(np.abs(cumsum))
                if current_max > max_abs:
                    max_abs = current_max
                    # Early exit if we reach a high bound? No, need exact max.
    
    return int(max_abs)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    n = 256
    rng = np.random.default_rng(42)
    
    # Initial random signs
    signs = rng.choice([-1, 1], size=n).astype(np.int16)
    
    # Evaluate initial
    current_disc = _max_ap_discrepancy(signs)
    best_signs = signs.copy()
    best_disc = current_disc
    
    # Simulated Annealing Parameters
    # Start with a high temperature, decay geometrically
    # We want to explore the space significantly.
    # Each flip changes the discrepancy by at most the number of APs containing that index.
    # An index i is in APs with various steps.
    # Max change in discrepancy is bounded by n (sum of all signs).
    # Let's set T0 = 10.0, T_min = 0.1
    T0 = 10.0
    T_min = 0.1
    # Number of iterations? We don't know how many we can do.
    # Let's do iterations until time limit.
    
    # Pre-calculate indices for quick flipping?
    # Flipping sign at index i:
    # signs[i] *= -1
    
    # To speed up evaluation, we could cache something, but n=256 is small.
    # _max_ap_discrepancy takes ~1-2 ms? Let's assume 2ms.
    # 50s / 0.002s = 25,000 iterations.
    
    # Let's run a loop
    # We use a geometric cooling schedule based on iteration count, 
    # but since we don't know the count, we can just use a fixed number of "steps" 
    # if we estimate speed, or just decay T based on time?
    # Better: Fixed number of iterations if fast, else stop.
    # Let's try 10,000 iterations first.
    
    max_iters = 10000
    # Adjust max_iters if we have more time?
    # For now, fixed max_iters with time check.
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Decay temperature
        # Simple linear or exponential decay
        # T = T0 * (T_min/T0) ** (it / max_iters)
        T = T0 * (T_min / T0) ** (it / max_iters)
        
        # Pick a random index to flip
        idx = rng.integers(0, n)
        
        # Flip
        old_sign = signs[idx]
        signs[idx] = -old_sign
        
        # Evaluate new discrepancy
        new_disc = _max_ap_discrepancy(signs)
        
        # Accept or reject
        if new_disc < current_disc:
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs.copy()
        else:
            # Boltzmann acceptance
            delta = new_disc - current_disc
            # If delta is 0, always accept? Or with prob 1?
            # If delta > 0, accept with prob exp(-delta / T)
            if delta == 0:
                # Accept to move around plateau
                pass 
            else:
                prob = np.exp(-delta / T)
                if rng.random() > prob:
                    # Reject: revert
                    signs[idx] = old_sign
                    continue
                else:
                    # Accept worse move
                    current_disc = new_disc
        
    # Return best found
    return best_signs.tolist()

# EVOLVE-BLOCK-END

"""Low discrepancy on all arithmetic progressions - Global Batch Consensus with Multi-Restart Polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    Returns (max_abs, d, i).
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int16)
    
    max_abs = 0
    best_d = 1
    best_i = 0
    
    for d in range(1, n + 1):
        for i in range(min(d, n)):
            seq = s[i:n:d]
            if len(seq) == 0:
                continue
            cumsum = np.cumsum(seq)
            current_max = np.max(np.abs(cumsum))
            if current_max > max_abs:
                max_abs = current_max
                best_d = d
                best_i = i
    return int(max_abs), best_d, best_i

def _get_top_k_worst_aps(signs, k=10):
    """
    Identify the top K APs with the highest discrepancy.
    Returns a list of (d, i, abs_sum, signed_sum) tuples.
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int16)
    
    candidates = []
    # We can't check all ~32k APs every time if we want speed, 
    # but n=256 is small enough that checking all is ~1-2ms.
    # Let's check all to be safe and accurate.
    for d in range(1, n + 1):
        for i in range(min(d, n)):
            seq = s[i:n:d]
            if len(seq) == 0:
                continue
            total_sum = np.sum(seq)
            abs_sum = abs(total_sum)
            # We care about total sum discrepancy primarily for the "batch" logic,
            # but the metric is max prefix sum. 
            # For simplicity in the batch consensus, we use the total sum as a proxy for "conflict intensity".
            # If total sum is 0, prefix sums might still be high, but usually total sum correlates.
            # Actually, let's stick to the metric: max prefix sum.
            # But calculating max prefix sum for all is the bottleneck.
            # Let's optimize: only compute max prefix if abs(total) is large? No, that's risky.
            # Let's just compute max prefix sum for all.
            cumsum = np.cumsum(seq)
            max_prefix = np.max(np.abs(cumsum))
            candidates.append((max_prefix, d, i, total_sum))
            
    # Sort by discrepancy descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:k]

def _batch_consensus_update(signs, worst_aps, n=256):
    """
    Attempt to reduce discrepancy by flipping signs in the union of indices 
    covered by the worst APs.
    
    Strategy:
    1. Collect all indices involved in the worst APs.
    2. For each index in the union, determine the "pressure" from each AP.
       If an AP has a positive sum, we want to flip +1s to -1s in that AP.
       If an AP has a negative sum, we want to flip -1s to +1s in that AP.
    3. For each index, count how many "votes" it gets to flip.
    4. Flip indices where the vote count exceeds a threshold (e.g., majority of APs covering it).
    """
    s = np.asarray(signs, dtype=np.int16)
    
    # Map index to list of (ap_index, desired_flip_direction)
    # desired_flip_direction: 
    #   If AP sum > 0, we want to decrease sum -> flip +1 to -1. So we vote for flipping +1s.
    #   If AP sum < 0, we want to increase sum -> flip -1 to +1. So we vote for flipping -1s.
    #   If AP sum == 0, no preference from total sum (though prefix might be bad).
    
    votes = np.zeros(n, dtype=np.int8)
    
    for _, d, i, total_sum in worst_aps:
        if total_sum == 0:
            continue
            
        # Determine which signs we want to flip in this AP
        # If total_sum > 0: we have too many +1s. We want to flip +1s.
        # If total_sum < 0: we have too many -1s. We want to flip -1s.
        
        indices = range(i, n, d)
        
        if total_sum > 0:
            # Vote for flipping +1s
            for idx in indices:
                if s[idx] == 1:
                    votes[idx] += 1
        else:
            # Vote for flipping -1s
            for idx in indices:
                if s[idx] == -1:
                    votes[idx] += 1
    
    # Apply flips where vote count is high enough
    # Threshold: If an index is covered by M APs, and > M/2 vote for flip, flip it.
    # We need to know M (coverage count) for each index.
    
    coverage = np.zeros(n, dtype=np.int8)
    for _, d, i, _ in worst_aps:
        for idx in range(i, n, d):
            coverage[idx] += 1
    
    # Determine flips
    # Only flip if votes > coverage / 2
    # This is a simple majority rule.
    
    flip_mask = (votes > coverage // 2) & (coverage > 0)
    
    if not np.any(flip_mask):
        return signs, False
        
    new_signs = s.copy()
    new_signs[flip_mask] = -new_signs[flip_mask]
    
    return new_signs, True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 256
    rng = np.random.default_rng(42)
    
    best_signs = None
    best_disc = float('inf')
    
    # Multiple restarts to escape local minima
    # We will run several independent search processes.
    
    # Initial candidates: Random, Thue-Morse, Alternating
    thue_morse = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int16)
    alternating = np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.int16)
    
    initial_candidates = [
        rng.choice([-1, 1], size=n).astype(np.int16),
        thue_morse,
        alternating,
        rng.choice([-1, 1], size=n).astype(np.int16)
    ]
    
    for init_signs in initial_candidates:
        if time.time() > deadline:
            break
            
        signs = init_signs.copy()
        current_disc, _, _ = _get_max_ap_discrepancy(signs)
        
        if current_disc < best_disc:
            best_disc = current_disc
            best_signs = signs.copy()
            
        # Phase 1: Global Batch Consensus
        # Iterate: Find worst APs, apply batch flips, check improvement.
        # Stop if no improvement or time runs out.
        
        no_improve_count = 0
        max_batch_iters = 200 # Cap to ensure we don't spend too much on one restart
        
        for _ in range(max_batch_iters):
            if time.time() > deadline:
                break
                
            # 1. Find worst APs
            worst_aps = _get_top_k_worst_aps(signs, k=10)
            
            # 2. Apply batch update
            new_signs, changed = _batch_consensus_update(signs, worst_aps, n)
            
            if not changed:
                # Try a random perturbation if stuck in consensus
                # Flip 1-3 random signs
                num_flips = rng.integers(1, 4)
                flip_indices = rng.choice(n, size=num_flips, replace=False)
                new_signs = signs.copy()
                new_signs[flip_indices] = -new_signs[flip_indices]
                
            # 3. Evaluate
            new_disc, _, _ = _get_max_ap_discrepancy(new_signs)
            
            if new_disc < current_disc:
                signs = new_signs
                current_disc = new_disc
                no_improve_count = 0
                
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs.copy()
            else:
                no_improve_count += 1
                # If stuck for too many iterations, break to next restart
                if no_improve_count > 20:
                    break
        
        # Phase 2: Local Hill Climbing Polish
        # Fine-tune with single flips to squeeze out the last bit of discrepancy.
        polish_steps = 100
        for _ in range(polish_steps):
            if time.time() > deadline:
                break
                
            current_disc, current_d, current_i = _get_max_ap_discrepancy(signs)
            
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs.copy()
                
            ap_indices = list(range(current_i, n, current_d))
            
            best_flip_idx = -1
            best_flip_disc = current_disc
            
            for idx in ap_indices:
                signs[idx] = -signs[idx]
                new_disc, _, _ = _get_max_ap_discrepancy(signs)
                if new_disc < best_flip_disc:
                    best_flip_disc = new_disc
                    best_flip_idx = idx
                    best_flip_val = signs[idx]
                signs[idx] = -signs[idx]
                
            if best_flip_idx != -1:
                signs[best_flip_idx] = best_flip_val
                current_disc = best_flip_disc
            else:
                # Local optimum reached
                break
    
    # Final safety check
    if best_signs is None:
        return [1] * n
        
    # Ensure we return a list of ints
    return best_signs.tolist()

# EVOLVE-BLOCK-END

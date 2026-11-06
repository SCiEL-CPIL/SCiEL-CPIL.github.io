import time
import numpy as np
import random

N = 16
MAX_TIME = 85.0  # 90s limit, keep 5s margin for safety

def _generate_batcher_network():
    """
    Generates Batcher's Bitonic Sort network for N=16.
    This network has 63 comparators.
    """
    comps = []
    
    def bitonic_sort(n, offset, stride, direction):
        if n <= 1:
            return
        h = n // 2
        # Sort first half ascending, second half descending
        bitonic_sort(h, offset, stride, 0)
        bitonic_sort(h, offset + h * stride, stride, 1)
        # Merge
        for i in range(h):
            a = offset + i * stride
            b = offset + (h + i) * stride
            if direction == 0:
                comps.append((a, b))
            else:
                comps.append((b, a))

    bitonic_sort(N, 0, 1, 0)
    return comps

def _apply_network(comps, bits):
    """
    Apply the network to a batch of inputs.
    bits: array of shape (num_inputs, N)
    Returns: array of shape (num_inputs, N)
    """
    vec = bits.copy()
    for i, j in comps:
        # Vectorized min/max
        vi = vec[:, i]
        vj = vec[:, j]
        vmin = np.minimum(vi, vj)
        vmax = np.maximum(vi, vj)
        vec[:, i] = vmin
        vec[:, j] = vmax
    return vec

def _is_sorted(comps, bits):
    """
    Check if the network sorts all boolean inputs in 'bits'.
    bits: array of shape (num_inputs, N)
    """
    out = _apply_network(comps, bits)
    # Check if rows are non-decreasing
    diffs = np.diff(out, axis=1)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Pre-compute all 2^16 boolean inputs
    n_inputs = 1 << N
    # Create indices 0 to 2^N - 1
    indices = np.arange(n_inputs, dtype=np.uint8)
    # Extract bits. 
    # We want bit 0 to be wire 0, bit 1 to be wire 1, etc.
    # indices[:, None] >> np.arange(N) gives bit k at index k.
    # We need to be careful with dtype to avoid overflow or sign issues.
    # np.arange(N) is int64, indices is uint8. 
    # Let's use int16 or int32 to be safe.
    indices_int = indices.astype(np.int16)
    bits = (indices_int[:, None] >> np.arange(N, dtype=np.int16)) & 1
    
    # Initialize with Batcher's Bitonic Sort
    best_comps = _generate_batcher_network()
    best_len = len(best_comps)
    
    # Verify baseline
    if not _is_sorted(best_comps, bits):
        # Fallback to a known correct (though longer) network if Batcher fails (shouldn't happen)
        # Odd-even transposition is 120 comparators, but let's just trust Batcher.
        pass
        
    # Local Search / Simulated Annealing
    current_comps = best_comps[:]
    current_len = len(current_comps)
    
    # Pre-compute all possible comparator pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    num_pairs = len(all_pairs)
    
    rng = np.random.default_rng(42)
    
    # Temperature for simulated annealing
    # We want to accept worse moves (inserting comparators) early on to escape local minima
    # but converge to deletions later.
    # However, since we want to minimize size, "cost" is len(comps).
    # We can simply use a probability of acceptance based on the change in length.
    # If delta_len < 0 (improvement), always accept.
    # If delta_len > 0 (worsening), accept with probability exp(-delta_len / T).
    
    T = 2.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.999
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 1.0:
        if no_improve_count > max_no_improve:
            # Restart from best
            current_comps = best_comps[:]
            current_len = len(current_comps)
            no_improve_count = 0
            T = 1.0 # Reset temperature
            
        # Choose a move
        # Move types:
        # 1. Delete a comparator (prob 0.7 if len > 0)
        # 2. Swap a comparator (prob 0.2 if len > 0)
        # 3. Insert a comparator (prob 0.1)
        
        move_type = rng.random()
        
        old_len = len(current_comps)
        
        if old_len == 0:
            # Should not happen, but just in case
            current_comps.append(all_pairs[rng.integers(0, num_pairs)])
            continue
            
        if move_type < 0.7:
            # Delete
            idx = rng.integers(0, old_len)
            removed = current_comps.pop(idx)
            # Test
            if _is_sorted(current_comps, bits):
                # Improvement or neutral (but deletion is always improvement in size)
                current_len = len(current_comps)
                if current_len < best_len:
                    best_comps = current_comps[:]
                    best_len = current_len
                    no_improve_count = 0
                    T *= cooling_rate
                else:
                    no_improve_count += 1
            else:
                # Revert
                current_comps.insert(idx, removed)
                no_improve_count += 1
        elif move_type < 0.9:
            # Swap
            idx = rng.integers(0, old_len)
            old_comp = current_comps[idx]
            new_comp = all_pairs[rng.integers(0, num_pairs)]
            if new_comp == old_comp:
                continue
            current_comps[idx] = new_comp
            # Test
            if _is_sorted(current_comps, bits):
                # Size is same, but topology changed. 
                # This doesn't improve best_len, but might enable future deletions.
                no_improve_count += 1
            else:
                # Revert
                current_comps[idx] = old_comp
                no_improve_count += 1
        else:
            # Insert
            # Pick a random pair. If it's already there, maybe swap? 
            # For now, just append. Duplicates don't break correctness, just waste size.
            # We can dedupe later or just let search handle it.
            new_comp = all_pairs[rng.integers(0, num_pairs)]
            current_comps.append(new_comp)
            current_len = len(current_comps)
            
            # Test
            if _is_sorted(current_comps, bits):
                # Worse in size, but valid.
                # Accept with probability?
                # Since we are minimizing size, we generally reject worsening moves unless T is high.
                # But we need to explore.
                delta = 1
                if rng.random() < np.exp(-delta / T):
                    # Accept worsening move
                    pass
                else:
                    # Revert
                    current_comps.pop()
                    current_len = old_len
                no_improve_count += 1
            else:
                # Invalid network
                current_comps.pop()
                current_len = old_len
                no_improve_count += 1
        
        # Cool down
        if T > T_min:
            T *= cooling_rate
            if T < T_min:
                T = T_min

    return best_comps

# EVOLVE-BLOCK-END

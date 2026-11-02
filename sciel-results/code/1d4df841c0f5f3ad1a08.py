"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # 1. Generate a broad candidate pool
    # We include vectors of weight 1, 2, 3, and 4 to give SA more flexibility.
    # Weight 1: 11 * 2 = 22
    # Weight 2: 55 * 4 = 220
    # Weight 3: 165 * 8 = 1320
    # Weight 4: 330 * 16 = 5280
    # Total ~ 6800 candidates. This is manageable for bitsets.
    
    candidates = []
    for w in [1, 2, 3, 4]:
        if w > D:
            continue
        for indices in itertools.combinations(range(D), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = [0] * D
                for idx, sign in zip(indices, signs):
                    v[idx] = sign
                candidates.append(v)
                
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # 2. Precompute norms and conflict bitsets
    # Norm squared for each candidate
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Compute conflict bitsets
    # adj[i] is a bitset of j such that i and j conflict.
    # Conflict: dist_sq(i,j) < max(norm_sq[i], norm_sq[j])
    # dist_sq = n_i + n_j - 2 dot(i,j)
    # n_i + n_j - 2 dot < max(n_i, n_j)
    # Let M = max(n_i, n_j), m = min(n_i, n_j)
    # M + m - 2 dot < M  =>  m < 2 dot  =>  dot > m/2
    # Since dot is integer:
    # If m is even, dot >= m/2 + 1
    # If m is odd, dot >= (m+1)/2
    
    # We can compute this efficiently.
    # However, n_cand is ~7000. 7000x7000 matrix is 49M elements. 
    # Memory: 49M * 8 bytes (float64) = 400MB. Might be tight but okay?
    # Or we can do it in chunks to save memory?
    # Or use the property that most vectors have small weights, so dot products are small integers.
    
    # Let's try to be memory efficient.
    # adj_bits = [0] * n_cand
    # for i in range(n_cand):
    #     # Compute dot products with all j
    #     dots = candidates @ candidates[i]
    #     # Determine conflicts
    #     # ...
    
    # Actually, 7000x7000 int64 matrix is 392 MB. NumPy can handle this usually.
    # Let's compute the full Gram matrix in chunks to avoid peak memory spikes if needed, 
    # but let's try the direct way first. If it crashes, we'll chunk.
    # For safety, let's chunk the Gram matrix computation.
    
    adj_bits = [0] * n_cand
    
    # We process rows in chunks of 100 to keep memory usage low and allow for progress.
    chunk_size = 200
    for start in range(0, n_cand, chunk_size):
        end = min(start + chunk_size, n_cand)
        # G_chunk: (end-start) x n_cand
        G_chunk = candidates[start:end] @ candidates.T
        # norms_chunk: (end-start,)
        n_chunk = norms_sq[start:end]
        # norms_all: (n_cand,)
        n_all = norms_sq
        
        # For each row i in chunk:
        # Conflict with j if:
        # dist_sq = n_i + n_j - 2 dot < max(n_i, n_j)
        
        for idx in range(end - start):
            i = start + idx
            n_i = n_chunk[idx]
            dots = G_chunk[idx]
            
            # Vectorized conflict check
            # M = max(n_i, n_all)
            # m = min(n_i, n_all)
            # Condition: dots > m/2
            
            # To avoid division, use: 2 * dots > m
            # m = np.where(n_all > n_i, n_i, n_all)
            m = np.minimum(n_i, n_all)
            
            conflict_mask = 2 * dots > m
            # Exclude self
            conflict_mask[i] = False
            
            # Convert mask to bitset
            # np.where returns indices. 
            # For 7000 bits, we can use a Python int.
            # bits = 0
            # for j in np.where(conflict_mask)[0]:
            #     bits |= (1 << j)
            
            # Faster: 
            # We can use a helper to convert boolean array to int bitset?
            # Or just iterate if the number of conflicts is small?
            # Average degree might be high.
            
            # Let's use a simple loop for bit setting, it's in Python but C-level numpy operations are faster for the mask.
            # Actually, iterating over np.where is okay for ~7000 items.
            
            # Optimization: 
            # We can pack the boolean array into bytes and then to int?
            # No, Python int from bytes is big-endian. 
            # Let's just iterate. 7000 * 7000 / 200 chunks = 245k iterations of inner loop. 
            # Each inner loop does ~7000 checks? No, np.where is fast.
            
            conflict_indices = np.where(conflict_mask)[0]
            bits = 0
            for j in conflict_indices:
                bits |= (1 << j)
            adj_bits[i] = bits
            
        if time.time() > deadline - 60:
            # If we are running out of time just to build the graph, 
            # we might want to abort and use a smaller pool?
            # But let's assume it finishes.
            pass

    # 3. Initial Solution: Greedy Insertion
    # Start with empty set.
    # To make it faster, we can start with a random subset or just empty.
    # Let's start with a greedy pass: 
    # Order candidates by norm (smaller first? larger first?)
    # Smaller norms are "easier" to fit? 
    # Actually, let's just pick random candidates and insert if valid.
    
    current_set = set()
    # Greedy initialization
    # Shuffle indices
    indices = list(range(n_cand))
    random.shuffle(indices)
    
    for idx in indices:
        if time.time() > deadline - 30:
            break
        # Check if idx is compatible with all in current_set
        # Conflict if adj_bits[idx] & (bitset of current_set) != 0
        # We need to maintain a bitset of the current set for fast intersection.
        # But current_set is a set of ints. 
        # Let's maintain current_bitset.
        
        # We'll maintain current_bitset separately.
        # But for the greedy phase, let's just do it simply.
        
        # Actually, let's initialize current_bitset = 0
        # And update it.
        
    # Let's restart the initialization with bitsets for speed.
    current_bitset = 0
    current_indices = []
    
    # Greedy Pass 1: Try to insert many
    # Sort by norm? 
    # Let's just shuffle and try.
    for idx in indices:
        if time.time() > deadline - 30:
            break
        # Check compatibility
        if (adj_bits[idx] & current_bitset) == 0:
            # Compatible
            current_bitset |= (1 << idx)
            current_indices.append(idx)
            
    # If we didn't get many, we might want to restart greedy with different order?
    # For now, proceed to SA.
    
    # 4. Simulated Annealing
    # State: current_bitset, current_size
    # Move: 
    #   1. Insert a random candidate not in set.
    #   2. Remove a random candidate from set.
    #   3. Swap: Remove one, Insert another.
    
    # Objective: Maximize size.
    # Score = len(current_indices)
    
    best_size = len(current_indices)
    best_bitset = current_bitset
    best_indices = list(current_indices)
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    # Number of steps? 
    # We have ~100s left.
    # Each step is O(1) for bitset ops + O(1) for random.
    # We can do millions of steps.
    
    # Let's define a helper to check if a bit is set.
    # We need to map bitset to indices for removal?
    # For removal, we pick a random index from current_indices.
    
    # To allow fast removal, we maintain a list of indices in the set.
    # When we remove, we swap with last and pop.
    
    # Let's restructure state for SA:
    # set_indices: list of indices
    # in_set: boolean array or set? 
    # Bitset is best for insertion check.
    # For removal, we need to know which indices are in the set.
    
    # Let's use a list `active_indices` and a boolean array `is_active`?
    # Or just the list and the bitset.
    # To pick a random element to remove: random.choice(active_indices).
    
    active_indices = list(current_indices)
    
    # Precompute some random candidates for insertion attempts?
    # Just use random.randint(0, n_cand-1)
    
    steps = 0
    T = T_start
    cooling_rate = 0.99999 # Very slow cooling?
    # Or time-based cooling?
    
    # Let's use a fixed number of steps based on time.
    # Est. 100 steps per microsecond? No, Python is slower.
    # Let's say 1000 steps per ms -> 1,000,000 steps per second.
    # We have 100s -> 100M steps.
    # Let's aim for 50M steps.
    
    while time.time() < deadline - 5:
        steps += 1
        
        # Pick a move
        # 70% Insert, 10% Remove, 20% Swap?
        # Actually, if we are small, Insert is best. If large, Swap is best.
        # Let's mix.
        
        move_type = random.random()
        
        if move_type < 0.6:
            # Insert
            # Pick random candidate
            idx = random.randint(0, n_cand - 1)
            if (1 << idx) & current_bitset:
                continue # Already in set
            
            # Check conflict
            if (adj_bits[idx] & current_bitset) != 0:
                continue # Conflict
            
            # Accept
            current_bitset |= (1 << idx)
            active_indices.append(idx)
            
            # Update best
            if len(active_indices) > best_size:
                best_size = len(active_indices)
                best_bitset = current_bitset
                best_indices = list(active_indices)
                
        elif move_type < 0.8:
            # Remove
            if not active_indices:
                continue
            idx = random.choice(active_indices)
            
            # Remove
            current_bitset &= ~(1 << idx)
            # Remove from list efficiently?
            # List removal is O(n). 
            # For n~1000, this is okay.
            # Find index in list
            try:
                pos = active_indices.index(idx)
                active_indices[pos] = active_indices[-1]
                active_indices.pop()
            except ValueError:
                pass # Should not happen
            
            # No cost function change other than size decrease.
            # We don't update best here.
            
        else:
            # Swap
            if not active_indices:
                continue
            # Remove random
            idx_out = random.choice(active_indices)
            # Insert random
            idx_in = random.randint(0, n_cand - 1)
            
            if idx_in == idx_out:
                continue
            if (1 << idx_in) & current_bitset:
                continue
            
            # Check if idx_in is compatible with (current_set - {idx_out})
            # current_bitset_without_out = current_bitset & ~(1 << idx_out)
            # Check adj_bits[idx_in] & current_bitset_without_out == 0
            
            temp_bitset = current_bitset & ~(1 << idx_out)
            if (adj_bits[idx_in] & temp_bitset) != 0:
                continue
            
            # Accept swap
            current_bitset = temp_bitset | (1 << idx_in)
            
            # Update list
            try:
                pos = active_indices.index(idx_out)
                active_indices[pos] = idx_in
            except ValueError:
                pass
            
            # Size unchanged, so no new best size possible from swap alone?
            # Unless we consider "quality" of packing? 
            # But score is just k. So swaps only help if they enable future insertions.
            # SA accepts swaps even if size doesn't change, to explore state space.
            # But here, we only accept if valid. 
            # Should we accept invalid swaps? No, we must maintain validity.
            # So this is a "valid swap" move.
            
        # Cooling
        # Simple geometric cooling
        T *= 0.9999
        if T < T_end:
            T = T_end
            
    # 5. Final Result
    # Use the best found set
    if not best_indices:
        # Fallback
        return np.array([candidates[i] for i in active_indices], dtype=np.int64)
        
    result = candidates[best_indices]
    return result

# EVOLVE-BLOCK-END

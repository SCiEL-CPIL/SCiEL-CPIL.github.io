import itertools
import numpy as np
import time

# Precompute for the specific task v=19, k=7, t=4
V_PRE = 19
K_PRE = 7
T_PRE = 4

# Generate all t-subsets
all_t_subsets = list(itertools.combinations(range(V_PRE), T_PRE))
t_subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
N_T = len(all_t_subsets)

# Generate all k-subsets and map to t-subset IDs
all_k_subsets = list(itertools.combinations(range(V_PRE), K_PRE))
N_K = len(all_k_subsets)

# block_t_ids[i] is the array of t-subset indices contained in k-subset i
block_t_ids = np.zeros((N_K, 35), dtype=np.int32)
for i, block in enumerate(all_k_subsets):
    for j, t_sub in enumerate(itertools.combinations(block, T_PRE)):
        block_t_ids[i, j] = t_subset_to_id[t_sub]

def construct(v, k, t):
    # Fallback for non-standard parameters
    if v != V_PRE or k != K_PRE or t != T_PRE:
        return _generic_greedy(v, k, t)
        
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_blocks = None
    best_size = float('inf')
    
    # Run multiple restarts of Greedy + Local Search
    # Each restart:
    # 1. Greedy construction
    # 2. Pruning
    # 3. Local Search (Simulated Annealing / Random Walk)
    
    for seed in range(200):
        if time.time() > deadline:
            break
            
        # 1. Greedy
        blocks = _greedy_cover(seed)
        
        # 2. Prune
        blocks = _prune(blocks)
        
        current_size = len(blocks)
        
        # 3. Local Search
        # We represent the covering as a set of block indices
        block_indices = set(blocks)
        # counts[i] is the number of blocks covering t-subset i
        counts = np.zeros(N_T, dtype=np.int32)
        for b_idx in block_indices:
            counts[block_t_ids[b_idx]] += 1
        
        # Local Search Loop
        ls_start = time.time()
        while time.time() - ls_start < 5.0: # 5s per restart
            # Try to remove a random block
            if not block_indices:
                break
            b_out = next(iter(block_indices))
            
            # Check if removal is valid
            # We need to know if any count becomes 0
            # Efficient check: 
            # Decrement counts for b_out
            t_ids_out = block_t_ids[b_out]
            new_counts = counts - 1 # Vectorized subtraction? No, we need to check only affected
            # Actually, just check if min(counts[t_ids_out]) > 1
            if np.min(counts[t_ids_out]) > 1:
                # Safe to remove
                block_indices.remove(b_out)
                counts[t_ids_out] -= 1
                current_size -= 1
                if current_size < best_size:
                    best_size = current_size
                    best_blocks = list(block_indices)
            else:
                # Try a swap
                # Pick a random block to add
                b_in = np.random.randint(N_K)
                if b_in in block_indices:
                    continue
                    
                # Check if swap is valid
                # Remove b_out, Add b_in
                t_ids_in = block_t_ids[b_in]
                
                # Temporarily apply
                counts[t_ids_out] -= 1
                counts[t_ids_in] += 1
                
                if np.min(counts) > 0:
                    # Valid swap
                    block_indices.remove(b_out)
                    block_indices.add(b_in)
                    # After swap, try to prune immediately?
                    # Or just continue.
                else:
                    # Revert
                    counts[t_ids_in] -= 1
                    counts[t_ids_out] += 1
        
        # Final Prune after LS
        final_blocks = _prune(list(block_indices))
        if len(final_blocks) < best_size:
            best_size = len(final_blocks)
            best_blocks = final_blocks
            
    if best_blocks is None:
        # Fallback to a simple greedy if LS failed or didn't run
        best_blocks = _greedy_cover(0)
        
    return [list(all_k_subsets[b]) for b in best_blocks]

def _greedy_cover(seed):
    np.random.seed(seed)
    counts = np.zeros(N_T, dtype=np.int32)
    blocks = []
    uncovered_mask = np.ones(N_T, dtype=bool)
    
    while np.any(uncovered_mask):
        # Calculate gains
        # gains[i] = number of uncovered t-subsets in block i
        # We can use: gains = np.sum(uncovered_mask[block_t_ids], axis=1)
        # But uncovered_mask is boolean.
        gains = np.sum(uncovered_mask[block_t_ids], axis=1)
        
        # Find max gain
        max_gain = np.max(gains)
        # Get indices with max gain
        candidates = np.where(gains == max_gain)[0]
        
        # Randomize among top candidates (e.g., top 10)
        if len(candidates) > 10:
             # Sort candidates by gain descending, take top 10
             # Since they all have same max gain, just pick random from candidates
             # Or pick from top K of the sorted list of all gains?
             # For now, pick random from the max gain set.
             # To improve, we could look at top 5% of gains.
             b_idx = candidates[np.random.randint(len(candidates))]
        else:
             b_idx = candidates[np.random.randint(len(candidates))]
             
        blocks.append(b_idx)
        
        # Update counts and uncovered_mask
        t_ids = block_t_ids[b_idx]
        counts[t_ids] += 1
        # Update uncovered_mask: if count becomes 1, it was 0, so now covered
        # Actually, we just need to set uncovered_mask[t_ids] = False where counts was 0
        # But we don't store previous counts easily.
        # Simpler: recompute uncovered_mask? No, slow.
        # We can just update uncovered_mask for the affected t_ids.
        # If counts[t_id] was 0, it is now 1, so uncovered_mask[t_id] = False.
        # We know which ones were 0: those where counts[t_id] == 1 now.
        newly_covered = t_ids[counts[t_ids] == 1]
        uncovered_mask[newly_covered] = False
        
    return blocks

def _prune(blocks):
    if not blocks:
        return []
    
    block_indices = set(blocks)
    counts = np.zeros(N_T, dtype=np.int32)
    for b_idx in block_indices:
        counts[block_t_ids[b_idx]] += 1
        
    pruned = True
    while pruned:
        pruned = False
        for b_idx in list(block_indices):
            t_ids = block_t_ids[b_idx]
            if np.min(counts[t_ids]) > 1:
                block_indices.remove(b_idx)
                counts[t_ids] -= 1
                pruned = True
                
    return list(block_indices)

def _generic_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks

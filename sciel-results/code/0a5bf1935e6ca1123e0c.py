import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin
    
    # 1. Precomputation
    all_t = list(itertools.combinations(range(v), t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # Precompute coverage vectors for all k-subsets
    # This is the most expensive part but enables O(1) updates in SA
    all_k = list(itertools.combinations(range(v), k))
    N_k = len(all_k)
    
    # coverage_vecs[i] is a numpy array of 0s and 1s of length N_t
    # where 1 indicates the t-subset is covered by block i
    coverage_vecs = np.zeros((N_k, N_t), dtype=np.uint8)
    
    # Map from tuple to index for fast lookup
    k_to_idx = {b: i for i, b in enumerate(all_k)}
    
    for i, b in enumerate(all_k):
        for s in itertools.combinations(b, t):
            coverage_vecs[i, t_idx[s]] = 1
            
    # 2. Initial Solution: Fast Greedy
    # We need a valid starting point.
    # Instead of the complex greedy in the seed, we use a simple randomized greedy 
    # that is fast, then let SA do the heavy lifting.
    
    def get_initial_solution():
        random.seed(42)
        uncovered = np.ones(N_t, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Pick a random uncovered t-subset
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            target_idx = uncovered_indices[random.randint(0, len(uncovered_indices)-1)]
            target_t = all_t[target_idx]
            
            # Find a block containing this t-subset
            # To be fast, we just pick a random block from all_k that contains it?
            # Or better: precompute for each t-subset, the list of blocks containing it.
            # This list might be large, so we sample.
            
            # Actually, let's just pick a random block and check if it covers any uncovered.
            # This is inefficient. 
            # Better: Precompute "blocks_containing_t" as a list of lists.
            # Memory: 11628 * ~1000 blocks? Too much.
            
            # Let's stick to the seed's greedy but simplified:
            # Pick random t-subset, pick random block containing it.
            # We need a way to find blocks containing a specific t-subset quickly.
            # We can precompute: for each t-subset, a list of block indices.
            # C(19,5) = 11628. Each t-subset is in C(19-5, 8-5) = C(14,3) = 364 blocks.
            # Total entries: 11628 * 364 ~ 4.2 million. This is fine for memory (int32).
            
            # Let's precompute this map outside the function if possible? No, must be inside.
            # But 4.2M ints is ~16MB. Acceptable.
            
            pass # Placeholder, will implement in main flow
            
    # Let's restructure: Precompute t_to_blocks
    t_to_blocks = [[] for _ in range(N_t)]
    for i, b in enumerate(all_k):
        for s in itertools.combinations(b, t):
            t_to_blocks[t_idx[s]].append(i)
            
    def greedy_start():
        random.seed(123)
        uncovered = np.ones(N_t, dtype=bool)
        blocks = []
        
        while uncovered.any():
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            target_idx = uncovered_indices[random.randint(0, len(uncovered_indices)-1)]
            
            # Pick a random block from those containing target_idx
            candidates = t_to_blocks[target_idx]
            if not candidates:
                # Should not happen for valid parameters
                break
            block_idx = candidates[random.randint(0, len(candidates)-1)]
            
            blocks.append(block_idx)
            # Update uncovered
            uncovered[coverage_vecs[block_idx] == 1] = False
            
        return blocks

    init_blocks = greedy_start()
    
    # 3. Simulated Annealing
    # State: list of block indices
    # Coverage: numpy array of counts
    # We maintain a set of block indices for O(1) duplicate check? 
    # Or just a list and a set for membership.
    
    current_blocks = list(init_blocks)
    block_set = set(current_blocks)
    
    # Initial coverage counts
    current_counts = np.zeros(N_t, dtype=np.int32)
    for b_idx in current_blocks:
        current_counts += coverage_vecs[b_idx]
        
    # Ensure validity (greedy should be valid)
    if not np.all(current_counts > 0):
        # Fallback if greedy failed (unlikely)
        return [list(all_k[b]) for b in current_blocks]
        
    best_blocks = list(current_blocks)
    best_size = len(current_blocks)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Move types:
    # 1. Remove a block (if redundant)
    # 2. Add a block (random)
    # 3. Swap a block (remove one, add another)
    
    step = 0
    no_improve = 0
    
    while time.time() < deadline:
        step += 1
        
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to remove a block
            if len(current_blocks) > 1:
                i = random.randint(0, len(current_blocks)-1)
                b_idx = current_blocks[i]
                
                # Check if removing it breaks coverage
                # new_counts = current_counts - coverage_vecs[b_idx]
                # If min(new_counts) > 0, it's a valid removal.
                # We can check: for all s in block b, current_counts[s] > 1
                # This is faster than full array subtraction.
                block_vec = coverage_vecs[b_idx]
                # Get indices where block_vec is 1
                covered_indices = np.where(block_vec == 1)[0]
                if len(covered_indices) > 0 and np.all(current_counts[covered_indices] > 1):
                    # Valid removal
                    current_blocks.pop(i)
                    block_set.discard(b_idx)
                    current_counts -= block_vec
                    
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        elif move_type < 0.7:
            # Try to add a block
            # This increases size, so only accept if it allows future removals?
            # In SA, we accept worse solutions with probability exp(-delta/T).
            # Delta = +1 (size increases).
            # This helps escape local minima where we are stuck with redundant blocks
            # that can't be removed because they are "needed" but actually a different
            # set of blocks would be better.
            
            # Pick a random block not in current_set
            # To avoid infinite loops of adding useless blocks, we should bias towards
            # blocks that cover currently low-count t-subsets.
            
            # Simple: Random block
            b_idx = random.randint(0, N_k-1)
            if b_idx not in block_set:
                # Check if this is a "good" addition?
                # Heuristic: Does it cover any t-subset with count == 1?
                # If so, it might allow removal of the unique block later.
                block_vec = coverage_vecs[b_idx]
                covered_indices = np.where(block_vec == 1)[0]
                
                # Acceptance probability
                # If it covers a critical subset, maybe higher chance?
                # Standard SA: delta = 1 (cost increase)
                prob = np.exp(-1.0 / T)
                
                if random.random() < prob:
                    current_blocks.append(b_idx)
                    block_set.add(b_idx)
                    current_counts += block_vec
                    no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        else:
            # Swap: Remove one, Add one
            if len(current_blocks) > 0:
                i = random.randint(0, len(current_blocks)-1)
                b_out_idx = current_blocks[i]
                
                # Pick b_in
                # Bias towards blocks that cover subsets currently covered only by b_out?
                # Or just random.
                b_in_idx = random.randint(0, N_k-1)
                
                if b_in_idx == b_out_idx or b_in_idx in block_set:
                    no_improve += 1
                else:
                    # Simulate swap
                    # new_counts = current_counts - vec_out + vec_in
                    # Valid if min(new_counts) > 0
                    vec_out = coverage_vecs[b_out_idx]
                    vec_in = coverage_vecs[b_in_idx]
                    
                    # Check validity efficiently
                    # Subsets covered by out but not in: must have count > 1
                    # Subsets covered by in but not out: count can be anything (will increase)
                    # Subsets covered by both: count changes by 0
                    
                    # Indices covered by out
                    out_covered = np.where(vec_out == 1)[0]
                    # Indices covered by in
                    in_covered = np.where(vec_in == 1)[0]
                    
                    # Critical check: out_covered \ in_covered
                    # We need current_counts[x] > 1 for all x in out_covered where vec_in[x] == 0
                    
                    # Efficient check:
                    # Create mask of out_covered
                    # Check counts for those where in is 0
                    
                    # Since N_t is 11628, we can just do the check on the arrays
                    # But creating temporary arrays every step is slow.
                    
                    # Alternative:
                    # 1. Subtract vec_out from a copy? No, too slow.
                    # 2. Check: for all s in out_covered, if vec_in[s]==0, current_counts[s] > 1.
                    
                    # Let's use the fact that we have the vectors.
                    # diff_mask = (vec_out == 1) & (vec_in == 0)
                    # if np.any(diff_mask):
                    #     if np.any(current_counts[diff_mask] == 1):
                    #         invalid
                    
                    # This involves boolean indexing which is fast in numpy.
                    
                    diff_mask = (vec_out == 1) & (vec_in == 0)
                    if np.any(diff_mask):
                        if np.any(current_counts[diff_mask] == 1):
                            # Invalid swap
                            no_improve += 1
                        else:
                            # Valid swap
                            current_blocks[i] = b_in_idx
                            block_set.discard(b_out_idx)
                            block_set.add(b_in_idx)
                            current_counts -= vec_out
                            current_counts += vec_in
                            no_improve = 0
                    else:
                        # Valid swap (in covers everything out covered, or more)
                        current_blocks[i] = b_in_idx
                        block_set.discard(b_out_idx)
                        block_set.add(b_in_idx)
                        current_counts -= vec_out
                        current_counts += vec_in
                        no_improve = 0
            
        # Cooling
        if no_improve > 1000:
            T *= 0.9 # Faster cooling if stuck
        else:
            T *= cooling
            
        if T < T_min:
            T = T_min
            
        # Periodic restart or reset?
        # If stuck for too long, maybe randomize?
        if no_improve > 5000:
            # Reheat?
            T = 0.5
            no_improve = 0

    # Final cleanup: remove any redundant blocks
    final_counts = np.zeros(N_t, dtype=np.int32)
    for b_idx in best_blocks:
        final_counts += coverage_vecs[b_idx]
        
    final_blocks = []
    for b_idx in best_blocks:
        vec = coverage_vecs[b_idx]
        covered = np.where(vec == 1)[0]
        if len(covered) == 0 or np.all(final_counts[covered] > 1):
            # Redundant
            final_counts -= vec
        else:
            final_blocks.append(b_idx)
            
    # Verify
    if not np.all(final_counts > 0):
        # Fallback to best_blocks
        final_blocks = best_blocks
    else:
        best_blocks = final_blocks

    return [list(all_k[b]) for b in best_blocks]

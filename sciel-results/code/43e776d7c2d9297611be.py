import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (kill at 150s)
    
    # 1. Precomputation
    # Generate all t-subsets and assign indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_t_subsets)
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Generate all k-subsets (blocks)
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    # Precompute bitmask for each block
    # bit i is 1 if the block covers the t-subset with index i
    block_bitmasks = [0] * num_blocks
    # Also store the list of indices for local search updates (optional, but bitmask is faster for checks)
    # For local search, we need to know which t-subsets are covered.
    # We can extract indices from bitmask if needed, or store list.
    # Storing list is O(C(k,t)) memory per block. C(7,4)=35. 50k * 35 is fine.
    block_tsub_indices = []
    
    for b_idx, block in enumerate(all_k_subsets):
        mask = 0
        indices = []
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_index[tsub]
            mask |= (1 << idx)
            indices.append(idx)
        block_bitmasks[b_idx] = mask
        block_tsub_indices.append(indices)
        
    # Precompute inverse: for each t-subset, which blocks cover it?
    # This is needed for local search (finding replacements).
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, indices in enumerate(block_tsub_indices):
        for s_idx in indices:
            tsub_to_blocks[s_idx].append(b_idx)
            
    # Helper: Run greedy with a specific random seed for tie-breaking
    def run_greedy(seed_val):
        rng = random.Random(seed_val)
        uncovered_mask = (1 << num_tsubs) - 1
        selected = []
        
        # Optimization: We don't need to check all blocks if we can filter them.
        # But C(19,7) is only ~50k. Scanning 50k ints is fast.
        # We can also maintain a list of "active" blocks, but since all blocks are initially active,
        # and we just skip if gain==0, a full scan is simple and fast enough.
        
        while uncovered_mask:
            if time.time() > deadline:
                return None # Timeout
            
            best_gain = 0
            best_block = -1
            # To allow random tie-breaking, we can iterate in random order?
            # Or just pick first max? 
            # Better: collect all max gains and pick random.
            # But that's expensive.
            # Alternative: Iterate, track max, if tie, randomize.
            # Simplest: Just iterate 0..N-1. If we want diversity, we can shuffle the block indices?
            # Shuffling 50k items every iteration is too slow.
            # Let's stick to deterministic greedy first, and use multiple restarts with different initial states?
            # No, greedy is deterministic if tie-breaking is fixed.
            # To get different results from restarts, we need random tie-breaking.
            # We can iterate through blocks in a random permutation?
            # Generating a random permutation of 50k ints is ~5ms. 150 iterations * 5ms = 750ms. Acceptable.
            
            # However, doing this inside the while loop is expensive.
            # Instead, we can just pick the first max.
            # To vary the result, we can try different "tie-breaking" strategies or just accept that
            # simple greedy is deterministic. 
            # But the directive says "Lazy Evaluation... avoiding overhead".
            # Let's try a simple deterministic greedy first. If it's too consistent, we can add noise.
            # Actually, randomizing the order of checking blocks is the standard way to get different greedy solutions.
            
            # Let's optimize the inner loop.
            # Instead of random permutation every time, we can just iterate.
            # If we want diversity, we can use a "Random Greedy" where we pick a random block among those with max gain.
            
            # Let's implement:
            # 1. Find max gain
            # 2. Collect all blocks with max gain
            # 3. Pick random one
            
            # This requires two passes or storing candidates.
            # Given 50k blocks, two passes are fine.
            
            # Pass 1: Find max gain
            max_gain = 0
            for b in range(num_blocks):
                if uncovered_mask == 0: break
                gain = (block_bitmasks[b] & uncovered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
            
            if max_gain == 0:
                # Should not happen if uncovered_mask != 0
                break
                
            # Pass 2: Pick random block with max_gain
            candidates = []
            for b in range(num_blocks):
                if (block_bitmasks[b] & uncovered_mask).bit_count() == max_gain:
                    candidates.append(b)
                    
            b = rng.choice(candidates)
            selected.append(b)
            uncovered_mask &= ~block_bitmasks[b]
            
        if uncovered_mask == 0:
            return selected
        else:
            return None

    best_solution = None
    best_len = float('inf')
    
    # Run multiple greedy restarts
    # How many? Each greedy takes ~0.5s? 
    # Let's estimate. 150 iterations. 50k blocks * 2 passes * bit_count.
    # bit_count is C-speed. 100k ops * 150 = 15M ops. Very fast. < 1s per greedy.
    # So we can run ~50-100 restarts in 115s? 
    # Wait, local search also takes time.
    # Let's run greedy, then local search.
    # If local search is fast, we can do many iterations.
    
    # Strategy:
    # 1. Run greedy (deterministic or random tie-break)
    # 2. Run local search
    # 3. If time left, run another greedy with different seed
    
    # Let's start with a few restarts.
    
    for seed in range(100):
        if time.time() > deadline - 5.0: # Leave 5s for final processing
            break
            
        sol = run_greedy(seed)
        if sol is None:
            continue
            
        # Local Search
        # State: selected list of block indices
        # We need cover_count to check redundancy.
        # cover_count[s] = number of blocks in selected covering s.
        
        # Initialize cover_count
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in sol:
            for s in block_tsub_indices[b]:
                cover_count[s] += 1
        
        # Redundancy check: block b is redundant if for all s in block_tsub_indices[b], cover_count[s] > 1
        # Swap: remove b, add c if c covers all s where cover_count[s] == 1 (for b)
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # 1. Remove redundant blocks
            redundant = []
            for b in sol:
                is_red = True
                for s in block_tsub_indices[b]:
                    if cover_count[s] == 1:
                        is_red = False
                        break
                if is_red:
                    redundant.append(b)
            
            if redundant:
                for b in redundant:
                    sol.remove(b)
                    for s in block_tsub_indices[b]:
                        cover_count[s] -= 1
                improved = True
                continue
            
            # 2. Try 1-for-1 swaps
            # Find block with min "unique coverage" (number of s where cover_count[s]==1)
            # If min unique is 0, it's redundant (handled above).
            
            min_unique = float('inf')
            target_block = -1
            
            for b in sol:
                u_count = 0
                for s in block_tsub_indices[b]:
                    if cover_count[s] == 1:
                        u_count += 1
                if u_count < min_unique:
                    min_unique = u_count
                    target_block = b
            
            if target_block == -1:
                break
                
            # Find replacement for target_block
            # Must cover all s where cover_count[s] == 1 and s in target_block
            unique_s_list = [s for s in block_tsub_indices[target_block] if cover_count[s] == 1]
            
            # Candidates: blocks not in sol that cover all unique_s_list
            # We can iterate through all blocks? 50k blocks.
            # Check if candidate covers unique_s_list.
            # unique_s_list is small.
            
            found_replacement = False
            # To speed up, we can precompute for each t-subset the list of blocks.
            # But here we need intersection of blocks covering all s in unique_s_list.
            
            # Simple approach: Iterate all blocks not in sol.
            sol_set = set(sol)
            
            for c in range(num_blocks):
                if c in sol_set:
                    continue
                
                # Check if c covers all s in unique_s_list
                valid = True
                for s in unique_s_list:
                    # Check if c covers s
                    # We have block_bitmasks.
                    # if (block_bitmasks[c] >> s) & 1:
                    if not (block_bitmasks[c] & (1 << s)):
                        valid = False
                        break
                
                if valid:
                    # Perform swap
                    sol.remove(target_block)
                    sol.append(c)
                    
                    # Update cover_count
                    for s in block_tsub_indices[target_block]:
                        cover_count[s] -= 1
                    for s in block_tsub_indices[c]:
                        cover_count[s] += 1
                    
                    improved = True
                    found_replacement = True
                    break
            
            if not found_replacement:
                # Try random restart of local search? Or just break?
                # If no swap found, we are stuck in local minimum.
                # We can try a "kick": randomly remove a block and add a random one?
                # Or just break and start a new greedy.
                break
                
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol[:]
            
            # If we hit the theoretical lower bound or a known good value, we could stop?
            # Lower bound is 131. Known best is 151.
            # If we find 150, that's an improvement.
            
    if best_solution is None:
        # Fallback: return empty or invalid
        return []
        
    # Convert indices to actual blocks
    result = [list(all_k_subsets[i]) for i in best_solution]
    return result

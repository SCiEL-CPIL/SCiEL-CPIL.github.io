import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit

    # 1. Generate all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # 2. Generate all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_blocks)

    # 3. Precompute mappings
    # block_to_subs[b] = list of t-subset indices covered by block b
    # sub_to_blocks[s] = list of block indices covering t-subset s
    block_to_subs = [[] for _ in range(n_blocks)]
    sub_to_blocks = [[] for _ in range(n_t)]
    
    for b_idx, block in enumerate(all_k_blocks):
        for t_sub in itertools.combinations(block, t):
            s_idx = t_sub_to_idx.get(t_sub)
            if s_idx is not None:
                block_to_subs[b_idx].append(s_idx)
                sub_to_blocks[s_idx].append(b_idx)

    # 4. Greedy Construction
    # We use a simple greedy: pick block with max gain.
    # To speed up, we maintain a 'gain' array.
    gains = np.array([len(subs) for subs in block_to_subs], dtype=np.int32)
    uncovered = np.ones(n_t, dtype=bool)
    active_set = set()
    active_list = []
    
    # Pre-sort blocks by initial gain? Or just scan.
    # Scanning is O(N_blocks) per step. N_blocks ~ 75k. Steps ~ 300. 22M ops. Acceptable.
    # But we can be smarter.
    
    while uncovered.sum() > 0:
        if time.time() > deadline - 30.0:
            break
            
        # Find block with max gain among non-active blocks
        best_b = -1
        max_gain = 0
        
        # Optimization: Only consider blocks that have positive gain
        # We can iterate over all blocks and check if they cover any uncovered subset.
        # This is expensive if done naively.
        # Let's just iterate all blocks and compute gain on the fly if not cached?
        # Better: maintain a list of candidates?
        
        # For simplicity and robustness within time limit, we scan all blocks.
        # We can skip blocks already in active_set.
        for b in range(n_blocks):
            if b in active_set:
                continue
            # Compute gain: number of uncovered subsets in block_to_subs[b]
            # This is a sum of a boolean array slice.
            # We can precompute a numpy array for each block? No, too much memory.
            # We can use the uncovered array.
            # gain = sum(uncovered[s] for s in block_to_subs[b])
            # To speed this up, we can maintain the gains array and update it.
            # But updating gains for all blocks covering a newly covered subset is expensive.
            
            # Let's stick to the simple scan but optimize the inner loop.
            # If gains[b] == 0, skip.
            if gains[b] == 0:
                continue
            
            # Recompute actual gain to be sure (gains might be stale if we don't update properly)
            # Actually, if we update gains correctly, we don't need to recompute.
            # Let's try to maintain gains correctly.
            # Initial gains are correct.
            # When we add a block b, we cover some subsets.
            # For each subset s covered by b:
            #   If s was uncovered, it becomes covered.
            #   For every block b' covering s, if s was the only uncovered subset for b'? No.
            #   Gain of b' decreases by 1 if s was uncovered.
            
            # Let's implement the update properly.
            pass # Placeholder for logic below

    # Re-implementing Greedy with proper gain updates
    active_set = set()
    active_list = []
    uncovered = np.ones(n_t, dtype=bool)
    gains = np.array([len(subs) for subs in block_to_subs], dtype=np.int32)
    
    # To speed up finding max gain, we can use a max-heap or just scan.
    # Scanning 75k items is fast in Python if simple.
    # Let's use a simple scan but break early if possible? No, need max.
    
    while uncovered.sum() > 0:
        if time.time() > deadline - 30.0:
            break
            
        best_b = -1
        max_gain = 0
        
        # Find best block
        # We can optimize by only checking blocks with gains > 0
        # But we need to find the max.
        for b in range(n_blocks):
            if b in active_set:
                continue
            if gains[b] > max_gain:
                max_gain = gains[b]
                best_b = b
        
        if best_b == -1 or max_gain == 0:
            # Should not happen if valid covering exists and we haven't covered all
            break
            
        active_set.add(best_b)
        active_list.append(best_b)
        
        # Update uncovered and gains
        for s in block_to_subs[best_b]:
            if uncovered[s]:
                uncovered[s] = False
                # Decrease gain of all blocks covering s
                for b_prime in sub_to_blocks[s]:
                    if b_prime not in active_set:
                        gains[b_prime] -= 1
                    # If b_prime is active, we don't care about its gain for selection,
                    # but it might be needed for swap logic later? No, swap logic uses counts.
                    
    # 5. Local Search: Deletion and Swap
    # We maintain coverage_counts for the active set
    coverage_counts = np.zeros(n_t, dtype=np.int32)
    for b in active_list:
        for s in block_to_subs[b]:
            coverage_counts[s] += 1
            
    best_len = len(active_list)
    best_set = list(active_list)
    
    # Helper to check if a block is redundant
    def is_redundant(b):
        for s in block_to_subs[b]:
            if coverage_counts[s] == 1:
                return False
        return True

    # Helper to find swap candidates
    def find_swap_candidate(b):
        # Find unique subsets for b
        unique_subs = []
        for s in block_to_subs[b]:
            if coverage_counts[s] == 1:
                unique_subs.append(s)
                
        if not unique_subs:
            return None # Should be handled by deletion
        
        # Sort unique subsets by size of sub_to_blocks[s] to prune early
        unique_subs.sort(key=lambda s: len(sub_to_blocks[s]))
        
        candidates = None
        for s in unique_subs:
            if candidates is None:
                candidates = set(sub_to_blocks[s])
            else:
                candidates.intersection_update(sub_to_blocks[s])
                if not candidates:
                    break
        
        if not candidates:
            return None
            
        candidates.discard(b) # Can't swap with self
        
        # Filter candidates: must not be in active_set
        for c in candidates:
            if c not in active_set:
                return c
                
        return None

    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # 1. Deletion Pass
        changed = True
        while changed:
            changed = False
            # Iterate in random order to break symmetry
            active_list_copy = list(active_list)
            random.shuffle(active_list_copy)
            
            for b in active_list_copy:
                if time.time() > deadline - 5.0:
                    break
                    
                if b not in active_set:
                    continue
                    
                if is_redundant(b):
                    # Remove b
                    active_set.remove(b)
                    active_list.remove(b)
                    for s in block_to_subs[b]:
                        coverage_counts[s] -= 1
                    changed = True
                    improved = True
                    break # Restart deletion pass
                    
        if len(active_list) < best_len:
            best_len = len(active_list)
            best_set = list(active_list)
            
        # 2. Swap Pass
        # Try to swap blocks to enable further deletions
        swaps_done = 0
        max_swaps_per_iter = 100 # Limit to avoid infinite loops in one phase
        
        active_list_copy = list(active_list)
        random.shuffle(active_list_copy)
        
        for b in active_list_copy:
            if time.time() > deadline - 5.0:
                break
            if swaps_done >= max_swaps_per_iter:
                break
                
            if b not in active_set:
                continue
                
            # Check if b is redundant (should have been caught by deletion, but just in case)
            if is_redundant(b):
                # Remove it
                active_set.remove(b)
                active_list.remove(b)
                for s in block_to_subs[b]:
                    coverage_counts[s] -= 1
                improved = True
                continue
                
            # Find swap candidate
            new_b = find_swap_candidate(b)
            if new_b is not None:
                # Perform swap
                # Remove b
                for s in block_to_subs[b]:
                    coverage_counts[s] -= 1
                active_set.remove(b)
                active_list.remove(b)
                
                # Add new_b
                for s in block_to_subs[new_b]:
                    coverage_counts[s] += 1
                active_set.add(new_b)
                active_list.append(new_b)
                
                swaps_done += 1
                improved = True
                # After a swap, structure changes, so we break to restart deletion/swap loop
                break 

    # 3. Final Cleanup
    # One last deletion pass
    changed = True
    while changed:
        changed = False
        for b in list(active_list):
            if time.time() > deadline - 1.0:
                break
            if b not in active_set:
                continue
            if is_redundant(b):
                active_set.remove(b)
                active_list.remove(b)
                for s in block_to_subs[b]:
                    coverage_counts[s] -= 1
                changed = True

    # Construct result
    # We want the best set found (lowest len)
    # active_list is the current valid set.
    # best_set might be better if we didn't converge?
    # Actually, if we found a better set during search, we saved it.
    # But active_list is always valid.
    # If best_len < len(active_list), we should return best_set?
    # No, if we improved, active_list is the improved one.
    # The 'best_set' tracking is redundant if we always maintain validity and minimize.
    # However, if a swap leads to a dead end (no deletions), we might have increased size?
    # No, swap keeps size same. Deletion decreases size.
    # So active_list size is non-increasing.
    # Thus, the final active_list is the best.
    
    result = [list(all_k_blocks[b]) for b in active_list]
    return result

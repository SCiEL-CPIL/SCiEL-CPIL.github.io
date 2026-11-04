import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Precompute all t-subsets and map to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t = len(all_t_subs)
    
    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)
    
    # Map each k-subset to a bitset of t-subset indices it contains
    # Python int can handle ~11628 bits easily
    k_sub_bitsets = [0] * n_k
    # Map each t-subset to list of k-subset indices that contain it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    
    for i, k_sub in enumerate(all_k_subs):
        bitset = 0
        for t_sub in itertools.combinations(k_sub, t):
            t_idx = t_sub_to_idx[t_sub]
            bitset |= (1 << t_idx)
            t_sub_to_k_idxs[t_idx].append(i)
        k_sub_bitsets[i] = bitset

    def run_greedy(random_seed):
        random.seed(random_seed)
        uncovered = (1 << n_t) - 1  # All t-subsets uncovered
        blocks = []
        
        while uncovered and time.time() < deadline:
            # Find the lowest set bit (first uncovered t-subset)
            # This is a heuristic: cover the "hardest" one? 
            # Actually, lowest index is arbitrary. 
            # A better heuristic: pick the uncovered t-subset that has the fewest 
            # candidate blocks remaining? 
            # For now, stick to lowest index for speed, but randomize tie-breaking in gain.
            
            # Get the index of the lowest set bit
            # (uncovered & -uncovered) gives the lowest set bit as a power of 2
            lowest_bit = uncovered & -uncovered
            t_idx = lowest_bit.bit_length() - 1
            
            # Get candidates
            candidates = t_sub_to_k_idxs[t_idx]
            
            best_gain = -1
            best_candidates = []
            
            for b_idx in candidates:
                gain = (k_sub_bitsets[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [b_idx]
                elif gain == best_gain:
                    best_candidates.append(b_idx)
            
            if not best_candidates:
                break
                
            # Random tie-breaking
            chosen_b_idx = random.choice(best_candidates)
            
            blocks.append(chosen_b_idx)
            uncovered &= ~k_sub_bitsets[chosen_b_idx]
            
        return blocks

    def reduce_blocks(blocks):
        # blocks is a list of indices
        if not blocks:
            return blocks
            
        # Initialize cover counts
        cover_count = [0] * n_t
        # Use a set for O(1) lookup of blocks
        block_set = set(blocks)
        
        for b_idx in blocks:
            # Iterate through set bits of the bitset to update counts
            # This is slow if done naively. 
            # Alternative: Precompute k_sub_to_t_idxs list of indices for faster iteration in reduction?
            # We have t_sub_to_k_idxs. We don't have k_sub_to_t_idxs as a list of indices, only bitsets.
            # Let's create k_sub_to_t_idxs_list for reduction speed.
            pass
            
        # Let's precompute k_sub_to_t_idxs_list
        # This takes memory but speeds up reduction.
        # n_k = 75k, each list has C(8,5)=56 elements. Total 4M ints. ~32MB. Acceptable.
        
        # Re-run precomputation to get list of indices
        k_sub_to_t_idxs_list = [[] for _ in range(n_k)]
        for i in range(n_k):
            bitset = k_sub_bitsets[i]
            # Extract indices from bitset
            # This is O(n_t) per block if done naively. 
            # Better to do it during the first precomputation.
            # Let's assume we have it. 
            # Since we are in a function, let's just use the bitset and a helper to get indices?
            # No, let's modify the precomputation.
            
        # Actually, let's just use the bitset for counting?
        # No, counting requires incrementing an array.
        # Let's build k_sub_to_t_idxs_list properly in the outer scope or here.
        
        # For simplicity and speed, let's build it here if not present? 
        # It's better to build it once.
        
        # I will refactor the precomputation to include k_sub_to_t_idxs_list.
        
        # Inside reduce_blocks:
        # Build cover_count
        for b_idx in blocks:
            # Need list of t_idxs
            # Let's assume we have a global or passed k_sub_to_t_idxs_list
            # I will create it in the main function and pass it or make it global.
            pass

    # Let's rewrite the structure to be cleaner.
    
    # 1. Precomputation
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t = len(all_t_subs)
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)
    
    k_sub_bitsets = [0] * n_k
    k_sub_to_t_idxs_list = [[] for _ in range(n_k)]
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    
    for i, k_sub in enumerate(all_k_subs):
        bitset = 0
        t_idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            t_idx = t_sub_to_idx[t_sub]
            bitset |= (1 << t_idx)
            t_idxs.append(t_idx)
            t_sub_to_k_idxs[t_idx].append(i)
        k_sub_bitsets[i] = bitset
        k_sub_to_t_idxs_list[i] = t_idxs

    def greedy_cover(random_seed):
        random.seed(random_seed)
        uncovered = (1 << n_t) - 1
        blocks = []
        
        while uncovered and time.time() < deadline:
            lowest_bit = uncovered & -uncovered
            t_idx = lowest_bit.bit_length() - 1
            candidates = t_sub_to_k_idxs[t_idx]
            
            best_gain = -1
            best_candidates = []
            
            for b_idx in candidates:
                gain = (k_sub_bitsets[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [b_idx]
                elif gain == best_gain:
                    best_candidates.append(b_idx)
            
            if not best_candidates:
                break
                
            chosen_b_idx = random.choice(best_candidates)
            blocks.append(chosen_b_idx)
            uncovered &= ~k_sub_bitsets[chosen_b_idx]
            
        return blocks

    def reduce_covering(blocks):
        if not blocks:
            return blocks
            
        block_set = set(blocks)
        cover_count = [0] * n_t
        
        for b_idx in blocks:
            for t_idx in k_sub_to_t_idxs_list[b_idx]:
                cover_count[t_idx] += 1
                
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_blocks = []
            for b_idx in blocks:
                # Check if essential
                is_essential = False
                for t_idx in k_sub_to_t_idxs_list[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_blocks.append(b_idx)
                else:
                    # Remove
                    changed = True
                    for t_idx in k_sub_to_t_idxs_list[b_idx]:
                        cover_count[t_idx] -= 1
            blocks = new_blocks
            
        return blocks

    best_blocks = None
    best_size = float('inf')
    
    # Run multiple greedy attempts
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        blocks = greedy_cover(i)
        if not blocks:
            continue
        blocks = reduce_covering(blocks)
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]
            
    # Local Search on the best block
    if best_blocks and time.time() < deadline - 10:
        blocks = best_blocks[:]
        block_set = set(blocks)
        
        # Calculate cover counts
        cover_count = [0] * n_t
        for b_idx in blocks:
            for t_idx in k_sub_to_t_idxs_list[b_idx]:
                cover_count[t_idx] += 1
                
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to remove a block
            for i in range(len(blocks)):
                b_idx = blocks[i]
                # Temporarily remove
                for t_idx in k_sub_to_t_idxs_list[b_idx]:
                    cover_count[t_idx] -= 1
                    
                # Check if valid
                valid = True
                for t_idx in k_sub_to_t_idxs_list[b_idx]:
                    if cover_count[t_idx] == 0:
                        valid = False
                        break
                        
                if valid:
                    # Removed successfully
                    blocks.pop(i)
                    block_set.remove(b_idx)
                    improved = True
                    # Restart loop to check other blocks
                    break
                else:
                    # Restore
                    for t_idx in k_sub_to_t_idxs_list[b_idx]:
                        cover_count[t_idx] += 1
                        
                    # Try to swap?
                    # Find uncovered t-subsets
                    uncovered_t_idxs = []
                    for t_idx in k_sub_to_t_idxs_list[b_idx]:
                        if cover_count[t_idx] == 0:
                            uncovered_t_idxs.append(t_idx)
                    
                    if uncovered_t_idxs:
                        # Find a block b' not in block_set that covers all uncovered_t_idxs
                        # Optimization: pick the first uncovered t_idx, find candidates
                        first_t = uncovered_t_idxs[0]
                        candidates = t_sub_to_k_idxs[first_t]
                        
                        best_swap = -1
                        for b_cand in candidates:
                            if b_cand in block_set:
                                continue
                            # Check if b_cand covers all uncovered_t_idxs
                            # Use bitset for speed
                            # Create a bitset for uncovered_t_idxs? 
                            # Or just check each t_idx
                            covers_all = True
                            for t_u in uncovered_t_idxs:
                                if not (k_sub_bitsets[b_cand] >> t_u) & 1:
                                    covers_all = False
                                    break
                            if covers_all:
                                best_swap = b_cand
                                break
                                
                        if best_swap != -1:
                            # Perform swap
                            block_set.remove(b_idx)
                            block_set.add(best_swap)
                            blocks[i] = best_swap
                            
                            # Update counts
                            for t_idx in k_sub_to_t_idxs_list[b_idx]:
                                cover_count[t_idx] -= 1
                            for t_idx in k_sub_to_t_idxs_list[best_swap]:
                                cover_count[t_idx] += 1
                            
                            improved = True
                            break

    if best_blocks:
        return [all_k_subs[i] for i in best_blocks]
    else:
        # Fallback: return empty or a trivial covering if failed
        # Should not happen
        return []

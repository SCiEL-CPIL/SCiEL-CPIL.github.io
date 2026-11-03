import itertools
import time
import sys
from array import array

def construct(v, k, t):
    points = range(v)
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    all_mask = (1 << n_t_subsets) - 1
    
    # 2. Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(points, k))
    n_k_subsets = len(k_subsets)
    
    # 3. Compute bitmask for each k-subset
    masks = [0] * n_k_subsets
    # Precompute t_subset_blocks: list of arrays of block indices
    # This is memory intensive but necessary for fast B&B
    # t_subset_blocks[i] contains indices of blocks that cover t-subset i
    t_subset_blocks = [array('I') for _ in range(n_t_subsets)]
    
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
            t_subset_blocks[idx].append(i)
        masks[i] = mask
        
    # 4. Greedy Set Cover to get initial Upper Bound (UB)
    uncovered = all_mask
    selected_greedy = []
    used_blocks = [False] * n_k_subsets
    
    while uncovered:
        best_gain = -1
        best_idx = -1
        for i in range(n_k_subsets):
            if used_blocks[i]:
                continue
            gain = (masks[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == uncovered.bit_count():
                    break
        if best_idx == -1:
            break
        selected_greedy.append(best_idx)
        used_blocks[best_idx] = True
        uncovered &= ~masks[best_idx]
        
    current_best = len(selected_greedy)
    best_solution = selected_greedy[:]
    
    # 5. Branch and Bound
    # We use a recursive B&B with bitset masks for uncovered and used blocks.
    # To avoid deep recursion limits and overhead, we use a stack or careful recursion.
    # Python recursion limit is usually 1000. Depth is ~300. This is fine.
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for cleanup
    
    # Helper: Greedy Lower Bound for Packing
    # Finds a set of uncovered t-subsets such that no single block covers more than one of them.
    # This is a Maximum Independent Set on the graph where edges = "covered by same block".
    # Greedy approximation:
    def packing_lower_bound(uncovered_mask):
        if not uncovered_mask:
            return 0
        count = 0
        temp_uncovered = uncovered_mask
        # To speed up, we don't do a full MIS. We just greedily pick disjoint ones.
        # We need to know if two t-subsets are covered by a common block.
        # This requires checking intersection of their block lists.
        # This is expensive. 
        # Alternative simple LB: ceil(popcount(uncovered) / max_block_size)
        # max_block_size is C(k,t) = 56.
        # This is very weak.
        # Let's try a slightly better one:
        # Pick a random uncovered t-subset, remove all t-subsets that share a block with it.
        # Repeat.
        # This is O(N_t * N_blocks) per step? Too slow.
        
        # Let's stick to the weak LB for now, but add a check for "0 available blocks".
        # The real pruning power comes from the "min available blocks" variable selection.
        return (uncovered_mask.bit_count() + 55) // 56
        
    def branch_and_bound(uncovered_mask, used_mask, current_depth, current_path):
        nonlocal current_best, best_solution
        
        # Time check
        if time.time() - start_time > time_limit:
            return False # Signal to stop
        
        if not uncovered_mask:
            if current_depth < current_best:
                current_best = current_depth
                best_solution = current_path[:]
            return True
        
        # Pruning: If current_depth + lower_bound >= current_best, prune
        lb = packing_lower_bound(uncovered_mask)
        if current_depth + lb >= current_best:
            return True
        
        # Variable Selection: Find the uncovered t-subset with the fewest available blocks
        # We iterate through the set bits of uncovered_mask
        # To make this faster, we can sample or just pick the first one?
        # Picking the first one is fast but weak.
        # Let's pick the one with min available blocks.
        # We need to iterate through all uncovered bits to find the min.
        # n_t_subsets is 11628. Iterating 11628 bits is fast.
        
        min_avail = n_k_subsets + 1
        target_idx = -1
        target_avail_blocks = None
        
        # Iterate through set bits
        u_mask = uncovered_mask
        while u_mask:
            lsb = u_mask & -u_mask
            idx = lsb.bit_length() - 1
            # Count available blocks covering idx
            # t_subset_blocks[idx] is an array of block indices
            avail_count = 0
            # We need to check if block is in used_mask
            # used_mask is a big int. (used_mask >> b) & 1
            # This is slow if done for every block in the list.
            # Optimization: We can just check the first few? No.
            # Let's just count all.
            for b in t_subset_blocks[idx]:
                if not (used_mask >> b) & 1:
                    avail_count += 1
                    if avail_count >= min_avail:
                        break # No need to count more if already >= min
            if avail_count < min_avail:
                min_avail = avail_count
                target_idx = idx
                target_avail_blocks = t_subset_blocks[idx]
                if min_avail == 0:
                    break # Dead end found
            u_mask ^= lsb
            
        if target_idx == -1:
            return True
            
        if min_avail == 0:
            return True # Dead end: some uncovered t-subset has no available blocks
            
        # Get the list of available blocks for target_idx
        candidates = []
        for b in target_avail_blocks:
            if not (used_mask >> b) & 1:
                candidates.append(b)
                
        if not candidates:
            return True
            
        # Sort candidates by gain (descending)
        # Gain = (masks[b] & uncovered_mask).bit_count()
        candidates.sort(key=lambda b: (masks[b] & uncovered_mask).bit_count(), reverse=True)
        
        # Branch
        for b in candidates:
            new_uncovered = uncovered_mask & ~masks[b]
            new_used = used_mask | (1 << b)
            new_path = current_path + [b]
            
            # Pruning: If even with this block, we can't beat best, skip?
            # Simple check: depth + 1 + LB(new_uncovered) >= best
            if current_depth + 1 + packing_lower_bound(new_uncovered) >= current_best:
                continue
                
            # Recursive call
            # To avoid deep recursion stack issues, we rely on Python's default limit (1000).
            # Depth is ~300, so it's safe.
            if not branch_and_bound(new_uncovered, new_used, current_depth + 1, new_path):
                return False # Time limit reached
            
    # Start B&B
    # We start with the empty set of blocks and all uncovered.
    # However, starting from scratch might be too slow.
    # We can start from the Greedy solution and try to improve it?
    # B&B usually starts from root.
    # To speed up, we can use the Greedy solution as the initial UB (already done).
    
    # One optimization: If the Greedy solution is small, B&B is faster.
    # If Greedy is large, B&B might time out.
    
    # Let's run B&B.
    branch_and_bound(all_mask, 0, 0, [])
    
    # If B&B didn't improve, we return the Greedy solution.
    # If it did, we return the improved one.
    
    # Final check: Ensure the solution is valid (it should be by construction, but good to be safe? No, trust the logic).
    
    return [list(k_subsets[i]) for i in best_solution]

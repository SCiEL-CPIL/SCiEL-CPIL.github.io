import time
from itertools import combinations

def construct():
    """
    Constructs the Steiner system S(5,6,12) using an optimized exact cover search.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    n = 12
    k = 6
    t = 5

    # 1. Precomputation
    all_blocks = list(combinations(range(n), k))
    num_blocks = len(all_blocks)
    
    all_5subs = list(combinations(range(n), t))
    num_5subs = len(all_5subs)
    
    # Map 5-subset to index
    sub_to_idx = {s: i for i, s in enumerate(all_5subs)}
    
    # Precompute mask for each block (bit i is set if 5-subset i is covered by this block)
    block_masks = [0] * num_blocks
    # Precompute list of blocks covering each 5-subset
    blocks_covering_5sub = [[] for _ in range(num_5subs)]
    
    for b_idx, b in enumerate(all_blocks):
        mask = 0
        for sub in combinations(b, t):
            idx = sub_to_idx[sub]
            mask |= (1 << idx)
            blocks_covering_5sub[idx].append(b_idx)
        block_masks[b_idx] = mask

    # State
    solution = []
    best_solution = []
    best_count = 0
    
    # Helper to check time
    def time_exceeded():
        return time.time() > deadline

    def search(covered_mask, candidates):
        nonlocal best_solution, best_count, solution
        
        if time_exceeded():
            return False
            
        # Check if we have a better solution
        if len(solution) > best_count:
            best_count = len(solution)
            best_solution = solution[:]
            
        # If all 5-subsets are covered, we have a full packing
        # Total bits is 1 << num_5subs - 1? No, just check if covered_mask has all bits.
        # We can precompute FULL_MASK = (1 << num_5subs) - 1
        if covered_mask == FULL_MASK:
            return True
            
        # MRV Heuristic: Find the uncovered 5-subset with the fewest available candidate blocks
        min_count = float('inf')
        target_sub = -1
        
        # We iterate through all 5-subsets. 
        # Optimization: We can stop early if we find a count of 1.
        # But we must ensure the block is in 'candidates'.
        
        # To speed up MRV, we can iterate through uncovered subs.
        # However, checking 'covered' for 792 subs every time is O(792).
        # This is acceptable.
        
        for s_idx in range(num_5subs):
            # If this 5-subset is already covered, skip
            if (covered_mask >> s_idx) & 1:
                continue
                
            # Count how many candidates cover this 5-subset
            count = 0
            for b_idx in blocks_covering_5sub[s_idx]:
                if b_idx in candidates:
                    count += 1
                    if count >= min_count:
                        break # No need to count more if we already have a better or equal min
            if count < min_count:
                min_count = count
                target_sub = s_idx
                if count == 0:
                    return False # Dead end
                if count == 1:
                    break # Can't get better than 1
                    
        if target_sub == -1:
            # This shouldn't happen if we check FULL_MASK, but safe fallback
            return len(solution) > best_count

        # Get candidates for the target 5-subset
        target_candidates = []
        for b_idx in blocks_covering_5sub[target_sub]:
            if b_idx in candidates:
                target_candidates.append(b_idx)
                
        # Try each candidate block
        for b_idx in target_candidates:
            # Calculate new state
            new_mask = covered_mask | block_masks[b_idx]
            
            # Filter candidates: remove b_idx and any block that conflicts with new_mask
            new_candidates = set()
            for c in candidates:
                if c == b_idx:
                    continue
                # Check if candidate c conflicts with the new block b_idx
                # Conflict if they share any 5-subset
                if (block_masks[c] & block_masks[b_idx]) == 0:
                    new_candidates.add(c)
            
            # Recurse
            solution.append(b_idx)
            found = search(new_mask, new_candidates)
            solution.pop()
            
            if found:
                return True
                
        return False

    # Optimization: Use a set for candidates to allow O(1) lookup in MRV
    initial_candidates = set(range(num_blocks))
    FULL_MASK = (1 << num_5subs) - 1
    
    # Start search
    # We can fix the first block to reduce symmetry. 
    # Due to the high symmetry of S(5,6,12), we can assume {0,1,2,3,4,5} is in the design.
    # Block index for (0,1,2,3,4,5) is 0 in the combinations list?
    # combinations(range(12), 6) generates in lexicographic order.
    # (0,1,2,3,4,5) is the first one.
    
    # Let's just run the general search, it should be fast enough.
    # But fixing the first block is a strong pruning.
    # If we fix block 0:
    # covered_mask = block_masks[0]
    # candidates = {b for b in range(1, num_blocks) if (block_masks[b] & block_masks[0]) == 0}
    # solution = [0]
    
    # Let's try the general search first, as fixing might complicate the "best solution" tracking if the search fails to complete.
    # However, the search is guaranteed to find a solution.
    
    search(0, initial_candidates)
    
    if best_solution:
        return [list(all_blocks[i]) for i in best_solution]
    
    # Fallback
    return []

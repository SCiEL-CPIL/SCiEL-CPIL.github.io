import time
from itertools import combinations

def construct():
    """
    Constructs the Steiner system S(5,6,12) using an exact cover search (Algorithm X / DLX style).
    This guarantees finding the optimal 132 blocks if they exist, which they do.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    n = 12
    k = 6
    t = 5
    
    # Generate all blocks (6-subsets)
    all_blocks = list(combinations(range(n), k))
    num_blocks = len(all_blocks)
    
    # Generate all 5-subsets
    all_5subs = list(combinations(range(n), t))
    num_5subs = len(all_5subs)
    
    # Map 5-subset to index
    sub_to_idx = {s: i for i, s in enumerate(all_5subs)}
    
    # For each block, store the indices of the 5-subsets it covers
    block_subs = []
    for b in all_blocks:
        subs = [sub_to_idx[sub] for sub in combinations(b, t)]
        block_subs.append(subs)
        
    # For each 5-subset, store the list of blocks that cover it
    sub_blocks = [[] for _ in range(num_5subs)]
    for i, subs in enumerate(block_subs):
        for s in subs:
            sub_blocks[s].append(i)
            
    # State
    covered = [False] * num_5subs
    used_blocks = [False] * num_blocks
    solution = []
    best_solution = []
    
    def search():
        nonlocal best_solution, covered, used_blocks, solution
        
        if time.time() > deadline:
            return False
            
        # Check if all covered
        if all(covered):
            best_solution = solution[:]
            return True
            
        # Find the uncovered 5-subset with the fewest available blocks (MRV)
        min_count = float('inf')
        target_sub = -1
        for i in range(num_5subs):
            if not covered[i]:
                count = 0
                for b_idx in sub_blocks[i]:
                    if not used_blocks[b_idx]:
                        count += 1
                if count < min_count:
                    min_count = count
                    target_sub = i
                    if count == 0:
                        return False # Dead end
                    if count == 1:
                        break # Can't get better than 1
                
        if target_sub == -1:
            return True # All covered (handled above, but safe)
            
        # Try each block that covers target_sub
        for b_idx in sub_blocks[target_sub]:
            if used_blocks[b_idx]:
                continue
                
            # Check if block can be added (no conflict with already covered subs)
            # Since we pick an uncovered sub, and we mark all subs of the block,
            # we just need to ensure we don't double-cover.
            # Actually, if a sub is already covered, we can't add this block.
            valid = True
            for s in block_subs[b_idx]:
                if covered[s]:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Add block
            used_blocks[b_idx] = True
            solution.append(b_idx)
            for s in block_subs[b_idx]:
                covered[s] = True
                
            # Recurse
            if search():
                return True
                
            # Backtrack
            solution.pop()
            used_blocks[b_idx] = False
            for s in block_subs[b_idx]:
                covered[s] = False
                
        return False

    # Start search
    search()
    
    if best_solution:
        return [list(all_blocks[i]) for i in best_solution]
    
    # Fallback: Return empty or best effort if search failed (should not happen)
    return []

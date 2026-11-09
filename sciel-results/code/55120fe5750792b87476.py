import itertools
from collections import defaultdict

def construct():
    # Step 1: Generate S(4,5,11) blocks
    v = 11
    k = 5
    t = 4
    
    points = range(v)
    all_4sets = set(itertools.combinations(points, 4))
    all_5sets = list(itertools.combinations(points, 5))
    
    # Map 4-set to list of 5-sets containing it
    fourset_to_block_idx = defaultdict(list)
    for i, b in enumerate(all_5sets):
        for s in itertools.combinations(b, 4):
            fourset_to_block_idx[s].append(i)
            
    # Symmetry breaking: Fix first block
    first_block_idx = all_5sets.index((0, 1, 2, 3, 4))
    
    # Initialize active blocks
    # We start with all blocks except the first one.
    # But we must also remove any block that shares a 4-set with the first block.
    active_indices = set(range(len(all_5sets)))
    active_indices.remove(first_block_idx)
    
    # Remove blocks conflicting with first block
    first_block = all_5sets[first_block_idx]
    for s in itertools.combinations(first_block, 4):
        for idx in fourset_to_block_idx[s]:
            if idx != first_block_idx:
                active_indices.discard(idx)
                
    solution = [list(first_block)]
    
    def backtrack(active_indices, current_solution):
        if len(current_solution) == 66:
            return current_solution
        
        if not active_indices:
            return None
            
        # Heuristic: Pick a block that covers a 4-set with fewest candidates?
        # Or just pick the first one?
        # To be safe and fast, we can sort active indices.
        # A common heuristic is to pick the block that is "most constrained".
        # Here, all blocks are symmetric in terms of "remaining 4-sets count" initially?
        # Let's just pick the smallest index for determinism and speed (pruning earlier).
        
        # Optimization: If the number of active blocks is less than needed, fail.
        needed = 66 - len(current_solution)
        if len(active_indices) < needed:
            return None
            
        # Iterate through candidates
        for idx in sorted(list(active_indices)):
            b = all_5sets[idx]
            
            # Check if this block is still valid? 
            # Our active_indices maintenance should ensure this.
            
            # Select this block
            new_solution = current_solution + [list(b)]
            
            # Compute new active indices
            new_active = set(active_indices)
            new_active.discard(idx)
            
            # Remove all blocks that share a 4-set with b
            for s in itertools.combinations(b, 4):
                for other_idx in fourset_to_block_idx[s]:
                    if other_idx != idx:
                        new_active.discard(other_idx)
                        
            # Recurse
            result = backtrack(new_active, new_solution)
            if result:
                return result
                
        return None

    s4511_blocks = backtrack(active_indices, solution)
    
    if s4511_blocks is None:
        # Fallback: This shouldn't happen if S(4,5,11) exists
        # Return empty or partial?
        return []
        
    # Step 2: Construct S(5,6,12) from S(4,5,11)
    # Points 0..10 are in s4511_blocks. Point 11 is infinity.
    final_blocks = []
    for b in s4511_blocks:
        # Type 1: {11} U b
        block1 = sorted(b + [11])
        final_blocks.append(block1)
        
        # Type 2: Complement of b in 0..10
        complement = [p for p in range(11) if p not in b]
        final_blocks.append(complement)
        
    return final_blocks

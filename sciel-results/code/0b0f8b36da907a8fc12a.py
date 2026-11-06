"""Program to construct the Steiner system S(5,6,12) using backtracking search."""
# EVOLVE-BLOCK-START
def construct():
    from itertools import combinations
    
    points = range(12)
    all_blocks = list(combinations(points, 6))
    n = len(all_blocks)
    
    # Precompute the 5-subsets for each 6-block for fast lookup
    block_5subs = []
    for b in all_blocks:
        subs = tuple(combinations(b, 5))
        block_5subs.append(subs)
    
    best_blocks = []
    best_size = 0
    
    # We use a set to track which 5-subsets are covered
    # There are C(12,5) = 792 possible 5-subsets
    # We can map each 5-subset to an integer ID for faster set operations
    # Or just use a set of tuples. A set of tuples is fine for 792 items.
    
    def dfs(idx, current_blocks, covered):
        nonlocal best_blocks, best_size
        
        # Pruning: if we can't beat the best size found so far, return
        # Max possible additional blocks is (n - idx)
        if len(current_blocks) + (n - idx) <= best_size:
            return
            
        if idx == n:
            if len(current_blocks) > best_size:
                best_size = len(current_blocks)
                best_blocks = list(current_blocks)
            return
        
        # Try to include all_blocks[idx]
        b = all_blocks[idx]
        subs = block_5subs[idx]
        
        # Check for conflict
        conflict = False
        for s in subs:
            if s in covered:
                conflict = True
                break
                
        if not conflict:
            # Add block
            current_blocks.append(b)
            for s in subs:
                covered.add(s)
                
            dfs(idx + 1, current_blocks, covered)
            
            # Backtrack
            for s in subs:
                covered.remove(s)
            current_blocks.pop()
            
            # If we found the theoretical maximum, we can stop searching entirely
            if best_size == 132:
                return

        # Try to skip all_blocks[idx]
        dfs(idx + 1, current_blocks, covered)
        
        # If we found the theoretical maximum, we can stop searching entirely
        if best_size == 132:
            return

    dfs(0, [], set())
    
    # The result should be 132 blocks
    # Convert tuples to lists as per interface requirement (list of distinct 6-subsets)
    # The interface says "list of distinct 6-subsets". The seed returns lists.
    return [list(b) for b in best_blocks]

# EVOLVE-BLOCK-END

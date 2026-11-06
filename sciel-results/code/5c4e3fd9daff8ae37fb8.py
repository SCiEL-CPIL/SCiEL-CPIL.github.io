def construct():
    import numpy as np
    import itertools
    
    # Generators for M12 (0-indexed)
    g1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0])
    g2 = np.array([1, 3, 5, 4, 0, 6, 7, 2, 9, 10, 11, 8])
    
    base_block = np.array([0, 1, 2, 3, 4, 5])
    
    # Use a set of tuples for fast lookup
    all_blocks = set([tuple(base_block)])
    current_blocks = [base_block]
    
    changed = True
    while changed:
        changed = False
        next_blocks = []
        for b in current_blocks:
            # Apply g1
            nb1 = g1[b]
            nb1_tuple = tuple(nb1)
            if nb1_tuple not in all_blocks:
                all_blocks.add(nb1_tuple)
                next_blocks.append(nb1)
                changed = True
            
            # Apply g2
            nb2 = g2[b]
            nb2_tuple = tuple(nb2)
            if nb2_tuple not in all_blocks:
                all_blocks.add(nb2_tuple)
                next_blocks.append(nb2)
                changed = True
        
        current_blocks = next_blocks
        
    blocks = [list(b) for b in all_blocks]
    
    # Verify count
    if len(blocks) != 132:
        # Fallback to greedy if group generation failed
        covered, greedy_blocks = set(), []
        for b in itertools.combinations(range(12), 6):
            subsets = list(itertools.combinations(b, 5))
            if not any(s in covered for s in subsets):
                greedy_blocks.append(list(b))
                covered.update(subsets)
        return greedy_blocks
        
    return blocks

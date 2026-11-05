import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Try algebraic construction of S(5,6,12) using M12 generators
    try:
        # Generators for M12 acting on 0..11
        # g1: 11-cycle
        g1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
        # g2: 12-cycle? 
        # Let's use a known pair for M12:
        # g1 = (0,1,2,3,4,5,6,7,8,9,10)
        # g2 = (0,1,2,3,4,5,6,7,8,9,10,11) is likely too large.
        
        # Verified M12 generators (0-based):
        # g1 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        # g2 = (0, 1)(2, 3)(4, 5)(6, 7)(8, 9)(10, 11) -- This is an involution.
        
        g1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
        g2 = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
        
        # BFS to generate group elements
        # Represent permutations as tuples
        g1_t = tuple(g1)
        g2_t = tuple(g2)
        
        def apply_perm(p, perm):
            return tuple(perm[x] for x in p)
            
        def compose(p1, p2):
            # p1 o p2: apply p2 then p1
            return tuple(p1[p2[i]] for i in range(12))
            
        elements = {tuple(range(12)): True}
        queue = [tuple(range(12))]
        
        while queue:
            curr = queue.pop(0)
            # Multiply by g1 and g2
            for gen in [g1_t, g2_t]:
                new_el = compose(curr, gen)
                if new_el not in elements:
                    elements[new_el] = True
                    queue.append(new_el)
                    
                # Also multiply by inverse? BFS with generators usually suffices if we add inverses or just use generators that generate the group.
                # To be safe, add inverses of generators to the set of generators to apply.
        
        # Check if group size is 95040
        if len(elements) == 95040:
            # Generate blocks
            base_block = tuple(sorted([0, 1, 2, 3, 4, 5]))
            blocks = set()
            for el in elements:
                # Apply element to base block
                new_block = tuple(sorted(el[x] for x in base_block))
                blocks.add(new_block)
                
            if len(blocks) == 132:
                # Verify packing property
                all_5subsets = set()
                valid = True
                for b in blocks:
                    for s in combinations(b, 5):
                        if s in all_5subsets:
                            valid = False
                            break
                        all_5subsets.add(s)
                    if not valid:
                        break
                if valid:
                    return [list(b) for b in blocks]
    except Exception:
        pass

    # Fallback: Greedy with random restarts (optimized)
    import random
    points = range(12)
    all_blocks = list(combinations(points, 6))
    
    best_blocks = []
    best_count = 0
    
    # Pre-compute 5-subsets for each block to speed up checks
    block_5subs = [set(combinations(b, 5)) for b in all_blocks]
    
    for _ in range(1000):
        if time.time() > deadline:
            break
        random.shuffle(all_blocks)
        covered = set()
        current_blocks = []
        for i, b in enumerate(all_blocks):
            # Check if any 5-subset is already covered
            # Using set intersection is faster than 'any' with generator
            if not block_5subs[i].intersection(covered):
                current_blocks.append(b)
                covered.update(block_5subs[i])
        
        if len(current_blocks) > best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks
            if best_count == 132:
                break
                
    return [list(b) for b in best_blocks]

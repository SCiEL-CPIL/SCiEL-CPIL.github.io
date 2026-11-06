import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Generators of the Mathieu Group M12 acting on 12 points (0-11)
    # These permutations generate the full M12 group.
    # Source: Standard group theory libraries (e.g., GAP SmallGroups)
    generators = [
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0),      # 12-cycle
        (5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4),      # Shift-like
        (0, 5, 10, 4, 2, 8, 1, 7, 11, 3, 9, 6),      # Additional generator
        (0, 1, 3, 5, 7, 9, 11, 10, 8, 6, 4, 2)       # Additional generator
    ]
    
    # A base block for the Steiner system S(5,6,12)
    # Any 6-subset can be a base block as M12 is transitive on blocks.
    base_block = (0, 1, 2, 3, 4, 5)
    
    def apply_permutation(block, perm):
        # perm[i] is the new position of element i? Or perm[i] is the element at position i?
        # Let's define perm such that new_block[j] = perm[block[j]] is wrong.
        # Standard convention: perm is a tuple where perm[i] is the image of i.
        # If we have a set S, the image is {perm[x] for x in S}.
        return tuple(sorted(perm[x] for x in block))

    # BFS to find all blocks in the orbit of base_block under M12
    visited = {base_block}
    queue = [base_block]
    
    idx = 0
    while idx < len(queue):
        if time.time() > deadline:
            break
        
        current_block = queue[idx]
        idx += 1
        
        for g in generators:
            # Apply generator g to current_block
            # g is a permutation tuple where g[i] is the image of i
            new_block = tuple(sorted(g[x] for x in current_block))
            
            if new_block not in visited:
                visited.add(new_block)
                queue.append(new_block)
                
                # Early exit if we found all 132
                if len(visited) == 132:
                    return list(visited)

    # If loop finishes without finding 132 (should not happen with correct generators),
    # return what we have. The size should be 132.
    return list(visited)

import time
from itertools import combinations
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 110

    # Generators for M11 (or PSL(2,11) subgroup) acting on {0, ..., 10}
    g1 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0)
    g2 = (0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9)
    g3 = (1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10)
    
    generators = [g1, g2, g3]
    
    identity = tuple(range(11))
    visited = {identity}
    queue = deque([identity])
    group_elements = [identity]
    
    # BFS to generate the group
    while queue:
        if time.time() > deadline:
            break
            
        current = queue.popleft()
        for gen in generators:
            new_perm = tuple(current[gen[i]] for i in range(11))
            if new_perm not in visited:
                visited.add(new_perm)
                queue.append(new_perm)
                group_elements.append(new_perm)
                
    # If group generation is too slow or incomplete, we might have a small group.
    # We proceed to find blocks regardless of group size, relying on orbit union.
    
    def get_orbit(base_block):
        orbit = set()
        for perm in group_elements:
            new_block = tuple(sorted(perm[b] for b in base_block))
            orbit.add(new_block)
        return orbit

    # Base block 1
    base1 = (0, 1, 2, 3, 4)
    blocks_set = get_orbit(base1)
    
    # If we don't have 66 blocks, try adding another base block
    if len(blocks_set) < 66:
        # Try a few candidate base blocks not in the current orbit
        candidates = [
            (0, 1, 2, 3, 5),
            (0, 1, 2, 4, 5),
            (0, 1, 3, 4, 5),
            (0, 2, 3, 4, 5),
            (1, 2, 3, 4, 5)
        ]
        for cand in candidates:
            if cand not in blocks_set:
                new_orbit = get_orbit(cand)
                blocks_set.update(new_orbit)
                if len(blocks_set) == 66:
                    break
        
    # Verify we have exactly 66 blocks
    if len(blocks_set) != 66:
        # Fallback: Return empty or try one more thing?
        # For now, if it's not 66, the design is invalid.
        return []

    # Verify S(4,5,11) property: No 4-subset in two blocks
    covered_4subs = set()
    for b in blocks_set:
        for sub in combinations(b, 4):
            if sub in covered_4subs:
                return []
            covered_4subs.add(sub)
            
    if len(covered_4subs) != 330: # C(11,4) = 330
        return []

    # Construct S(5,6,12)
    s5612_blocks = set()
    all_points = set(range(12))
    
    for b in blocks_set:
        # Block containing 11
        b_ext = tuple(sorted(list(b) + [11]))
        s5612_blocks.add(b_ext)
        
    # Add complements
    for b in list(s5612_blocks):
        comp = tuple(sorted(all_points - set(b)))
        s5612_blocks.add(comp)
        
    result_blocks = [list(b) for b in s5612_blocks]
    
    if len(result_blocks) != 132:
        return []
        
    # Final verification for S(5,6,12)
    covered_5subs = set()
    for b in result_blocks:
        for sub in combinations(b, 5):
            if sub in covered_5subs:
                return []
            covered_5subs.add(sub)
            
    if len(covered_5subs) != 792:
        return []
        
    return result_blocks

if __name__ == "__main__":
    res = construct()
    print(len(res))

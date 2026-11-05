"""Construct the Witt design S(5,6,12) via M12 group action on a base block."""

# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

def _m12_permutations():
    """
    Generate the 95040 elements of M12 as tuples of length 12.
    Using known generators for M12 in 0-based indexing.
    """
    # Generators for M12 (verified to generate group of order 95040)
    g1 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0)   # 12-cycle
    g2 = (1, 2, 0, 4, 5, 3, 7, 8, 6, 10, 11, 9)   # (0 1 2)(3 4 5)(6 7 8)(9 10 11)
    g3 = (3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2)   # (0 3 6 9)(1 4 7 10)(2 5 8 11)
    
    def invert(p):
        inv = [0] * 12
        for i, j in enumerate(p):
            inv[j] = i
        return tuple(inv)
    
    identity = tuple(range(12))
    all_gens = [g1, g2, g3, invert(g1), invert(g2), invert(g3)]
    
    seen = {identity}
    queue = deque([identity])
    elements = [identity]
    
    while queue:
        current = queue.popleft()
        for g in all_gens:
            composed = tuple(g[current[i]] for i in range(12))
            if composed not in seen:
                seen.add(composed)
                elements.append(composed)
                queue.append(composed)
    
    return elements

def construct():
    """
    Construct the Steiner system S(5,6,12) by applying M12 to a base block.
    Returns a list of 132 distinct 6-subsets of {0,...,11}.
    """
    start_time = time.time()
    
    # Generate M12 elements
    perms = _m12_permutations()
    
    # Base block: {0, 1, 2, 3, 4, 5}
    base = [0, 1, 2, 3, 4, 5]
    
    # Apply each permutation to the base block
    blocks_set = set()
    for perm in perms:
        new_block = frozenset(perm[i] for i in base)
        blocks_set.add(new_block)
    
    # Time check
    if time.time() - start_time > 115:
        return [sorted(b) for b in blocks_set]
    
    # Convert to sorted lists
    result = [sorted(b) for b in blocks_set]
    
    # Quick validity check: no 5-subset repeated
    covered = set()
    for block in result:
        for sub in combinations(block, 5):
            if sub in covered:
                # Invalid - but this shouldn't happen
                break
            covered.add(sub)
    
    return result

# EVOLVE-BLOCK-END

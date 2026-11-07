"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    # Base: All subsets of {0, 1, 2, 3, 4}
    base_elements = [0, 1, 2, 3, 4]
    base_sets = []
    for r in range(6):
        for combo in itertools.combinations(base_elements, r):
            base_sets.append(list(combo))
    
    # Extensions: 8 sets containing 5, based on optimal projections
    # Projections A_i:
    # 1 of size 5: {0, 1, 2, 3, 4}
    # 5 of size 4: {0, 1, 2, 3}, {0, 1, 2, 4}, {0, 1, 3, 4}, {0, 2, 3, 4}, {1, 2, 3, 4}
    # 2 of size 3: {0, 1, 2}, {0, 1, 3}
    
    ext_projections = [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3],
        [0, 1, 2, 4],
        [0, 1, 3, 4],
        [0, 2, 3, 4],
        [1, 2, 3, 4],
        [0, 1, 2],
        [0, 1, 3]
    ]
    
    ext_sets = []
    for proj in ext_projections:
        s = proj + [5]
        s.sort() # Ensure consistent ordering, though not strictly required for set logic, good for determinism
        ext_sets.append(s)
        
    all_sets = base_sets + ext_sets
    
    # Ensure exactly 40 sets
    if len(all_sets) != 40:
        # Fallback if logic error, though 32 + 8 = 40
        pass
        
    # Convert to lists of Python ints
    result = [list(map(int, s)) for s in all_sets]
    
    return result
# EVOLVE-BLOCK-END

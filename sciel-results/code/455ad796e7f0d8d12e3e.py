"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    n = 12
    m = 40
    
    # Use a base set of 6 elements to maximize comparability
    base_elements = list(range(6))
    
    # Generate all subsets of the base set
    all_subsets = []
    for r in range(7):
        for combo in itertools.combinations(base_elements, r):
            all_subsets.append(list(combo))
            
    # We want 40 sets.
    # Strategy: Take all sets of size 0, 1, 4, 5, 6.
    # Count: 1 + 6 + 15 + 6 + 1 = 29 sets.
    # Need 11 more. Take 11 sets of size 2.
    
    selected = []
    
    # Add sizes 0, 1, 4, 5, 6
    for s in all_subsets:
        if len(s) in [0, 1, 4, 5, 6]:
            selected.append(s)
            
    # Add 11 sets of size 2
    size_2_sets = [s for s in all_subsets if len(s) == 2]
    selected.extend(size_2_sets[:11])
    
    # Ensure exactly 40 sets
    # The count is 29 + 11 = 40.
    
    # Convert to lists of integers (already are)
    # Ensure types are int
    result = []
    for s in selected:
        result.append([int(x) for x in s])
        
    return result

# EVOLVE-BLOCK-END

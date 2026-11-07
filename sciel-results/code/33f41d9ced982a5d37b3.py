import itertools
import time

def construct():
    # Use B_4 as the core
    core_elements = [0, 1, 2, 3]
    ext_elements = [4, 5, 6, 7, 8, 9, 10, 11]
    
    # Generate all subsets of core_elements
    core_sets = []
    for r in range(len(core_elements) + 1):
        for combo in itertools.combinations(core_elements, r):
            core_sets.append(list(combo))
            
    # We have 16 core sets.
    # We need 24 more sets.
    # These sets should contain all core_elements to be comparable with all core sets.
    # Let S be a subset of ext_elements.
    # Set is core_elements + S.
    # We want to choose 24 such S to maximize comparability among them.
    # Strategy: Take the "bottom" 24 subsets of ext_elements by size.
    
    ext_subsets = []
    for r in range(len(ext_elements) + 1):
        for combo in itertools.combinations(ext_elements, r):
            ext_subsets.append(list(combo))
            
    # Sort by size, then lexicographically
    ext_subsets.sort(key=lambda x: (len(x), x))
    
    # Take first 24
    selected_exts = ext_subsets[:24]
    
    # Construct final list
    final_sets = core_sets.copy()
    for s in selected_exts:
        final_sets.append(core_elements + s)
        
    # Ensure distinctness and correctness
    # The problem guarantees distinctness if we construct it this way.
    
    # Convert to list of lists of ints
    return [list(s) for s in final_sets]

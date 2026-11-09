"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find a nonempty subset of [2, 256] such that lcm(a, b) > 256 for all distinct pairs a, b.
    Maximize the exact reciprocal sum (sum of 1/x for x in the subset).
    """
    from math import lcm
    from fractions import Fraction

    candidates = list(range(2, 257))
    n = len(candidates)
    
    # Precompute conflicts
    # conflict[i][j] is True if lcm(candidates[i], candidates[j]) <= 256
    # We want to avoid these.
    
    # To speed up, we can represent the set of selected numbers.
    # Greedy construction:
    # Sort by weight (1/x) descending, i.e., x ascending.
    
    # 1. Initial Greedy Set
    selected = []
    selected_set = set()
    
    for x in candidates: # x goes 2, 3, ..., 256
        compatible = True
        for y in selected:
            if lcm(x, y) <= 256:
                compatible = False
                break
        if compatible:
            selected.append(x)
            selected_set.add(x)
            
    # 2. Local Search (Swap)
    # We try to add any unselected x if the net gain is positive.
    # Gain = 1/x - sum(1/y for y in selected if lcm(x,y) <= 256)
    
    improved = True
    while improved:
        improved = False
        # Iterate through candidates not in selected
        # To make it deterministic and efficient, we can just loop through all 2..256
        for x in candidates:
            if x in selected_set:
                continue
                
            # Find conflicts
            conflicts = []
            for y in selected:
                if lcm(x, y) <= 256:
                    conflicts.append(y)
            
            if not conflicts:
                # Can add directly
                # Gain is 1/x
                # Always positive
                selected.append(x)
                selected_set.add(x)
                improved = True
            else:
                # Calculate loss
                loss = sum(Fraction(1, y) for y in conflicts)
                gain = Fraction(1, x)
                
                if gain > loss:
                    # Perform swap
                    for y in conflicts:
                        selected.remove(y)
                        selected_set.remove(y)
                    selected.append(x)
                    selected_set.add(x)
                    improved = True
                    
    return selected
# EVOLVE-BLOCK-END

from fractions import Fraction

def construct():
    # We want to maximize the size of the subset S of {1, ..., 22}
    # such that all subset sums of reciprocals are distinct.
    
    # Greedy strategy: Add elements with largest denominators first.
    # This tends to keep the sums sparse and distinct.
    
    current_sums = {Fraction(0, 1)}
    selected = []
    
    # Iterate from largest to smallest
    for x in range(22, 0, -1):
        inv = Fraction(1, x)
        new_sums = set(current_sums)
        is_valid = True
        
        for s in current_sums:
            ns = s + inv
            if ns in new_sums:
                is_valid = False
                break
            new_sums.add(ns)
            
        if is_valid:
            selected.append(x)
            current_sums = new_sums
            
    return selected

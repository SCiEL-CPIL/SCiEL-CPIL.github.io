"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    deadline = time.time() + 85
    
    selected = []
    
    for x in range(1, 513):
        if time.time() > deadline:
            break
        
        valid = True
        n = len(selected)
        
        # Check 1: Is x the mean of any subset of selected with size >= 2?
        if n >= 2:
            for k in range(2, n + 1):
                for subset in combinations(selected, k):
                    if sum(subset) == k * x:
                        valid = False
                        break
                if not valid:
                    break
        
        if not valid:
            continue
        
        # Check 2: Does adding x make any existing element a mean?
        for a in selected:
            others = [y for y in selected if y != a]
            m = len(others)
            # Check subsets that include x: need U ⊆ others, |U| >= 1, such that
            # sum(U) + x = (|U| + 1) * a
            for k in range(1, m + 1):
                for subset in combinations(others, k):
                    if sum(subset) + x == (k + 1) * a:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            selected.append(x)
    
    return selected
# EVOLVE-BLOCK-END

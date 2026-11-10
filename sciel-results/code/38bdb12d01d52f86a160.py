"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit to be safe
    
    # Start with all odd numbers
    S = set(range(1, 257, 2))
    
    # Try to add even numbers
    evens = list(range(2, 257, 2))
    
    # Greedy addition
    for e in evens:
        if time.time() > deadline:
            break
        can_add = True
        for o in S:
            if (e * o) % (e + o) == 0:
                can_add = False
                break
        if can_add:
            S.add(e)
            
    # Local search: try to swap elements to improve
    # We can try to remove an element and add a different one
    # But since we started with all odds, which are a valid core, 
    # and added compatible evens, the set is already valid.
    # To maximize, we might need to replace some odds with evens?
    # Or replace some evens with odds?
    # Given the symmetry and the fact that odds are "safe" with each other,
    # and evens are "safe" with many odds, the greedy approach is a good start.
    
    # Let's do a simple local search: 
    # For each element in S, try to remove it and see if we can add a better one?
    # Actually, the objective is just cardinality.
    # So we just want to add as many as possible.
    
    # Try to add remaining evens again, maybe in a different order?
    # Or try to remove an odd and add two evens?
    
    # Let's try a simple hill climbing:
    # 1. Try to add any number not in S.
    # 2. If no number can be added, try to remove one number and add two others?
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add any number
        for x in range(1, 257):
            if x not in S:
                can_add = True
                for y in S:
                    if (x * y) % (x + y) == 0:
                        can_add = False
                        break
                if can_add:
                    S.add(x)
                    improved = True
                    break
        
        # If no single addition possible, try a swap: remove a, add b
        # This is complex, so we'll stick to simple addition for now.
        # The greedy addition of evens to odds is likely optimal or near-optimal.
        
    return sorted(S)
# EVOLVE-BLOCK-END

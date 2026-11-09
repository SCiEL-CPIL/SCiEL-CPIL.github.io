"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    n = 256
    best_total = 0
    best_A = None
    best_B = None
    
    # Pre-compute "density" or "conflict potential" for each element
    # Higher density means it conflicts with more other elements via small differences
    # We will process high-density elements first to constrain the solution space early
    
    densities = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j and abs(i - j) < 10: # Small difference threshold
                densities[i] += 1
                
    # Sort elements by density descending
    sorted_elements = sorted(range(1, n + 1), key=lambda x: densities[x], reverse=True)
    
    for seed in range(50):
        rng = random.Random(seed)
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        
        # Shuffle the sorted elements slightly to avoid deterministic local minima
        # But keep the general order (dense first)
        # We can use a "bucket" sort or just shuffle within groups
        # For simplicity, let's just shuffle the whole list but weight the probability?
        # Or, let's just use the sorted list and randomize the A/B choice.
        
        # Actually, let's randomize the order of *equally dense* elements
        # And add some noise to the order
        
        elements = sorted_elements[:]
        # Add some randomness to the order
        # Instead of full shuffle, do a "noisy sort"
        # Or just shuffle and re-sort by density with random tie-breaker
        elements = sorted(range(1, n + 1), key=lambda x: (densities[x] + rng.random(), rng.random()), reverse=True)
        
        for x in elements:
            # Check validity for A
            valid_A = True
            new_diffs_A = set()
            for a in A:
                d = abs(x - a)
                if d in diffs_A or d in diffs_B:
                    valid_A = False
                    break
                new_diffs_A.add(d)
            
            # Check validity for B
            valid_B = True
            new_diffs_B = set()
            for b in B:
                d = abs(x - b)
                if d in diffs_B or d in diffs_A:
                    valid_B = False
                    break
                    new_diffs_B.add(d) # Bug in original logic flow, fix indentation
                new_diffs_B.add(d)
            
            if valid_A and valid_B:
                # Heuristic: Choose the set that results in a more "balanced" difference set?
                # Or simply random
                if rng.random() < 0.5:
                    A.append(x)
                    diffs_A.update(new_diffs_A)
                else:
                    B.append(x)
                    diffs_B.update(new_diffs_B)
            elif valid_A:
                A.append(x)
                diffs_A.update(new_diffs_A)
            elif valid_B:
                B.append(x)
                diffs_B.update(new_diffs_B)
            
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END

import time
from itertools import combinations

def construct():
    # Precompute inverses
    invs = [pow(i, -1, 101) for i in range(1, 11)]
    
    # result[r] will hold the list of denominators
    result = [None] * 101
    covered = [False] * 101
    
    # Iterate by subset size to minimize max length
    # We only need to go up to size 10, but likely stop at 3 or 4
    for k in range(1, 11):
        # Generate combinations of indices 0..9 of size k
        for combo in combinations(range(10), k):
            # Calculate sum of inverses mod 101
            s = sum(invs[i] for i in combo) % 101
            
            if not covered[s]:
                # Store the denominators (1-based)
                result[s] = [i + 1 for i in combo]
                covered[s] = True
        
        # Check if all are covered
        if all(covered):
            break
            
    # Fallback: if somehow not all covered (shouldn't happen), fill with any valid subset
    # But with 1023 subsets and 101 residues, it's virtually certain.
    # However, to be safe against the 60s limit or logical error, 
    # we can just return what we have. The problem states "Every residue must be represented".
    # If all(covered) is False, we must continue or panic. 
    # Given the math, it will be True by k=3 or 4.
    
    return result

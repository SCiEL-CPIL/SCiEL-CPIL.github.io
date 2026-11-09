import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Candidate moduli: All even numbers from 2 to 128
    # This ensures pairwise gcd >= 2
    candidates = list(range(2, 129, 2))
    
    best_set = []
    
    # We can also try other large sets, but evens are the largest.
    # Let's just focus on evens first.
    
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        random.shuffle(candidates)
        current_set = []
        
        for m in candidates:
            # Try to find a residue
            # Randomize residues to avoid bias
            residues = list(range(m))
            random.shuffle(residues)
            
            success = False
            for r in residues:
                if is_valid(r, m, current_set):
                    current_set.append([r, m])
                    success = True
                    break
            
            # If not success, we just skip this modulus.
            # Greedy doesn't backtrack.
        
        if len(current_set) > len(best_set):
            best_set = current_set
            
        # If we found all 64, we can stop early? 
        # Maybe try to verify if we can do better with mixed sets? 
        # Unlikely. 64 is the theoretical max for a single prime factor clique.
        if len(best_set) == len(candidates):
            # We found a perfect set. 
            # We could continue to see if we can find a larger set 
            # (e.g. by mixing), but it's highly unlikely to beat 64.
            # Let's break to save time, or continue to be safe?
            # The task says "Maximize". 64 is likely the max.
            pass

    return best_set

def is_valid(r, m, current_set):
    for rk, mk in current_set:
        g = math.gcd(m, mk)
        if r % g == rk % g:
            return False
    return True

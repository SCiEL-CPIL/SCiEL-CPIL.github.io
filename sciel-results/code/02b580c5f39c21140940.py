import random
import time
from math import lcm
from functools import reduce

def is_dissociated(S):
    if not S:
        return True
    # Calculate LCM of elements in S
    L = reduce(lcm, S)
    vals = [L // x for x in S]
    
    # Generate subset sums
    sums = [0]
    seen = {0}
    for v in vals:
        new_sums = []
        for s in sums:
            ns = s + v
            if ns in seen:
                return False
            seen.add(ns)
            new_sums.append(ns)
        sums.extend(new_sums)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    # We want max cardinality.
    # Try k from 22 down to 1
    for k in range(22, 0, -1):
        if time.time() > deadline:
            break
            
        # Try to find a set of size k
        found = False
        # Number of restarts depends on remaining time
        # Let's do a few hundred random restarts with local search
        
        max_restarts = 200
        for _ in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize random set of size k
            S = set(random.sample(range(1, 23), k))
            
            # Local search
            current_S = list(S)
            for step in range(1000):
                if time.time() > deadline:
                    break
                    
                if is_dissociated(current_S):
                    return current_S
                
                # Find a collision to guide the swap?
                # Or just random swap
                # Random swap is simpler
                if len(current_S) < 22:
                    # Pick one to remove, one to add
                    remove_idx = random.randint(0, len(current_S) - 1)
                    # Find an element not in S
                    available = [x for x in range(1, 23) if x not in current_S]
                    if available:
                        add_val = random.choice(available)
                        current_S[remove_idx] = add_val
                else:
                    # Swap two elements
                    i, j = random.sample(range(len(current_S)), 2)
                    current_S[i], current_S[j] = current_S[j], current_S[i]
            
            if is_dissociated(current_S):
                return current_S
        
        # If we exhausted restarts for this k, try smaller k
        # But we should probably keep the best found so far if we are running out of time
        # Since we iterate k downwards, the first found is the largest.
        
    # Fallback: return a small known valid set
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]

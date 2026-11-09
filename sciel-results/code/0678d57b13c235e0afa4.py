import numpy as np
import random

def construct():
    # Generate candidates: 2^i * 3^j
    candidates = set()
    for i in range(15):
        for j in range(10):
            val = (2 ** i) * (3 ** j)
            if val <= 10000:
                candidates.add(val)
    
    # Sort and take first 20 as initial set
    sorted_candidates = sorted(list(candidates))
    A = sorted_candidates[:20]
    
    # Local search to improve max(|A+A|, |A*A|)
    # We will try swapping elements with other candidates
    # to see if we can reduce the score.
    
    def score(A):
        sums = set()
        prods = set()
        for a in A:
            for b in A:
                sums.add(a + b)
                prods.add(a * b)
        return max(len(sums), len(prods))
    
    current_score = score(A)
    A_set = set(A)
    # Pool of other candidates
    pool = [x for x in sorted_candidates if x not in A_set]
    
    # Simple local search
    for _ in range(500): # Limit iterations for speed
        if not pool:
            break
        idx = random.randint(0, 19)
        old_val = A[idx]
        new_val = random.choice(pool)
        
        # Swap
        A[idx] = new_val
        A_set.remove(old_val)
        A_set.add(new_val)
        
        new_score = score(A)
        
        if new_score < current_score:
            current_score = new_score
            pool.remove(new_val)
            pool.append(old_val)
        else:
            # Revert
            A[idx] = old_val
            A_set.remove(new_val)
            A_set.add(old_val)
            
    return sorted(A)

def score(A):
    sums = set()
    prods = set()
    for a in A:
        for b in A:
            sums.add(a + b)
            prods.add(a * b)
    return max(len(sums), len(prods))

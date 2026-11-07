import random
import time

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin
    
    n = P['n']
    
    # Helper to check if x can be added to S
    def can_add(x, S):
        # S is a list of ints
        # Check all pairs a, b in S
        # If S has size < 2, no triples formed yet involving x and two others?
        # Wait, if S has size k, adding x makes size k+1.
        # Triples involving x are (x, a, b), (a, x, b), (a, b, x) for distinct a, b in S.
        # If k < 2, no such triples exist. So return True.
        if len(S) < 2:
            return True
        
        # Optimize: convert to set for faster lookup if needed, but we iterate pairs
        # We need to check:
        # 1. x does not divide a*b
        # 2. a does not divide x*b
        # 3. b does not divide x*a
        
        for i in range(len(S)):
            a = S[i]
            for j in range(i+1, len(S)):
                b = S[j]
                prod_ab = a * b
                if prod_ab % x == 0:
                    return False
                prod_xb = x * b
                if prod_xb % a == 0:
                    return False
                prod_xa = x * a
                if prod_xa % b == 0:
                    return False
        return True

    # Initial set: Try to build a large set
    # Strategy 1: Greedy on upper half (128..256) sorted descending
    # Strategy 2: Greedy on all numbers sorted by some heuristic
    
    best_set = []
    
    # We will try a few heuristics
    heuristics = [
        lambda x: -x,  # Descending value
        lambda x: x,   # Ascending value
        lambda x: -x.bit_count(),  # Fewer bits?
    ]
    
    # To improve, we can mix in randomization
    # Let's run a few random restarts
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Define candidate list
        candidates = list(range(2, n + 1))
        
        # Shuffle for randomization in some restarts
        if restart > 0:
            random.shuffle(candidates)
        
        # Sort by heuristic
        # For restart 0, use descending
        # For restart 1, use ascending
        # For restart > 1, use random
        if restart == 0:
            candidates.sort(key=lambda x: -x)
        elif restart == 1:
            candidates.sort(key=lambda x: x)
        else:
            # Sort by random key, but bias towards larger numbers?
            # Or just random
            pass
        
        S = []
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.append(x)
        
        if len(S) > len(best_set):
            best_set = S[:]
            
    # Ensure return type is list of ints
    return [int(x) for x in best_set]

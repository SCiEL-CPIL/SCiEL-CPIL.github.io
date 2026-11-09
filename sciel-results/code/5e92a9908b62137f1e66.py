"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    n = 128
    numbers = list(range(1, n + 1))
    best_set = set()
    best_size = 0
    
    start_time = time.time()
    time_limit = 55  # Leave 5s buffer
    
    while time.time() - start_time < time_limit:
        random.shuffle(numbers)
        S = set()
        
        for x in numbers:
            valid = True
            # Find all a in S such that a+x is also in S
            # These are the only candidates for 'a' in the triple {a,b,x}
            candidates = []
            for a in S:
                if a + x in S:
                    candidates.append(a)
            
            # Now check pairs from candidates
            len_cand = len(candidates)
            for i in range(len_cand):
                a = candidates[i]
                for j in range(i + 1, len_cand):
                    b = candidates[j]
                    # a and b are distinct because i != j
                    # We need a+b to be in S or equal to x
                    ab = a + b
                    if ab == x or ab in S:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    return sorted(list(best_set))
# EVOLVE-BLOCK-END

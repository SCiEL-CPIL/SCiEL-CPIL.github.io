import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 60.0  # 60s limit, keep 30s buffer for safety? No, 90s kill. Use 85s.
    
    best_set = []
    best_size = 0
    
    # Helper to check if x can be added to S
    def can_add(x, S):
        # S is a list of ints
        # 1. x must not divide sum of any 3 distinct elements in S
        # Note: if len(S) < 3, this part is vacuously true.
        if len(S) >= 3:
            for a, b, c in combinations(S, 3):
                if (a + b + c) % x == 0:
                    return False
        
        # 2. No a in S must divide sum of any 3 distinct elements in S U {x} that include x
        # The triples are {x, u, v} where u, v in S \ {a}
        # So we check if a | (x + u + v) for any u, v in S \ {a}
        for a in S:
            # We need pairs u, v from S excluding a
            # Optimization: if len(S) < 2, no pairs.
            if len(S) < 2:
                continue
            # Create list of S without a? 
            # Or just loop and skip a.
            # To avoid list creation, we can do:
            # for i in range(len(S)):
            #   u = S[i]
            #   if u == a: continue
            #   for j in range(i+1, len(S)):
            #      v = S[j]
            #      if v == a: continue
            #      ...
            # This is messy. Let's just filter.
            others = [v for v in S if v != a]
            if len(others) >= 2:
                for u, v in combinations(others, 2):
                    if (x + u + v) % a == 0:
                        return False
        
        return True

    # Initial greedy run (deterministic, high to low)
    # This is the seed behavior, but we'll do randomized ones.
    
    # Randomized Greedy
    for i in range(200): # Try many runs
        if time.time() > deadline:
            break
        
        current_set = []
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, current_set):
                current_set.append(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # If we hit the theoretical max (48), we can stop early?
            # The task says maximum=48. If we reach 48, it's optimal.
            if best_size == 48:
                break

    # If we haven't found 48, we might try local search on the best set
    # But given time constraints, multiple greedy runs are a good baseline improvement.
    
    return best_set

import random
import time
from itertools import combinations

def construct():
    best_set = []
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer
    
    # Helper to check if x can be added to S
    def can_add(S, x):
        # 1. Check if x divides sum of any 3 distinct elements in S
        if len(S) >= 3:
            for t in combinations(S, 3):
                if sum(t) % x == 0:
                    return False
        
        # 2. Check if any a in S divides sum of any 3 distinct elements in S U {x} excluding a
        # The new sums involving x are x + s1 + s2 where s1, s2 in S \ {a}
        for a in S:
            # We need to check sums of 3 distinct elements from (S U {x}) \ {a}
            # These are either:
            # - 3 elements from S \ {a} (already valid by assumption that S is valid)
            # - 2 elements from S \ {a} and x
            # So we only need to check x + s1 + s2
            others = [s for s in S if s != a]
            if len(others) >= 2:
                for s1, s2 in combinations(others, 2):
                    if (x + s1 + s2) % a == 0:
                        return False
        return True

    # Randomized Greedy
    def greedy_random():
        S = []
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(S, x):
                S.append(x)
        return S

    # Run multiple trials
    # We can also try different biases: e.g., prefer larger numbers, or smaller numbers.
    # The original greedy (largest first) is a specific case.
    # Let's mix strategies.
    
    # Strategy 1: Pure Random
    # Strategy 2: Random but biased towards larger numbers (like original but shuffled)
    # Strategy 3: Random but biased towards smaller numbers
    
    for i in range(10000): # Upper bound, will break on time
        if time.time() - start_time > time_limit:
            break
            
        # Occasionally use the original deterministic greedy to ensure we don't lose the baseline
        if i == 0:
            S = []
            for x in range(192, 0, -1):
                if can_add(S, x):
                    S.append(x)
        else:
            # Randomize the candidate order
            # We can bias the shuffle
            S = greedy_random()
            
        if len(S) > len(best_set):
            best_set = S
            # If we reach 48, we can stop early as it's the max possible
            if len(best_set) == 48:
                break

    return best_set

import random
import time

def construct():
    n = 512
    best_set = set()
    best_size = 0
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s kill
    
    # Function to check if x can be added to S without forming a 4-AP
    def can_add(S, x):
        # x is the 1st term: x, x+d, x+2d, x+3d
        max_d1 = (n - x) // 3
        for d in range(1, max_d1 + 1):
            if (x + d) in S and (x + 2*d) in S and (x + 3*d) in S:
                return False
        
        # x is the 2nd term: x-d, x, x+d, x+2d
        max_d2 = min((n - x) // 2, x - 1)
        for d in range(1, max_d2 + 1):
            if (x - d) in S and (x + d) in S and (x + 2*d) in S:
                return False
        
        # x is the 3rd term: x-2d, x-d, x, x+d
        max_d3 = min((x - 1) // 2, n - x)
        for d in range(1, max_d3 + 1):
            if (x - 2*d) in S and (x - d) in S and (x + d) in S:
                return False
        
        # x is the 4th term: x-3d, x-2d, x-d, x
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if (x - 3*d) in S and (x - 2*d) in S and (x - d) in S:
                return False
        
        return True

    def greedy_random():
        S = set()
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(S, x):
                S.add(x)
        return S

    def improve(S):
        S = set(S)
        # Initial Greedy Fill (deterministic order 1..n is fine for filling gaps)
        for x in range(1, n + 1):
            if x not in S and can_add(S, x):
                S.add(x)
        
        best_local_S = set(S)
        best_local_size = len(S)
        
        local_start = time.time()
        # Local Search with Kicks
        # We try to remove a random element and refill.
        # We only keep the result if it's larger.
        # This is a form of stochastic hill climbing.
        
        while time.time() - local_start < 2.0: # 2 seconds per branch
            if not S:
                break
            
            # Pick a random element to remove
            rem = random.choice(list(S))
            S.remove(rem)
            
            # Greedy Fill again
            for x in range(1, n + 1):
                if x not in S and can_add(S, x):
                    S.add(x)
            
            if len(S) > best_local_size:
                best_local_size = len(S)
                best_local_S = set(S)
            else:
                # Revert to the best state found so far in this branch
                # This ensures we don't drift away from the local optimum
                # while still allowing us to try different kicks from that optimum
                S = set(best_local_S)
                
                # Optional: If we are stuck for a while, we could break,
                # but the time limit handles termination.
                
        return best_local_S

    # Main Loop
    while time.time() - start_time < time_limit:
        S = greedy_random()
        S = improve(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
        # Early termination if we reach a known high bound
        # Literature suggests max is around 160-170 for n=512
        if best_size >= 170:
            break

    return sorted(best_set)

import random
import time

def construct():
    N = 256
    candidates = list(range(2, N + 1))
    best_set = []
    
    # Precompute prime factors or just use direct divisibility
    # Direct divisibility is fast enough for small numbers
    
    def is_valid_addition(x, S):
        # Check if x can be added to S
        # For all pairs (b, c) in S, b != c:
        # 1. x does not divide b * c
        # 2. b does not divide x * c
        # 3. c does not divide x * b
        for i in range(len(S)):
            b = S[i]
            for j in range(i + 1, len(S)):
                c = S[j]
                if (b * c) % x == 0:
                    return False
                if (x * c) % b == 0:
                    return False
                if (x * b) % c == 0:
                    return False
        return True

    # Run randomized greedy multiple times
    num_restarts = 200
    for _ in range(num_restarts):
        S = []
        remaining = candidates[:]
        random.shuffle(remaining)
        
        # Greedy pass
        for x in remaining:
            if is_valid_addition(x, S):
                S.append(x)
        
        if len(S) > len(best_set):
            best_set = S[:]
            
            # Optional: Local search?
            # Try to add more by swapping?
            # For now, just keep the best greedy result.
            
    return [int(x) for x in best_set]

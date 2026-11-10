import time
import random

def construct():
    N = 1000
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    best_size = 0
    best_set = []
    
    # Helper to check and add
    def greedy_build(order):
        S = []
        counts = [0] * (2 * N + 1)
        for x in order:
            valid = True
            # Check sums x + y for y in S
            for y in S:
                s = x + y
                if counts[s] >= 2:
                    valid = False
                    break
            # Check sum 2x
            if valid:
                s = 2 * x
                if counts[s] >= 2:
                    valid = False
            
            if valid:
                S.append(x)
                # Update counts
                for y in S:
                    counts[x + y] += 1
                counts[2 * x] += 1
        return S

    # 1. Natural order (same as seed, but maybe slightly optimized)
    order = list(range(1, N + 1))
    S = greedy_build(order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # 2. Reverse order
    if time.time() < deadline:
        order = list(range(N, 0, -1))
        S = greedy_build(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # 3. Random shuffles
    # Run many random restarts
    while time.time() < deadline:
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = greedy_build(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    return best_set

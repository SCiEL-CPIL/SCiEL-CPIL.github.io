import random
import time

def construct():
    N = 1000
    max_time = 110.0
    
    # Helper to calculate violation cost
    def calc_violation(counts):
        v = 0
        for i in range(2, 2 * N + 1):
            c = counts[i]
            if c > 2:
                v += (c - 2) * (c - 2)
        return v

    # Greedy initial solution
    def greedy_initial():
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        S = set()
        counts = [0] * (2 * N + 1)
        
        for x in candidates:
            valid = True
            if counts[2 * x] >= 2:
                valid = False
            else:
                for y in S:
                    if counts[x + y] >= 2:
                        valid = False
                        break
            if valid:
                S.add(x)
                counts[2 * x] += 1
                for y in S:
                    if y != x:
                        counts[x + y] += 1
        return S, counts

    S, counts = greedy_initial()
    current_violation = calc_violation(counts)
    current_size = len(S)
    
    # Energy function: Minimize -Size + Penalty * Violation
    # We want to maximize Size, so -Size.
    # Penalty should be high enough to force validity.
    SIZE_WEIGHT = 1
    PENALTY_WEIGHT = 10000
    
    current_energy = -current_size * SIZE_WEIGHT + current_violation * PENALTY_WEIGHT
    
    best_S = S.copy()
    best_size = current_size
    best_energy = current_energy
    
    T = 50.0
    T_min = 0.01
    alpha = 0.99995
    start_time = time.time()
    
    # Precompute list of all numbers for fast random selection
    all_nums = list(range(1, N + 1))
    
    while T > T_min and (time.time() - start_time) < max_time:
        # Generate a move
        # 1. Add random element not in S
        # 2. Remove random element in S
        # 3. Swap
        
        # To avoid O(N) list creation for 'not_in_S' every time, we can use a different strategy
        # Or just accept the cost. N=1000 is small.
        
        move = random.random()
        if move < 0.4:
            # Add
            if len(S) < N:
                # Pick random until found not in S? Or build list?
                # Building list is O(N).
                not_in_S = [x for x in all_nums if x not in S]
                if not_in_S:
                    x = random.choice(not_in_S)
                    S_new = S | {x}
                    counts_new = counts[:]
                    counts_new[2 * x] += 1
                    for y in S:
                        counts_new[x + y] += 1
                    new_violation = calc_violation(counts_new)
                    new_size = len(S_new)
                    new_energy = -new_size * SIZE_WEIGHT + new_violation * PENALTY_WEIGHT
                    dE = new_energy - current_energy
                    if dE < 0 or random.random() < pow(2.718281828, -dE / T):
                        S = S_new
                        counts = counts_new
                        current_violation = new_violation
                        current_size = new_size
                        current_energy = new_energy
                        if current_violation == 0 and current_size > best_size:
                            best_size = current_size
                            best_S = S.copy()
                            best_energy = current_energy
        elif move < 0.7:
            # Remove
            if S:
                y = random.choice(list(S))
                S_new = S - {y}
                counts_new = counts[:]
                counts_new[2 * y] -= 1
                for x in S:
                    if x != y:
                        counts_new[x + y] -= 1
                new_violation = calc_violation(counts_new)
                new_size = len(S_new)
                new_energy = -new_size * SIZE_WEIGHT + new_violation * PENALTY_WEIGHT
                dE = new_energy - current_energy
                if dE < 0 or random.random() < pow(2.718281828, -dE / T):
                    S = S_new
                    counts = counts_new
                    current_violation = new_violation
                    current_size = new_size
                    current_energy = new_energy
                    if current_violation == 0 and current_size > best_size:
                        best_size = current_size
                        best_S = S.copy()
                        best_energy = current_energy
        else:
            # Swap
            if S and len(S) < N:
                not_in_S = [x for x in all_nums if x not in S]
                if not_in_S:
                    x = random.choice(not_in_S)
                    y = random.choice(list(S))
                    S_new = (S - {y}) | {x}
                    counts_new = counts[:]
                    # Remove y
                    counts_new[2 * y] -= 1
                    for z in S:
                        if z != y:
                            counts_new[y + z] -= 1
                    # Add x
                    counts_new[2 * x] += 1
                    for z in S_new:
                        if z != x:
                            counts_new[x + z] += 1
                    new_violation = calc_violation(counts_new)
                    new_size = len(S_new)
                    new_energy = -new_size * SIZE_WEIGHT + new_violation * PENALTY_WEIGHT
                    dE = new_energy - current_energy
                    if dE < 0 or random.random() < pow(2.718281828, -dE / T):
                        S = S_new
                        counts = counts_new
                        current_violation = new_violation
                        current_size = new_size
                        current_energy = new_energy
                        if current_violation == 0 and current_size > best_size:
                            best_size = current_size
                            best_S = S.copy()
                            best_energy = current_energy
        
        T *= alpha

    return list(best_S)

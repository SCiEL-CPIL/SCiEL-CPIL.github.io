import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    def check_valid_addition(S, x, sums_set):
        S_list = list(S)
        n = len(S_list)
        new_sums = []
        new_sums.append(3*x)
        for s in S_list:
            new_sums.append(2*x + s)
        for i in range(n):
            for j in range(i, n):
                new_sums.append(x + S_list[i] + S_list[j])
        for s in new_sums:
            if s in sums_set:
                return False, None
        if len(new_sums) != len(set(new_sums)):
            return False, None
        return True, new_sums
    
    def greedy_forward(candidates):
        S = []
        sums_set = set()
        for x in candidates:
            if x < 1 or x > 4096:
                continue
            valid, new_sums = check_valid_addition(S, x, sums_set)
            if valid:
                S.append(x)
                for s in new_sums:
                    sums_set.add(s)
        return S, sums_set
    
    best = []
    best_sums = set()
    
    # Greedy from 1 to 4096
    S, sums_set = greedy_forward(range(1, 4097))
    if len(S) > len(best):
        best = S[:]
        best_sums = sums_set.copy()
    
    # Multiple random restarts with different biases
    rng = random.Random(42)
    for attempt in range(200):
        if time.time() > deadline - 1.0:
            break
        # Mix of strategies
        strategy = attempt % 4
        if strategy == 0:
            candidates = list(range(1, 4097))
            rng.shuffle(candidates)
        elif strategy == 1:
            # Bias toward smaller numbers first
            candidates = list(range(1, 4097))
            rng.shuffle(candidates)
            candidates.sort(key=lambda x: rng.random() * x)
        elif strategy == 2:
            # Bias toward larger numbers
            candidates = list(range(1, 4097))
            rng.shuffle(candidates)
            candidates.sort(key=lambda x: -rng.random() * x)
        else:
            # Random subset
            candidates = rng.sample(range(1, 4097), min(1000, 4096))
        
        S, sums_set = greedy_forward(candidates)
        if len(S) > len(best):
            best = S[:]
            best_sums = sums_set.copy()
    
    return best


import random
import time

def score_set(S):
    S_set = set(S)
    good_d = set()
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a = S[i]
            b = S[j]
            d = b - a
            c = b + d
            if c in S_set:
                good_d.add(d)
    return len(good_d)

def run_search(initial_S, time_budget):
    start_time = time.time()
    S = initial_S.copy()
    current_score = score_set(S)
    best_S = S.copy()
    best_score = current_score
    
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_budget:
        idx = random.randint(0, 31)
        old_val = S[idx]
        
        # Generate a new value that is distinct from others
        new_val = None
        for _ in range(10):
            candidate = random.randint(1, 512)
            if candidate == old_val:
                continue
            if candidate not in S:
                new_val = candidate
                break
        
        if new_val is None:
            new_val = random.randint(1, 512)
            if new_val in S:
                continue

        new_S = S.copy()
        new_S[idx] = new_val
        
        if len(set(new_S)) < 32:
            continue

        new_score = score_set(new_S)
        
        if new_score > current_score:
            S = new_S
            current_score = new_score
            if current_score > best_score:
                best_S = S.copy()
                best_score = current_score
        else:
            delta = current_score - new_score
            if delta == 0:
                accept = True
            else:
                accept = random.random() < pow(2.71828, -delta / T)
            
            if accept:
                S = new_S
                current_score = new_score
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_S, best_score

def construct():
    start_time = time.time()
    total_time = 115
    
    best_S = None
    best_score = 0
    
    # Run 1: Start from seed 1..32
    seed_S = list(range(1, 33))
    S1, score1 = run_search(seed_S, total_time / 4)
    if score1 > best_score:
        best_S = S1
        best_score = score1
        
    # Run 2: Start from random
    random_S = random.sample(range(1, 513), 32)
    S2, score2 = run_search(random_S, total_time / 4)
    if score2 > best_score:
        best_S = S2
        best_score = score2
        
    # Run 3: Start from another random
    random_S2 = random.sample(range(1, 513), 32)
    S3, score3 = run_search(random_S2, total_time / 4)
    if score3 > best_score:
        best_S = S3
        best_score = score3
        
    # Run 4: Start from a mixed structure
    mixed_S = list(range(1, 17)) + list(range(100, 116))
    S4, score4 = run_search(mixed_S, total_time / 4)
    if score4 > best_score:
        best_S = S4
        best_score = score4
        
    return best_S

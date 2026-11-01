import time
import random
import math

def calculate_score(U):
    if not U:
        return 0.0
    U = sorted(list(set(U)))
    if 0 not in U:
        U = sorted(list(set(U) | {0}))
        
    max_u = max(U)
    if max_u == 0:
        return 1.0
        
    sums = set()
    diffs = set()
    n = len(U)
    for i in range(n):
        ui = U[i]
        for j in range(n):
            uj = U[j]
            sums.add(ui + uj)
            diffs.add(ui - uj)
            
    num = len(diffs)
    den = len(sums)
    if den == 0:
        return 0.0
    ratio = num / den
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_score = 1.0
    best_set = [0, 1, 2]
    
    # High quality seeds based on manual analysis
    seeds = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 1, 2, 6],
        [0, 1, 2, 5],
        [0, 1, 3, 7],
        [0, 1, 4, 9],
        [0, 1, 4, 10],
        [0, 1, 2, 4, 8],
        [0, 1, 2, 5, 10],
        [0, 1, 3, 7, 12],
        [0, 1, 4, 10, 18],
        [0, 1, 2, 6, 12],
        [0, 1, 3, 6, 12],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 3, 7, 15, 31]
    ]
    
    for s in seeds:
        score = calculate_score(s)
        if score > best_score:
            best_score = score
            best_set = list(s)

    random.seed(42)
    
    def local_search(current_set, current_score, time_limit):
        nonlocal best_score, best_set
        
        no_improvement = 0
        while time.time() < time_limit and no_improvement < 200:
            improved = False
            # Generate neighbors
            neighbors = []
            
            # 1. Add element
            if len(current_set) < 12:
                max_val = max(current_set)
                for _ in range(10):
                    new_val = random.randint(1, max_val + 5)
                    if new_val not in current_set:
                        neighbors.append(sorted(current_set + [new_val]))
            
            # 2. Remove element
            if len(current_set) > 3:
                for i in range(1, len(current_set)):
                    neighbors.append(sorted([x for j, x in enumerate(current_set) if j != i]))
            
            # 3. Move element
            for i in range(1, len(current_set)):
                for delta in [-2, -1, 1, 2, -3, 3]:
                    new_val = current_set[i] + delta
                    if new_val >= 0 and new_val not in current_set:
                        cand = list(current_set)
                        cand[i] = new_val
                        neighbors.append(sorted(set(cand)))
            
            # Evaluate neighbors
            for neighbor in neighbors:
                if not neighbor or 0 not in neighbor:
                    continue
                s = calculate_score(neighbor)
                if s > current_score:
                    current_set = neighbor
                    current_score = s
                    no_improvement = 0
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                    break
            if not improved:
                no_improvement += 1
                
        return current_set, current_score

    # Run local search on best seed and a few others
    time_per_search = 10.0
    for s in seeds[:5]:
        if time.time() > deadline - 5:
            break
        s_copy = list(s)
        s_score = calculate_score(s_copy)
        local_search(s_copy, s_score, deadline - 5)

    # Random restarts
    while time.time() < deadline - 10:
        n = random.randint(3, 10)
        max_u = random.randint(n, n*n)
        # Generate random set
        inner = list(range(1, max_u+1))
        if len(inner) < n-1:
            continue
        selected = random.sample(inner, n-1)
        current_set = sorted([0] + selected)
        current_score = calculate_score(current_set)
        
        local_search(current_set, current_score, deadline - 10)

    return best_set

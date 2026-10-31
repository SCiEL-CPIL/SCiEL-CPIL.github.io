import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_score = -1
    best_set = [0]
    
    def compute_score(U):
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        U_arr = np.array(U, dtype=np.int64)
        
        # Vectorized sumset and diffset
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        
        n_sum = len(np.unique(sums))
        n_diff = len(np.unique(diffs))
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Phase 1: Generate diverse seeds
    seeds = []
    
    # 1. Known Golomb Rulers (Optimal for small n)
    # These maximize |U-U| for given max(U)
    golomb_rulers = [
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 31, 34],
        [0, 1, 4, 10, 18, 23, 31, 42, 45],
        [0, 1, 4, 10, 18, 23, 31, 42, 49, 54],
        [0, 1, 4, 10, 18, 23, 31, 42, 49, 62, 65],
        [0, 1, 4, 10, 18, 23, 31, 42, 49, 62, 77, 80],
        [0, 1, 4, 10, 18, 23, 31, 42, 49, 62, 77, 94, 97],
        [0, 1, 4, 10, 18, 23, 31, 42, 49, 62, 77, 94, 113, 116],
        [0, 1, 4, 10, 18, 23, 31, 42, 49, 62, 77, 94, 113, 131, 134],
    ]
    seeds.extend(golomb_rulers)
    
    # 2. Random Sidon Sets
    for _ in range(100):
        n = random.randint(5, 40)
        U = [0]
        current_max = 0
        for i in range(1, n):
            attempts = 0
            while attempts < 100:
                x = random.randint(1, max(10, current_max * 2))
                if x not in U:
                    U.append(x)
                    current_max = max(current_max, x)
                    break
                attempts += 1
            if attempts >= 100:
                break
        if len(U) > 2:
            seeds.append(U)
            
    # 3. "Clumped" sets
    for _ in range(50):
        n_clumps = random.randint(2, 8)
        U = [0]
        pos = 0
        for i in range(n_clumps):
            pos += random.randint(5, 30)
            for j in range(random.randint(2, 4)):
                x = pos + j
                if x not in U:
                    U.append(x)
        seeds.append(U)

    # Phase 2: Local Search from each seed
    time_per_seed = 1.5
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
            
        current = sorted(set(seed))
        if 0 not in current:
            current.append(0)
            current.sort()
            
        current_score = compute_score(current)
        if current_score > best_score:
            best_score = current_score
            best_set = list(current)
            
        # Simulated Annealing / Local Search
        T = 0.005  
        min_T = 0.0001
        cooling = 0.99
        no_improve = 0
        max_no_improve = 300
        
        seed_start_time = time.time()
        
        while time.time() - seed_start_time < time_per_seed and no_improve < max_no_improve:
            if time.time() > deadline - 10:
                break
                
            improved = False
            max_cur = max(current)
            
            # Generate a neighbor
            move_type = random.random()
            
            if move_type < 0.4 and len(current) < 80:
                # Add an element
                new_el = random.randint(1, max_cur + 50)
                if new_el not in current:
                    cand = sorted(current + [new_el])
                    new_score = compute_score(cand)
                    if new_score > current_score:
                        current = cand
                        current_score = new_score
                        improved = True
                        no_improve = 0
                    else:
                        if random.random() < math.exp((new_score - current_score) / T):
                            current = cand
                            current_score = new_score
                            no_improve = 0
                        else:
                            no_improve += 1
            elif move_type < 0.8 and len(current) > 2:
                # Replace an element
                i = random.randint(1, len(current) - 1)
                old_el = current[i]
                if random.random() < 0.5:
                    new_el = random.randint(1, max(1, old_el + 20))
                else:
                    new_el = random.randint(1, max(1, max_cur + 20))
                
                if new_el != old_el and new_el not in current:
                    cand = list(current)
                    cand[i] = new_el
                    cand = sorted(set(cand))
                    if 0 in cand:
                        new_score = compute_score(cand)
                        if new_score > current_score:
                            current = cand
                            current_score = new_score
                            improved = True
                            no_improve = 0
                        else:
                            if random.random() < math.exp((new_score - current_score) / T):
                                current = cand
                                current_score = new_score
                                no_improve = 0
                            else:
                                no_improve += 1
            elif move_type < 0.9 and len(current) > 3:
                # Remove an element
                i = random.randint(1, len(current) - 1)
                cand = [x for j, x in enumerate(current) if j != i]
                new_score = compute_score(cand)
                if new_score > current_score:
                    current = cand
                    current_score = new_score
                    improved = True
                    no_improve = 0
                else:
                    if random.random() < math.exp((new_score - current_score) / T):
                        current = cand
                        current_score = new_score
                        no_improve = 0
                    else:
                        no_improve += 1
            else:
                # Swap two elements
                if len(current) > 3:
                    i = random.randint(1, len(current) - 1)
                    j = random.randint(1, len(current) - 1)
                    if i != j:
                        cand = list(current)
                        cand[i], cand[j] = cand[j], cand[i]
                        cand = sorted(set(cand))
                        if 0 in cand:
                            new_score = compute_score(cand)
                            if new_score > current_score:
                                current = cand
                                current_score = new_score
                                improved = True
                                no_improve = 0
                            else:
                                if random.random() < math.exp((new_score - current_score) / T):
                                    current = cand
                                    current_score = new_score
                                    no_improve = 0
                                else:
                                    no_improve += 1
            
            if improved:
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current)
            
            T *= cooling
            if T < min_T:
                T = min_T

    # Final Polish: Greedy local search on best_set
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        T = 0.0001
        cooling = 0.95
        no_improve = 0
        max_no_improve = 1000
        polish_start = time.time()
        
        while time.time() - polish_start < 5 and no_improve < max_no_improve:
            improved = False
            max_cur = max(current)
            move_type = random.random()
            
            if move_type < 0.4 and len(current) < 80:
                new_el = random.randint(1, max_cur + 20)
                if new_el not in current:
                    cand = sorted(current + [new_el])
                    new_score = compute_score(cand)
                    if new_score > current_score:
                        current = cand
                        current_score = new_score
                        improved = True
                        no_improve = 0
                    else:
                        if random.random() < math.exp((new_score - current_score) / T):
                            current = cand
                            current_score = new_score
                            no_improve = 0
                        else:
                            no_improve += 1
            elif move_type < 0.8 and len(current) > 2:
                i = random.randint(1, len(current) - 1)
                old_el = current[i]
                new_el = random.randint(1, max(1, old_el + 10))
                if new_el != old_el and new_el not in current:
                    cand = list(current)
                    cand[i] = new_el
                    cand = sorted(set(cand))
                    if 0 in cand:
                        new_score = compute_score(cand)
                        if new_score > current_score:
                            current = cand
                            current_score = new_score
                            improved = True
                            no_improve = 0
                        else:
                            if random.random() < math.exp((new_score - current_score) / T):
                                current = cand
                                current_score = new_score
                                no_improve = 0
                            else:
                                no_improve += 1
            elif move_type < 0.9 and len(current) > 3:
                i = random.randint(1, len(current) - 1)
                cand = [x for j, x in enumerate(current) if j != i]
                new_score = compute_score(cand)
                if new_score > current_score:
                    current = cand
                    current_score = new_score
                    improved = True
                    no_improve = 0
                else:
                    if random.random() < math.exp((new_score - current_score) / T):
                        current = cand
                        current_score = new_score
                        no_improve = 0
                    else:
                        no_improve += 1
            else:
                if len(current) > 3:
                    i = random.randint(1, len(current) - 1)
                    j = random.randint(1, len(current) - 1)
                    if i != j:
                        cand = list(current)
                        cand[i], cand[j] = cand[j], cand[i]
                        cand = sorted(set(cand))
                        if 0 in cand:
                            new_score = compute_score(cand)
                            if new_score > current_score:
                                current = cand
                                current_score = new_score
                                improved = True
                                no_improve = 0
                            else:
                                if random.random() < math.exp((new_score - current_score) / T):
                                    current = cand
                                    current_score = new_score
                                    no_improve = 0
                                else:
                                    no_improve += 1
            
            if improved:
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current)
            
            T *= cooling
            if T < 0.00001:
                T = 0.00001

    return list(best_set)

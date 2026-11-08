import time
from math import log, floor, sqrt, exp
import random
import numpy as np
from itertools import combinations

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def geometric_set(b, m):
    """Generate U = {floor(b^k) for k=0..m}, ensuring 0 is included and all are nonnegative distinct integers."""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = floor(b ** k)
        if val <= 0:
            val = 1
        if val not in seen:
            U.append(val)
            seen.add(val)
    return sorted(U)

def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    """
    U_set = set(U)
    candidates = set()
    
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
            
    return list(candidates)

def local_search_dynamic(U, deadline, max_val_limit=200, max_n_limit=10):
    """
    Local search with dynamic candidate generation.
    Optimized for small sets.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically
        if not candidates or random.random() < 0.15:
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = generate_candidates(U, new_limit)
            # Also add some random small numbers
            for _ in range(10):
                r = random.randint(1, max_val_limit)
                if r not in U:
                    candidates.append(r)
        
        if move_type < 0.4:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                random.shuffle(candidates)
                for c in candidates[:15]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.7:
            # Try to REMOVE an element
            if len(U) > 3:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max_val_limit)
        else:
            # Try to REPLACE an element
            if len(U) > 2 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                random.shuffle(candidates)
                for new_val in candidates[:15]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def simulated_annealing(U, deadline, max_iter=2000):
    """
    Simulated annealing.
    """
    U = sorted(set(U))
    n = len(U)
    current_score = score_set(U)
    best_score = current_score
    best_set = list(U)
    
    T_start = 0.05
    T_end = 0.0001
    cooling_rate = (T_end / T_start) ** (1.0 / max(max_iter, 1))
    T = T_start
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to ADD an element
            if len(U) < 10:
                max_val = max(U)
                if random.random() < 0.5:
                    base = random.choice(U)
                    new_val = base + random.randint(1, max(1, int(max_val * 0.2)))
                else:
                    new_val = random.randint(1, max_val * 2 + 5)
                
                if new_val > 0 and new_val not in U:
                    new_U = sorted(U + [new_val])
                    new_score = score_set(new_U)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < exp(delta / max(T, 1e-10)):
                        U = new_U
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(U)
        
        elif move_type < 0.6:
            # Try to REMOVE an element
            if len(U) > 3:
                idx = random.randint(1, len(U) - 1)
                new_U = [U[j] for j in range(len(U)) if j != idx]
                new_score = score_set(new_U)
                delta = new_score - current_score
                if delta > 0 or random.random() < exp(delta / max(T, 1e-10)):
                    U = new_U
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(U)
        
        else:
            # Try to REPLACE an element
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                max_val = max(U)
                if random.random() < 0.5:
                    base = random.choice(U)
                    new_val = base + random.randint(1, max(1, int(max_val * 0.15)))
                else:
                    new_val = random.randint(1, max_val * 2 + 5)
                
                if new_val > 0 and new_val not in U:
                    new_U = [U[j] if j != idx else new_val for j in range(len(U))]
                    new_U = sorted(set(new_U))
                    if len(new_U) == len(U):
                        new_score = score_set(new_U)
                        delta = new_score - current_score
                        if delta > 0 or random.random() < exp(delta / max(T, 1e-10)):
                            U = new_U
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(U)
        
        T *= cooling_rate
    
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    best_set = [0, 1, 2, 4, 8]
    best_score = score_set(best_set)

    # 1. Structured Candidates
    structured_candidates = [
        [0, 1, 2, 4, 8],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 3, 7, 15],
        [0, 1, 4, 10, 18],
        [0, 1, 4, 9, 16],
        [0, 1, 2, 5, 10],
        [0, 1, 3, 4, 9],
        [0, 1, 2, 4, 7, 12],
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Exhaustive search for small sets
    # Search for n=3,4,5 with max_val <= 30
    if time.time() < deadline - 60:
        max_val = 30
        for n in range(3, 6):
            # Generate all combinations of size n from 0..max_val
            # 0 must be included
            pool = list(range(1, max_val + 1))
            for comb in combinations(pool, n - 1):
                U = [0] + list(comb)
                U.sort()
                s = score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U
            if time.time() > deadline - 30:
                break

    # 3. Local Search from best set
    seeds = [best_set]
    for cand in structured_candidates[:3]:
        if cand != best_set:
            seeds.append(cand)
    
    max_val_limit = 200
    max_n_limit = 10
    
    restarts = 0
    while time.time() < deadline - 20:
        restarts += 1
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        local_score, local_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            if len(seeds) < 5:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set

    # 4. Simulated Annealing
    if time.time() < deadline - 10:
        sa_score, sa_set = simulated_annealing(list(best_set), deadline)
        if sa_score > best_score:
            best_score = sa_score
            best_set = sa_set

    # 5. Final Polish
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 1 and iteration < 200:
            improved = False
            iteration += 1
            n = len(current)
            # Try replacing each element with nearby values
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                for delta in range(-10, 11):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding an element
            if not improved and n < 10 and time.time() < deadline - 1:
                max_val = max(current)
                for new_val in range(max_val - 5, max_val + 6):
                    if new_val <= 0 or new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

            # Try removing an element
            if not improved and n > 3 and time.time() < deadline - 1:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))

import math
import random

def calc_score(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
            
    num = len(diffs) / len(sums)
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 0
    return 1 + math.log(num) / den

def local_search(U_full, size, max_iters=100, seed=42):
    random.seed(seed)
    if size < 1 or size > len(U_full):
        return None, 0
        
    U_set = {0}
    if size > 1:
        candidates = [x for x in U_full if x != 0]
        if len(candidates) < size - 1:
            return None, 0
        U_set.update(random.sample(candidates, size - 1))
    
    current_score = calc_score(list(U_set))
    best_score = current_score
    best_set = set(U_set)
    
    for _ in range(max_iters):
        if len(U_set) < 2:
            break
        to_remove = random.choice([x for x in U_set if x != 0])
        to_add_candidates = [x for x in U_full if x not in U_set]
        if not to_add_candidates:
            break
        to_add = random.choice(to_add_candidates)
        
        U_new = set(U_set)
        U_new.remove(to_remove)
        U_new.add(to_add)
        
        new_score = calc_score(list(U_new))
        if new_score > current_score:
            current_score = new_score
            U_set = U_new
            if current_score > best_score:
                best_score = current_score
                best_set = set(U_set)
                
    return list(best_set), best_score

def construct():
    best = [0, 1, 3]
    best_score = calc_score(best)
    
    # 1. Full sets search
    for B in range(2, 30):
        for n in range(2, 11):
            all_vals = []
            for mask in range(1 << n):
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                all_vals.append(val)
            
            U = all_vals
            s = calc_score(U)
            if s > best_score:
                best_score = s
                best = U
                
    # 2. Subset search for small B, n
    # N = 2^n. For n=6, N=64.
    # Local search is O(iters * N^2).
    # 100 iters * 4096 ops = 400k ops per run.
    # 5 seeds * 5 sizes * 4 B * 5 n = 500 runs.
    # 500 * 400k = 200M ops. Might be tight in Python.
    # Reduce to n=5 (N=32) and 3 seeds.
    for B in [2, 3, 4, 5]:
        for n in range(2, 7): # n up to 6
            all_vals = []
            for mask in range(1 << n):
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                all_vals.append(val)
                
            N = len(all_vals)
            target_sizes = [2, 3, 4, 5, 8, 12, 16]
            target_sizes = [s for s in target_sizes if s <= N]
            
            for size in target_sizes:
                for seed in range(3):
                    U_sub, s_sub = local_search(all_vals, size, max_iters=50, seed=seed + B * 100 + n)
                    if U_sub and s_sub > best_score:
                        best_score = s_sub
                        best = U_sub
                        
    return best

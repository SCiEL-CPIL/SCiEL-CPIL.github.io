import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    best_score = -1
    best_set = [0, 1, 3]

    def compute_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        if len(U) < 2:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 0
        
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
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
        if len(cand) < 2:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Phase 1: Exhaustive search
    # Increase M limit to 50 to be safe
    max_m_exhaustive = 50
    max_n_exhaustive = 6
    
    for n in range(2, max_n_exhaustive + 1):
        range_m = range(1, max_m_exhaustive + 1)
        for combo in combinations(range_m, n - 1):
            cand = [0] + list(combo)
            if time.time() > deadline - 20:
                break
            try_set(cand)
        if time.time() > deadline - 20:
            break

    # Phase 2: Local Search
    if time.time() < deadline - 5:
        current = list(best_set)
        no_improve_count = 0
        max_iterations = 20000
        
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # 1. Try adding an element
            if len(current) < 8:
                max_cur = max(current)
                search_limit = max(max_cur + 5, 30)
                for new_el in range(1, search_limit + 1):
                    if new_el in current:
                        continue
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        no_improve_count = 0
                        break
            
            # 2. Try replacing an element
            if len(current) >= 2:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    max_cur = max(current)
                    search_limit = max(max_cur + 5, 30)
                    for new_el in range(1, search_limit + 1):
                        if new_el in current:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand or len(cand) < 2:
                            continue
                        s = compute_score(cand)
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                            current = list(cand)
                            improved = True
                            no_improve_count = 0
                            break
                    if improved:
                        break
            
            # 3. Try removing an element
            if len(current) > 2:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        no_improve_count = 0
                        break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 20:
                    # Random restart
                    n = random.randint(3, 6)
                    max_m = random.randint(5, 30)
                    inner = random.sample(range(1, max_m + 1), min(n-1, max_m))
                    cand = sorted([0] + inner)
                    if len(cand) >= 2:
                        s = compute_score(cand)
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                            current = list(cand)
                            improved = True
                            no_improve_count = 0

    return list(best_set)

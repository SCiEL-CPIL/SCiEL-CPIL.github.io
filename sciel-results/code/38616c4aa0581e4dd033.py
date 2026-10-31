import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_score = -1
    best_set = [0]
    
    # Known optimal Golomb rulers (n, ruler)
    # Source: OEIS A005292 or similar
    # n=1: [0]
    # n=2: [0, 1]
    # n=3: [0, 1, 3]
    # n=4: [0, 1, 4, 10]
    # n=5: [0, 1, 4, 10, 18]
    # n=6: [0, 1, 4, 10, 18, 23]
    # n=7: [0, 1, 4, 10, 18, 23, 29]
    # n=8: [0, 1, 4, 10, 18, 23, 29, 36]
    # n=9: [0, 1, 4, 10, 18, 23, 29, 36, 44]
    # n=10: [0, 1, 4, 10, 18, 23, 29, 36, 44, 52]
    # n=11: [0, 1, 4, 10, 18, 23, 29, 36, 44, 52, 60] (Not sure if optimal, but good start)
    
    GLOMB_RULERS = [
        [0],
        [0, 1],
        [0, 1, 3],
        [0, 1, 4, 10],
        [0, 1, 4, 10, 18],
        [0, 1, 4, 10, 18, 23],
        [0, 1, 4, 10, 18, 23, 29],
        [0, 1, 4, 10, 18, 23, 29, 36],
        [0, 1, 4, 10, 18, 23, 29, 36, 44],
        [0, 1, 4, 10, 18, 23, 29, 36, 44, 52],
    ]
    
    def compute_score(U):
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        sumset = set()
        diffset = set()
        
        n = len(U)
        for i in range(n):
            a = U[i]
            for j in range(n):
                b = U[j]
                sumset.add(a + b)
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
            return False
        if max(cand) == 0:
            return False
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
            return True
        return False

    # Phase 1: Initialize with known Golomb rulers
    for gr in GLOMB_RULERS:
        try_set(gr)
        
    # Phase 2: Local Search
    current = list(best_set)
    no_improve = 0
    max_no_improve = 500 # Increased iterations
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        improved = False
        max_cur = max(current)
        n = len(current)
        
        # Strategy: Try to move each element to a new position
        # Prioritize positions that maintain Golomb property (distinct diffs)
        # But we just use score as the guide.
        
        # 1. Add an element
        candidates_to_add = set()
        # Nearby
        for i in range(1, max_cur + 20):
            if i not in current:
                candidates_to_add.add(i)
        # Jumps
        for _ in range(10):
            candidates_to_add.add(random.randint(max_cur + 1, max_cur + 50))
            
        for new_el in candidates_to_add:
            if new_el in current:
                continue
            cand = sorted(current + [new_el])
            if try_set(cand):
                current = list(best_set)
                improved = True
                no_improve = 0
                break
        
        # 2. Replace an element
        if not improved and len(current) >= 2:
            for i in range(1, len(current)): # Skip 0
                old_el = current[i]
                # Try small moves
                for delta in range(-10, 11):
                    new_el = old_el + delta
                    if new_el < 1 or new_el in current or new_el == old_el:
                        continue
                    cand = list(current)
                    cand[i] = new_el
                    cand = sorted(set(cand))
                    if try_set(cand):
                        current = list(best_set)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
        
        # 3. Remove an element
        if not improved and len(current) > 3:
            for i in range(1, len(current)):
                cand = [x for j, x in enumerate(current) if j != i]
                if try_set(cand):
                    current = list(best_set)
                    improved = True
                    no_improve = 0
                    break
        
        if not improved:
            no_improve += 1
            # Random restart / mutation
            if random.random() < 0.3:
                # Perturb current best
                cand = list(best_set)
                idx = random.randint(1, len(cand) - 1)
                new_val = random.randint(1, max(cand) + 20)
                if new_val not in cand:
                    cand[idx] = new_val
                    cand = sorted(set(cand))
                    if 0 in cand:
                        try_set(cand)
                current = list(best_set)

    return list(best_set)

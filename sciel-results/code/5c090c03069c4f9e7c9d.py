import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    best_score = -1
    best_set = [0]
    
    def compute_score(U):
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        sumset = set()
        diffset = set()
        
        # Optimization: iterate over unique pairs for sums? No, just all pairs.
        # For n < 50, n^2 is small.
        for a in U:
            for b in U:
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
            return
        if max(cand) == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Phase 1: Exhaustive search for small M
    # We know small sets can have high scores.
    for M in range(1, 16):
        if time.time() > deadline - 10:
            break
        for mask in range(1, 1 << M):
            if time.time() > deadline - 10:
                break
            cand = [0]
            for i in range(M):
                if mask & (1 << i):
                    cand.append(i + 1)
            try_set(cand)
            
    # Phase 2: Heuristic Search
    # Start from the best set found in Phase 1
    current = list(best_set)
    
    # Also try starting from known good structures like Golomb Rulers
    # Generate a few Golomb Rulers
    def generate_golomb_ruler(n):
        # Simple greedy or precomputed for small n
        # For now, just use local search to find them
        pass

    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        improved = False
        max_cur = max(current)
        
        # 1. Try adding an element
        # Search space for new element: 1 to max_cur + 20
        candidates_to_add = set()
        for i in range(1, max_cur + 20):
            if i not in current:
                candidates_to_add.add(i)
        # Also try some random large jumps
        for _ in range(20):
            candidates_to_add.add(random.randint(max_cur + 1, max_cur + 100))
            
        for new_el in candidates_to_add:
            if new_el in current:
                continue
            cand = sorted(current + [new_el])
            s = compute_score(cand)
            if s > best_score:
                best_score = s
                best_set = list(cand)
                current = list(cand)
                improved = True
                no_improve = 0
                break
        
        # 2. Try replacing an element
        if len(current) >= 3:
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                old_el = current[i]
                # Try replacing with neighbors
                for delta in range(-15, 16):
                    new_el = old_el + delta
                    if new_el < 1 or new_el in current:
                        continue
                    if new_el == old_el:
                        continue
                    cand = list(current)
                    cand[i] = new_el
                    cand = sorted(set(cand))
                    if 0 not in cand:
                        continue
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
        
        # 3. Try removing an element
        if len(current) > 3:
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
                    no_improve = 0
                    break
        
        if not improved:
            no_improve += 1
            # Random perturbation
            if random.random() < 0.3:
                # Swap two elements
                if len(current) > 2:
                    i = random.randint(1, len(current)-1)
                    j = random.randint(1, len(current)-1)
                    if i != j:
                        cand = list(current)
                        cand[i], cand[j] = cand[j], cand[i]
                        cand = sorted(set(cand))
                        if 0 in cand:
                            try_set(cand)
                            current = list(best_set)
                # Or add a random element
                else:
                    new_el = random.randint(1, max_cur + 50)
                    if new_el not in current:
                        cand = sorted(current + [new_el])
                        try_set(cand)
                        current = list(best_set)

    return list(best_set)

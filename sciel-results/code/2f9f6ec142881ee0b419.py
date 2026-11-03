import time
import random
import math

def _compute_score(U):
    U_set = frozenset(U)
    n = len(U_set)
    if n <= 1:
        return 1.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    sum_set = set()
    diff_set = set()
    for a in U_set:
        for b in U_set:
            sum_set.add(a + b)
            diff_set.add(a - b)
    
    sum_size = len(sum_set)
    diff_size = len(diff_set)
    
    if sum_size == 0:
        return 0.0
    
    ratio = diff_size / sum_size
    log_denom = math.log(2 * max_u + 1)
    if log_denom < 1e-10:
        return 1.0
    
    score = 1.0 + math.log(ratio) / log_denom
    return score


def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    random.seed(12345)
    
    best_score = -1e18
    best_set = None
    
    def update_best(s):
        nonlocal best_score, best_set
        if 0 not in s:
            s = list(s) + [0]
        s = sorted(set(s))
        sc = _compute_score(s)
        if sc > best_score:
            best_score = sc
            best_set = list(s)
        return sc
    
    # Phase 1: Try diverse initial configurations
    # Geometric progressions
    for base in range(2, 25):
        for n in range(2, min(15, int(math.log(1000000)/math.log(base)) + 2)):
            s = [0]
            v = 1
            for i in range(1, n):
                v *= base
                if v > 1000000:
                    break
                s.append(v)
            update_best(s)
    
    # Quadratic growth
    for n in range(3, 20):
        s = [i*i for i in range(n)]
        update_best(s)
    
    # Exponential
    for n in range(3, 15):
        s = [2**i for i in range(n)]
        update_best(s)
    
    # Blocks: {0,1, ..., b-1, b, 2b, 2b+1, ..., 3b-1, ...}
    for b in range(2, 10):
        for k in range(2, 8):
            s = []
            for i in range(k):
                for j in range(b):
                    s.append(i*b + j)
            update_best(s)
    
    # Random sets
    for trial in range(200):
        n = random.randint(3, 15)
        max_val = random.choice([50, 100, 500, 1000, 5000])
        s = {0}
        for _ in range(n-1):
            s.add(random.randint(1, max_val))
        update_best(list(s))
    
    # Phase 2: Local search from best
    current = list(best_set) if best_set else [0, 1, 3]
    current_score = best_score
    temperature = 0.5
    
    def try_move(s):
        s = list(s)
        max_e = max(s) if s else 0
        r = random.random()
        if r < 0.35:
            # Add
            if random.random() < 0.3:
                ne = random.randint(1, max_e + max_e//2 + 5)
            else:
                ne = random.randint(1, max_e + 10)
            if ne not in s:
                s.append(ne)
                s.sort()
                return s
        elif r < 0.6:
            # Remove
            if len(s) > 2:
                idx = random.randint(1, len(s)-1)
                s.pop(idx)
                return s
        else:
            # Swap
            if len(s) > 2:
                idx = random.randint(1, len(s)-1)
                nv = random.randint(1, max_e + 15)
                if nv not in s:
                    s[idx] = nv
                    s.sort()
                    return s
        return None
    
    stagnation = 0
    while time.time() < deadline - 10.0:
        new_s = try_move(current)
        if new_s is None:
            continue
        
        new_score = _compute_score(new_s)
        delta = new_score - current_score
        
        if new_score >= current_score or random.random() < math.exp(delta / max(temperature, 0.01)):
            current = new_s
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = list(new_s)
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1
        
        temperature *= 0.9995
        if temperature < 0.01:
            temperature = 0.1
        
        # If stuck, kick
        if stagnation > 200:
            # Perturb
            for _ in range(5):
                perturbed = try_move(current)
                if perturbed:
                    current = perturbed
                    current_score = _compute_score(current)
            stagnation = 0
            temperature = 0.5
    
    if best_set is None:
        return [0, 1, 3]
    if 0 not in best_set:
        best_set.append(0)
    return sorted(best_set)

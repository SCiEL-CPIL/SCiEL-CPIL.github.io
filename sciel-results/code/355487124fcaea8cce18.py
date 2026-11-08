import random
import math

def compute_score(U):
    U = sorted(set(U))
    if not U or 0 not in U or max(U) == 0:
        return 0
    max_u = max(U)
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    n_sum = len(sumset)
    n_diff = len(diffset)
    if n_sum == 0:
        return 0
    ratio = n_diff / n_sum
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0
    return 1 + math.log(ratio) / denom

def construct():
    best = [0, 1, 4, 7]
    best_score = compute_score(best)
    
    random.seed(123)
    
    # Local search
    for _ in range(2000):
        U = best[:]
        op = random.choice(['add', 'remove', 'modify'])
        if op == 'add' and len(U) < 8:
            new_val = random.randint(1, max(U) + 5)
            if new_val not in U:
                U.append(new_val)
                U = sorted(set(U))
        elif op == 'remove' and len(U) > 2:
            idx = random.randint(1, len(U) - 1)  # don't remove 0
            U.pop(idx)
            U = sorted(set(U))
        elif op == 'modify' and len(U) > 1:
            idx = random.randint(1, len(U) - 1)
            U[idx] = random.randint(1, max(U) + 5)
            U = sorted(set(U))
            if 0 not in U:
                continue
        
        score = compute_score(U)
        if score > best_score + 1e-9:
            best_score = score
            best = U[:]
    
    return best

import numpy as np
from itertools import combinations

def construct():
    import random
    import math
    
    def score(U):
        U = sorted(set(U))
        if 0 not in U:
            return -1
        max_u = max(U)
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        ratio = len(diffs) / len(sums)
        if ratio <= 1:
            return 1.0
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    
    # Start with powers of 3
    best_U = [0, 1, 3, 9]
    best_score = score(best_U)
    
    # Try local search
    random.seed(42)
    for _ in range(5000):
        U = list(best_U)
        # Random mutation
        op = random.choice(['add', 'remove', 'replace'])
        if op == 'add' and len(U) < 100:
            max_u = max(U) if U else 0
            new_el = random.randint(1, max_u * 2 + 10)
            U.append(new_el)
        elif op == 'remove' and len(U) > 1:
            idx = random.randint(1, len(U) - 1)  # keep 0
            U.pop(idx)
        elif op == 'replace' and len(U) > 1:
            idx = random.randint(1, len(U) - 1)
            max_u = max(U) if U else 0
            U[idx] = random.randint(1, max_u * 2 + 10)
        
        U = sorted(set(U))
        if 0 not in U:
            U.append(0)
            U = sorted(U)
        
        s = score(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    return best_U

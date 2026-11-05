import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 88.0
    
    n = 256
    
    def can_add(x, S):
        # S is a list of ints.
        # 1. x is a
        for b in S:
            if b <= x:
                continue
            for c in S:
                if c <= b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # 2. x is b
        for a in S:
            if a >= x:
                continue
            for c in S:
                if c <= x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # 3. x is c
        for a in S:
            if a >= x:
                continue
            for b in S:
                if b <= a or b >= x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True

    def can_add_two(v1, v2, S0):
        if not can_add(v1, S0):
            return False
        if not can_add(v2, S0):
            return False
            
        if v1 < v2:
            for c in S0:
                if c >= v2:
                    if (v2 + c) % v1 == 0:
                        return False
            for b in S0:
                if v1 < b <= v2:
                    if (b + v2) % v1 == 0:
                        return False
            for a in S0:
                if a < v1:
                    if (v1 + v2) % a == 0:
                        return False
        else:
            for c in S0:
                if c >= v1:
                    if (v1 + c) % v2 == 0:
                        return False
            for b in S0:
                if v2 < b <= v1:
                    if (b + v1) % v2 == 0:
                        return False
            for a in S0:
                if a < v2:
                    if (v2 + v1) % a == 0:
                        return False
        return True

    # Initial Greedy
    S = []
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
        if can_add(x, S):
            S.append(x)
    
    best_set = S[:]
    best_size = len(S)
    S_set = set(S)
    
    no_improvement_count = 0
    max_no_improvement = 200
    
    while time.time() < deadline:
        improved = False
        
        missing = [x for x in range(1, n+1) if x not in S_set]
        
        # 1. Try to add single element
        if missing:
            random.shuffle(missing)
            for x in missing:
                if time.time() > deadline:
                    break
                if can_add(x, S):
                    S.append(x)
                    S_set.add(x)
                    best_size += 1
                    best_set = S[:]
                    improved = True
                    break
        
        if improved:
            no_improvement_count = 0
            continue
        
        # 2. Try 1-2 swaps
        if len(missing) >= 2 and S:
            u = random.choice(S)
            # Try a few random pairs
            for _ in range(20):
                if time.time() > deadline:
                    break
                v1, v2 = random.sample(missing, 2)
                S_minus_u = [x for x in S if x != u]
                if can_add_two(v1, v2, S_minus_u):
                    S.remove(u)
                    S_set.remove(u)
                    S.append(v1)
                    S_set.add(v1)
                    S.append(v2)
                    S_set.add(v2)
                    best_size += 1
                    best_set = S[:]
                    improved = True
                    break
        
        if improved:
            no_improvement_count = 0
            continue
        
        # 3. Try 1-1 swaps
        if missing and S:
            u = random.choice(S)
            for _ in range(10):
                if time.time() > deadline:
                    break
                v = random.choice(missing)
                S_minus_u = [x for x in S if x != u]
                if can_add(v, S_minus_u):
                    S.remove(u)
                    S_set.remove(u)
                    S.append(v)
                    S_set.add(v)
                    improved = True
                    break
        
        if improved:
            no_improvement_count = 0
            continue
        
        no_improvement_count += 1
        if no_improvement_count > max_no_improvement:
            break
            
    return best_set
# EVOLVE-BLOCK-END

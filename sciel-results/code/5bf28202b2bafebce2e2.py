import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 256
    candidates = list(range(2, N + 1))
    
    def is_valid_addition(S, x):
        # S is a list
        n = len(S)
        for i in range(n):
            b = S[i]
            for j in range(n):
                if i == j: continue
                c = S[j]
                if (b * c) % x == 0:
                    return False
        return True

    def is_valid_set(S):
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(n):
                if i == j: continue
                b = S[j]
                for k in range(n):
                    if i == k or j == k: continue
                    c = S[k]
                    if (b * c) % a == 0:
                        return False
        return True

    def greedy(order):
        S = []
        for x in order:
            if is_valid_addition(S, x):
                S.append(x)
        return S

    def local_search(S):
        S_set = set(S)
        C = [x for x in candidates if x not in S_set]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = list(S)
            n = len(S_list)
            
            # Try 1-for-2 swap
            for i in range(n):
                if time.time() > deadline: break
                u = S_list[i]
                S_minus_u = [x for x in S_list if x != u]
                
                # Find candidates x that are compatible with S_minus_u
                # We check x against S_minus_u
                valid_xs = []
                for x in C:
                    if is_valid_addition(S_minus_u, x):
                        valid_xs.append(x)
                        
                # Try pairs from valid_xs
                found = False
                for idx_x in range(len(valid_xs)):
                    if found: break
                    x = valid_xs[idx_x]
                    for idx_v in range(idx_x + 1, len(valid_xs)):
                        if time.time() > deadline: break
                        v = valid_xs[idx_v]
                        
                        # Check compatibility of x, v with S_minus_u
                        # We know x is compat with S_minus_u
                        # We know v is compat with S_minus_u
                        # We need to check triples (x, v, a) for a in S_minus_u
                        ok = True
                        for a in S_minus_u:
                            if (v * a) % x == 0: ok = False; break
                            if (x * a) % v == 0: ok = False; break
                            if (x * v) % a == 0: ok = False; break
                            
                        if ok:
                            S = S_minus_u + [x, v]
                            S_set = set(S)
                            C = [c for c in candidates if c not in S_set]
                            improved = True
                            found = True
                            break
                            
            # Try simple addition
            if not improved and time.time() < deadline:
                for x in C:
                    if is_valid_addition(S, x):
                        S.append(x)
                        S_set = set(S)
                        C = [c for c in candidates if c not in S_set]
                        improved = True
                        break
                        
        return S

    best_set = []
    
    # Heuristics
    orders = [
        sorted(candidates, reverse=True),
        sorted(candidates),
    ]
    
    # Random restarts
    for _ in range(50):
        orders.append(sorted(candidates, key=lambda x: random.random()))
        
    for order in orders:
        if time.time() > deadline - 10:
            break
        S = greedy(order)
        S = local_search(S)
        if len(S) > len(best_set):
            best_set = S
            
    if not is_valid_set(best_set):
        # Fallback to primes
        best_set = [i for i in range(2, N+1) if all(i % j != 0 for j in range(2, int(i**0.5)+1))]

    return [int(x) for x in best_set]

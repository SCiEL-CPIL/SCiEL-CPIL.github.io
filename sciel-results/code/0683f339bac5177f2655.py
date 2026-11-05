import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    N = 180
    
    forbidden = []
    for b in range(1, N+1):
        for c in range(b+1, N+1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    forbidden.append((a, b, c))
    
    elem_triples = {i: [] for i in range(1, N+1)}
    for idx, t in enumerate(forbidden):
        for x in t:
            elem_triples[x].append(idx)
    
    def check_x(S, x):
        for idx in elem_triples[x]:
            a, b, c = forbidden[idx]
            if a in S and b in S and c in S:
                return False
        return True
    
    def full_check(S):
        for idx, t in enumerate(forbidden):
            a, b, c = t
            if a in S and b in S and c in S:
                return False
        return True
    
    best_S = set(range(91, 181))
    best_size = 90
    
    random.seed(42)
    
    for trial in range(5000):
        if time.time() - start_time > DEADLINE - 3:
            break
        
        # Strategy 1: start with [91..180], add small
        S = set(range(91, 181))
        cands = list(range(1, 91))
        random.shuffle(cands)
        for x in cands:
            if check_x(S, x):
                S.add(x)
        
        # Local search
        for _ in range(30):
            if time.time() - start_time > DEADLINE - 3:
                break
            improved = False
            excluded = [x for x in range(1, 91) if x not in S]
            random.shuffle(excluded)
            for x in excluded:
                if check_x(S, x):
                    S.add(x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    break
                for idx in elem_triples[x]:
                    a, b, c = forbidden[idx]
                    if a in S and b in S and c in S:
                        for rem in (a, b, c):
                            S.remove(rem)
                            S.add(x)
                            if check_x(S, x):
                                improved = True
                                if len(S) > best_size:
                                    best_size = len(S)
                                    best_S = set(S)
                                break
                            else:
                                S.remove(x)
                                S.add(rem)
                        if improved:
                            break
                if improved:
                    break
            if not improved:
                break
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Final greedy
    if time.time() - start_time < DEADLINE - 1:
        for x in range(1, 91):
            if x not in best_S and check_x(best_S, x):
                best_S.add(x)
                best_size = len(best_S)
    
    return sorted(best_S)

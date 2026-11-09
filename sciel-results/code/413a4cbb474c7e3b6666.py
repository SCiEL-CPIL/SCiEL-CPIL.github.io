import time
import random

def _can_add(S, x):
    # S is a set of ints
    # x is an int not in S
    # Returns True if S U {x} is valid
    n = 128
    S_list = list(S)
    n_s = len(S_list)
    
    # Case 1: x is one of a, b, c. WLOG a=x.
    # Need b, c in S, b != c, such that:
    # x+b in S, x+c in S, (b+c in S or b+c == x)
    for i in range(n_s):
        b = S_list[i]
        if b > n - x:
            continue
        if x + b in S:
            for j in range(i + 1, n_s):
                c = S_list[j]
                if c > n - x:
                    continue
                if x + c in S:
                    if b + c in S or b + c == x:
                        return False
    
    # Case 2: x is one of the sums. WLOG a+b=x.
    # Need a, b in S, a != b, a+b == x.
    # Then need c in S \ {a, b} such that a+c in S and b+c in S.
    for i in range(n_s):
        a = S_list[i]
        b = x - a
        if b <= 0 or b > n:
            continue
        if b in S and b != a:
            for k in range(n_s):
                c = S_list[k]
                if c == a or c == b:
                    continue
                if c > n - a or c > n - b:
                    continue
                if a + c in S and b + c in S:
                    return False
                    
    return True

def construct():
    n = 128
    start_time = time.time()
    deadline = 85.0
    
    # Initial Seed
    S0 = set(range(n // 2 + 1, n + 1))
    
    best_S = set(S0)
    best_size = len(S0)
    
    # Greedy Construction with Random Orders
    candidates = list(range(1, n // 2 + 1))
    
    for _ in range(100):
        if time.time() > deadline:
            break
            
        random.shuffle(candidates)
        S = set(S0)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if x not in S:
                if _can_add(S, x):
                    S.add(x)
                    
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
    # Local Search on best_S
    S = set(best_S)
    
    while time.time() < deadline:
        if not S:
            break
            
        y = random.choice(list(S))
        S.remove(y)
        
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            if x not in S:
                if _can_add(S, x):
                    S.add(x)
                    
        new_size = len(S)
        if new_size > best_size:
            best_size = new_size
            best_S = set(S)
        else:
            S = set(best_S)
            
    return list(best_S)

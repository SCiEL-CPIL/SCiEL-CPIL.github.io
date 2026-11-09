import time
import random

N = 256

def _validate(S):
    """Return True iff S satisfies: for all a<b<=c in S, a does not divide b+c."""
    L = sorted(S)
    m = len(L)
    for i in range(m):
        a = L[i]
        for j in range(i + 1, m):
            b = L[j]
            for k in range(j, m):
                c = L[k]
                if (b + c) % a == 0:
                    return False
    return True

def _can_add(x, S):
    """
    Check if x can be added to S while maintaining validity.
    S is a set of integers.
    x is an integer not in S.
    """
    S_list = list(S)
    
    # Case 1: x is a
    for b in S_list:
        if b <= x:
            continue
        for c in S_list:
            if c < b:
                continue
            if (b + c) % x == 0:
                return False
                
    # Case 2: x is b
    for a in S_list:
        if a >= x:
            continue
        for c in S_list:
            if c < x:
                continue
            if (x + c) % a == 0:
                return False
                
    # Case 3: x is c
    for b in S_list:
        if b > x:
            continue
        for a in S_list:
            if a >= b:
                continue
            if (b + x) % a == 0:
                return False
                
    return True

def _greedy_descending():
    n = N
    S = set()
    for x in range(n, 0, -1):
        if _can_add(x, S):
            S.add(x)
    return S

def _local_search_sa(S_init, deadline):
    """
    Simulated Annealing style local search.
    """
    best_S = set(S_init)
    best_size = len(best_S)
    current_S = set(S_init)
    current_size = len(current_S)
    
    rng = random.Random(42)
    
    # We don't really need T if we use the simple probability heuristic
    # which depends on the size difference.
    
    while time.time() < deadline:
        # Perturb
        k = rng.choice([1, 2, 3])
        if len(current_S) < k:
            k = len(current_S)
        if k == 0:
            break
            
        to_remove = rng.sample(list(current_S), k)
        S_new = current_S - set(to_remove)
        
        # Refill greedily (descending)
        for x in range(N, 0, -1):
            if x in S_new:
                continue
            if _can_add(x, S_new):
                S_new.add(x)
                
        new_size = len(S_new)
        
        if new_size > current_size:
            current_S = S_new
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_S = set(S_new)
        else:
            # Accept worse solution to escape local minima
            delta = current_size - new_size
            # Probability decreases with delta
            prob = 1.0 / (1.0 + delta)
            if rng.random() < prob:
                current_S = S_new
                current_size = new_size
                
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Generate initial candidate
    S_init = _greedy_descending()
    
    # Run local search
    best_set = _local_search_sa(S_init, deadline)
    
    return sorted(best_set)

if __name__ == "__main__":
    _s = construct()
    assert _validate(set(_s)), "constructed set is invalid"
    print("size", len(_s), "first", _s[:5], "last", _s[-5:])

import time
import random

def _score(S):
    Sset = set(S)
    dset = set()
    for a in S:
        for b in S:
            if b > a:
                d = b - a
                c = a + 2*d
                if c in Sset:
                    dset.add(d)
    return len(dset)

def construct():
    deadline = time.time() + 140
    n = 32
    max_val = 512
    random.seed(42)
    
    # Initial: random
    S = random.sample(range(1, max_val+1), n)
    best_S = list(S)
    best_score = _score(S)
    
    # Also try dense start
    S2 = list(range(1, n+1))
    sc2 = _score(S2)
    if sc2 > best_score:
        best_score = sc2
        best_S = list(S2)
        S = list(S2)
    
    Sset = set(S)
    current_score = best_score
    T = 1.0
    
    while time.time() < deadline - 5:
        i = random.randint(0, n-1)
        old = S[i]
        new = random.randint(1, max_val)
        if new in Sset:
            continue
        
        S[i] = new
        Sset.remove(old)
        Sset.add(new)
        
        ns = _score(S)
        delta = ns - current_score
        if delta > 0 or random.random() < pow(2.71828, delta / T):
            current_score = ns
            if ns > best_score:
                best_score = ns
                best_S = list(S)
        else:
            S[i] = old
            Sset.remove(new)
            Sset.add(old)
        
        T *= 0.999995
        if T < 0.01:
            T = 0.01
    
    return best_S

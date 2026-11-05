import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MAX_M = 128
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)
    
    # Precompute gcd for all pairs
    gcd_tab = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g
    
    def try_greedy(order):
        selected = []
        for idx in order:
            m = moduli[idx]
            for r in range(m):
                ok = True
                for (idx2, r2, m2) in selected:
                    if (r - r2) % gcd_tab[idx][idx2] == 0:
                        ok = False
                        break
                if ok:
                    selected.append((idx, r, m))
                    break
        return selected
    
    best = []
    
    # Compute compatibility count (number of moduli with gcd > 1)
    compat = [sum(1 for j in range(n) if j != i and gcd_tab[i][j] > 1) for i in range(n)]
    
    # Multiple deterministic orderings
    orderings = [
        sorted(range(n), key=lambda i: compat[i]),
        sorted(range(n), key=lambda i: -compat[i]),
        sorted(range(n), key=lambda i: moduli[i]),
        sorted(range(n), key=lambda i: -moduli[i]),
        list(range(n)),
        list(reversed(range(n))),
    ]
    
    for order in orderings:
        res = try_greedy(order)
        if len(res) > len(best):
            best = res
    
    # Random orderings
    random.seed(42)
    for _ in range(300):
        if time.time() > deadline - 20:
            break
        order = list(range(n))
        random.shuffle(order)
        res = try_greedy(order)
        if len(res) > len(best):
            best = res
    
    # Local search: try to add more moduli
    if time.time() > deadline - 10:
        return [[r, m] for (_, r, m) in best]
    
    selected_idx = set(idx for (idx, _, _) in best)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for idx in range(n):
            if idx in selected_idx:
                continue
            m = moduli[idx]
            for r in range(m):
                ok = True
                for (idx2, r2, m2) in best:
                    if (r - r2) % gcd_tab[idx][idx2] == 0:
                        ok = False
                        break
                if ok:
                    best.append((idx, r, m))
                    selected_idx.add(idx)
                    improved = True
                    break
            if improved:
                break
    
    return [[r, m] for (_, r, m) in best]

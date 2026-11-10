import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Fallback: return a known valid small n if search fails
    # We assume n=34 is possible. We need a valid coloring for n=34.
    # Since we don't have a hardcoded one, we rely on the search.
    # But to be safe, we should try to find n=35 first, then n=34.
    
    def make_structure(n):
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        num_triples = len(triples)

        k4s = list(combinations(range(n), 4))
        k4_triples = np.zeros((len(k4s), 4), dtype=np.int32)
        for i, k4 in enumerate(k4s):
            for j in range(4):
                t = tuple(sorted(k4[:j] + k4[j+1:]))
                k4_triples[i, j] = triple_to_idx[t]

        k5s = list(combinations(range(n), 5))
        k5_triples = np.zeros((len(k5s), 10), dtype=np.int32)
        for i, k5 in enumerate(k5s):
            count = 0
            for combo in combinations(range(5), 3):
                t = tuple(sorted([k5[c] for c in combo]))
                k5_triples[i, count] = triple_to_idx[t]
                count += 1

        return triples, triple_to_idx, num_triples, k4_triples, k5_triples

    def count_violations(colors, k4_triples, k5_triples):
        red_k4 = np.all(colors[k4_triples] == 0, axis=1).sum()
        blue_k5 = np.all(colors[k5_triples] == 1, axis=1).sum()
        return int(red_k4 + blue_k5)

    def search_algebraic(n, time_budget):
        # Parameters: T[d1, d2] where d1 = (b-a)%n, d2 = (c-b)%n
        # Note: (a,b,c) is sorted. d1>0, d2>0, d1+d2 < n.
        # We can just use a full n x n table and ignore invalid entries.
        T = np.random.randint(0, 2, (n, n), dtype=np.int8)
        
        triples, triple_to_idx, num_triples, k4_triples, k5_triples = make_structure(n)
        
        # Precompute d1, d2 for each triple
        d1s = np.zeros(num_triples, dtype=np.int32)
        d2s = np.zeros(num_triples, dtype=np.int32)
        for i, (a, b, c) in enumerate(triples):
            d1s[i] = (b - a) % n
            d2s[i] = (c - b) % n
            
        # Vectorized color generation from T
        colors = T[d1s, d2s]
        
        current_v = count_violations(colors, k4_triples, k5_triples)
        best_v = current_v
        best_T = T.copy()
        
        rng = np.random.default_rng(42)
        temp = 1.0
        min_temp = 0.01
        
        while time.time() < start_time + time_budget:
            # Flip a random entry in T
            i = rng.integers(0, n)
            j = rng.integers(0, n)
            old_val = T[i, j]
            new_val = 1 - old_val
            T[i, j] = new_val
            
            # Update colors
            # Find indices where d1s==i and d2s==j
            mask = (d1s == i) & (d2s == j)
            if np.any(mask):
                colors[mask] = new_val
            
            new_v = count_violations(colors, k4_triples, k5_triples)
            
            if new_v < best_v:
                best_v = new_v
                best_T = T.copy()
                temp *= 0.95
            elif new_v <= current_v:
                current_v = new_v
            else:
                delta = new_v - current_v
                if rng.random() < np.exp(-delta / max(temp, min_temp)):
                    current_v = new_v
                    temp *= 0.99
                else:
                    T[i, j] = old_val
                    colors[mask] = old_val
                    temp *= 0.99
                    
            if best_v == 0:
                break
                
        if best_v == 0:
            return best_T
        return None

    # Try n=35
    n_target = 35
    time_budget_35 = 60.0
    T_35 = search_algebraic(n_target, time_budget_35)
    
    if T_35 is not None:
        triples, triple_to_idx, num_triples, k4_triples, k5_triples = make_structure(n_target)
        d1s = np.zeros(num_triples, dtype=np.int32)
        d2s = np.zeros(num_triples, dtype=np.int32)
        for i, (a, b, c) in enumerate(triples):
            d1s[i] = (b - a) % n_target
            d2s[i] = (c - b) % n_target
        colors = T_35[d1s, d2s]
        return {'n': n_target, 'colors': colors.tolist()}
        
    # Try n=34
    n_target = 34
    time_budget_34 = 60.0
    T_34 = search_algebraic(n_target, time_budget_34)
    
    if T_34 is not None:
        triples, triple_to_idx, num_triples, k4_triples, k5_triples = make_structure(n_target)
        d1s = np.zeros(num_triples, dtype=np.int32)
        d2s = np.zeros(num_triples, dtype=np.int32)
        for i, (a, b, c) in enumerate(triples):
            d1s[i] = (b - a) % n_target
            d2s[i] = (c - b) % n_target
        colors = T_34[d1s, d2s]
        return {'n': n_target, 'colors': colors.tolist()}
        
    # Fallback to random search for n=34 if algebraic fails
    # This is less likely to succeed but better than nothing
    n_target = 34
    triples, triple_to_idx, num_triples, k4_triples, k5_triples = make_structure(n_target)
    rng = np.random.default_rng(42)
    colors = rng.integers(0, 2, size=num_triples).astype(np.int8)
    current_v = count_violations(colors, k4_triples, k5_triples)
    best_v = current_v
    best_cols = colors.copy()
    
    while time.time() < deadline - 5.0:
        idx = rng.integers(0, num_triples)
        old_color = colors[idx]
        new_color = 1 - old_color
        colors[idx] = new_color
        new_v = count_violations(colors, k4_triples, k5_triples)
        
        if new_v < best_v:
            best_v = new_v
            best_cols = colors.copy()
        elif new_v > current_v:
            colors[idx] = old_color
            # simple hill climb, no SA here to save time
        else:
            current_v = new_v
            
        if best_v == 0:
            break
            
    return {'n': n_target, 'colors': best_cols.tolist()}

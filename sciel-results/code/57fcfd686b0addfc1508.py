import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # Step 1: Generate Type D vectors (weight 4, norm 2)
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)
    sup_sets = [set(s) for s in supports]
    
    # Build incompatibility graph for supports
    incompat = [0] * N_sup
    for i in range(N_sup):
        for j in range(i + 1, N_sup):
            if len(sup_sets[i] & sup_sets[j]) >= 3:
                incompat[i] |= (1 << j)
                incompat[j] |= (1 << i)
    
    # Greedy max independent set on supports
    def greedy_supports():
        avail = set(range(N_sup))
        result = []
        while avail:
            min_deg = float('inf')
            u = None
            for v in avail:
                deg = sum(1 for w in avail if w != v and (incompat[v] >> w) & 1)
                if deg < min_deg:
                    min_deg = deg
                    u = v
            result.append(u)
            to_remove = {u}
            for v in list(avail):
                if (incompat[u] >> v) & 1:
                    to_remove.add(v)
            avail -= to_remove
        return result
    
    selected_supports = greedy_supports()
    
    # Generate all Type D vectors for selected supports
    type_d = []
    for i in selected_supports:
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            type_d.append(v)
    type_d = np.array(type_d, dtype=np.int8)
    
    # Step 2: Add Type A vectors (norm 2)
    type_a = []
    for i in range(D):
        for s in [1, -1]:
            v = np.zeros(D, dtype=np.int8)
            v[i] = 2 * s
            type_a.append(v)
    type_a = np.array(type_a, dtype=np.int8)
    
    # Step 3: Add Type M1 vectors (two +/-2, norm sqrt(8))
    type_m1 = []
    for i, j in combinations(range(D), 2):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = np.zeros(D, dtype=np.int8)
                v[i] = 2 * s1
                v[j] = 2 * s2
                type_m1.append(v)
    type_m1 = np.array(type_m1, dtype=np.int8)
    
    # Combine all candidates
    all_candidates = np.vstack([type_d, type_a, type_m1])
    N = len(all_candidates)
    norms = np.sqrt(np.sum(all_candidates**2, axis=1))
    max_norm = np.max(norms)
    
    # Build conflict matrix
    dots = all_candidates @ all_candidates.T
    thresh = (np.outer(norms, norms) - max_norm**2) / 2.0
    conflict = dots > thresh
    np.fill_diagonal(conflict, False)
    
    # Greedy independent set
    def greedy_is(conflict, n):
        selected = np.zeros(n, dtype=bool)
        remaining = np.ones(n, dtype=bool)
        while remaining.any():
            degs = conflict[remaining] @ remaining
            min_idx = np.argmin(degs)
            u = np.where(remaining)[0][min_idx]
            selected[u] = True
            to_remove = conflict[u].copy()
            to_remove[u] = True
            remaining[to_remove] = False
        return np.where(selected)[0]
    
    initial = greedy_is(conflict, N)
    
    # Local search with precomputed neighbors
    def local_search(conflict, n, initial):
        selected = set(initial.tolist())
        # Precompute neighbors for fast lookup
        neighbors = [set(np.where(conflict[i])[0].tolist()) for i in range(n)]
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for v in range(n):
                if v in selected:
                    continue
                # Check if v conflicts with any selected
                if not (neighbors[v] & selected):
                    selected.add(v)
                    improved = True
        return list(selected)
    
    best = local_search(conflict, N, initial)
    
    # Try a few restarts
    rng = np.random.RandomState(42)
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        order = np.arange(N)
        rng.shuffle(order)
        sel_r = set()
        for v in order:
            if not (neighbors[v] & sel_r):
                sel_r.add(v)
        if len(sel_r) > len(best):
            best = list(sel_r)
    
    result = all_candidates[best]
    return result.astype(np.int8)

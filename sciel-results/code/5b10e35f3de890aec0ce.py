import numpy as np
import random
import itertools
import time

def construct():
    D = 11
    # Generate candidate vectors: all non-zero vectors in {-1, 0, 1}^11 with weight 3
    # This gives a large pool of candidates with small norms, suitable for packing.
    candidates = []
    for support in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for k, s in enumerate(support):
                v[s] = signs[k]
            candidates.append(tuple(v))
    
    n_cand = len(candidates)
    
    # Precompute squared distances and build conflict graph
    # Conflict if ||v_i - v_j||^2 < max_norm^2
    # Here max_norm^2 = 3 (since all are weight 3)
    # So conflict if dist^2 < 3.
    # Since dist^2 is integer, conflict if dist^2 <= 2.
    # dist^2 = 0 (same) or dist^2 = 2.
    # dist^2 = 2 implies dot product = 2 (since 3+3-2dot=2 => dot=2).
    
    adj = [set() for _ in range(n_cand)]
    
    # To speed up, we can iterate and check dot products.
    # n_cand = 1320. Pairs = ~870k. This is fast enough in Python if optimized?
    # 870k iterations * simple math. Should take < 1 second.
    
    for i in range(n_cand):
        v_i = candidates[i]
        for j in range(i + 1, n_cand):
            v_j = candidates[j]
            # Calculate squared distance
            # dist2 = sum((vi - vj)^2)
            # Or use dot product: dist2 = 3 + 3 - 2 * dot
            dot = 0
            for k in range(D):
                dot += v_i[k] * v_j[k]
            dist2 = 6 - 2 * dot
            if dist2 < 3: # dist2 == 2
                adj[i].add(j)
                adj[j].add(i)
                
    # 1. Greedy Initialization
    def greedy_is():
        remaining = set(range(n_cand))
        current_set = set()
        while remaining:
            # Pick vertex with min degree in remaining
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = len(adj[v].intersection(remaining))
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0: break
            current_set.add(best_v)
            to_remove = {best_v}.union(adj[best_v].intersection(remaining))
            remaining.difference_update(to_remove)
        return current_set

    best_set = greedy_is()
    best_size = len(best_set)
    
    # 2. Simulated Annealing
    S = best_set.copy()
    current_size = len(S)
    
    start_time = time.time()
    time_limit = 100  # seconds, leaving buffer for overhead
    
    T = 1.0
    cooling = 0.9995
    
    # Optimization: Keep a list of indices not in S to pick from?
    # Or just random pick and check.
    
    while time.time() - start_time < time_limit:
        # Batch process for speed?
        # Let's do 1000 steps per loop check
        
        for _ in range(1000):
            v = random.randint(0, n_cand - 1)
            
            if v in S:
                continue
                
            N = adj[v].intersection(S)
            
            if len(N) == 0:
                S.add(v)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = S.copy()
            else:
                delta = 1 - len(N)
                if delta > 0:
                    S.difference_update(N)
                    S.add(v)
                    current_size += delta
                    if current_size > best_size:
                        best_size = current_size
                        best_set = S.copy()
                elif delta == 0:
                    if random.random() < 0.5:
                        S.difference_update(N)
                        S.add(v)
                else:
                    # delta < 0
                    p = np.exp(delta / T)
                    if random.random() < p:
                        S.difference_update(N)
                        S.add(v)
                        current_size += delta
                        
        T *= cooling
        if T < 0.01:
            break

    # Construct vectors from best_set
    vectors = [candidates[i] for i in best_set]
    
    return np.array(vectors, dtype=np.int64)

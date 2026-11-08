"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    # Build candidate pool: weight-4 vectors
    candidates = []
    for idx in itertools.combinations(range(D), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = [0] * D
            for i, s in zip(idx, signs):
                v[i] = s
            candidates.append(tuple(v))
    
    N = len(candidates)
    V = np.array(candidates, dtype=np.float64)
    
    # Build conflict graph: edge if dot > 2
    # Max norm squared is 4.
    # Condition: 8 - 2 dot >= 4 => dot <= 2.
    # Conflict if dot >= 3.
    
    conflicts = [set() for _ in range(N)]
    
    # Use numpy for speed
    chunk_size = 500
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        if time.time() > deadline - 40:
            break
        V_chunk = V[start:end]
        dots = V_chunk @ V.T
        for local_i in range(end - start):
            i = start + local_i
            mask = dots[local_i] > 2
            mask[i] = False
            for j in np.where(mask)[0]:
                j = int(j)
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    if time.time() > deadline - 10:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    best_set = []
    
    # Run multiple greedy passes with different orderings
    # 1. Standard degree ordering
    degrees = [len(c) for c in conflicts]
    order = sorted(range(N), key=lambda x: degrees[x])
    
    in_set = [False] * N
    current_set = []
    for idx in order:
        if in_set[idx]:
            continue
        can_add = True
        for nb in conflicts[idx]:
            if in_set[nb]:
                can_add = False
                break
        if can_add:
            in_set[idx] = True
            current_set.append(idx)
    
    if len(current_set) > len(best_set):
        best_set = current_set
    
    # 2. Randomized greedy (a few times)
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        in_set = [False] * N
        current_set = []
        
        # Random shuffle of indices, but prioritize low degree?
        # Just random order
        order = list(range(N))
        random.shuffle(order)
        
        for idx in order:
            if in_set[idx]:
                continue
            can_add = True
            for nb in conflicts[idx]:
                if in_set[nb]:
                    can_add = False
                    break
            if can_add:
                in_set[idx] = True
                current_set.append(idx)
        
        if len(current_set) > len(best_set):
            best_set = current_set

    if not best_set:
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    result_vectors = np.array([candidates[i] for i in best_set], dtype=np.int64)
    return result_vectors
# EVOLVE-BLOCK-END

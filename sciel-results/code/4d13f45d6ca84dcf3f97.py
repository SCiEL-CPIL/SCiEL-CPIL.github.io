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

    # Build candidate pool: weight-4 vectors (4 non-zero entries of ±1)
    # Total: C(11,4) * 2^4 = 330 * 16 = 5280
    candidates = []
    for idx in itertools.combinations(range(D), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = [0] * D
            for i, s in zip(idx, signs):
                v[i] = s
            candidates.append(tuple(v))
    
    N = len(candidates)  # 5280
    V = np.array(candidates, dtype=np.float64)
    
    # All weight-4 vectors have norm^2 = 4, so max_norm_sq = 4
    # Constraint: ||v_i - v_j||^2 >= 4
    # ||v_i - v_j||^2 = 8 - 2*v_i·v_j
    # So: 8 - 2*dot >= 4 => dot <= 2
    # Conflict if dot >= 3
    
    # Build conflict graph efficiently using chunked matrix multiply
    conflicts = [set() for _ in range(N)]
    
    chunk_size = 500
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        if time.time() > deadline - 40:
            break
        V_chunk = V[start:end]
        dots = V_chunk @ V.T  # shape (chunk_size, N)
        for local_i in range(end - start):
            i = start + local_i
            mask = dots[local_i] > 2
            mask[i] = False
            for j in np.where(mask)[0]:
                j = int(j)
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    if time.time() > deadline - 10:
        # Fallback: return weight-2 vectors (220 of them)
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    best_set = []
    
    # Strategy 1: Greedy by degree (ascending)
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
        best_set = current_set[:]
    
    # Strategy 2: Multiple randomized greedy restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        in_set = [False] * N
        current_set = []
        
        # Random order with slight bias toward low-degree nodes
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
            best_set = current_set[:]
    
    # Strategy 3: Local search / improvement on the best set found
    # Try to swap out a node for a better one to increase set size
    if time.time() < deadline - 15 and best_set:
        in_set = [False] * N
        for idx in best_set:
            in_set[idx] = True
        
        # Try removing each node and re-adding greedily to see if we can do better
        # This is expensive, so only do a limited number of trials
        set_size = len(best_set)
        improved = True
        max_improvement_rounds = 5
        
        for round_num in range(max_improvement_rounds):
            if time.time() > deadline - 15:
                break
            improved = False
            # Try removing a random node from the set and see if we can add more
            if set_size > 0:
                remove_idx = random.choice(best_set)
                in_set[remove_idx] = False
                best_set.remove(remove_idx)
                set_size -= 1
                
                # Try to add nodes that were previously blocked
                for idx in range(N):
                    if in_set[idx]:
                        continue
                    can_add = True
                    for nb in conflicts[idx]:
                        if in_set[nb]:
                            can_add = False
                            break
                    if can_add:
                        in_set[idx] = True
                        best_set.append(idx)
                        set_size += 1
                
                if set_size > len(best_set):
                    improved = True
        
        if not best_set:
            # Fallback
            vectors = []
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i], v[j] = si, sj
                        vectors.append(v)
            return np.array(vectors, dtype=np.int64)
    
    if not best_set:
        # Fallback to seed
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

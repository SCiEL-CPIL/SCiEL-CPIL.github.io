import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    D = 11

    # Type A: Norm 2 vectors (weight 1, value 2)
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
    vectors_A = np.array(vectors_A, dtype=np.int64)

    # Type D: Norm 2 vectors (weight 4, value 1/-1)
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)
    
    # Conflict graph for supports (intersection >= 3)
    support_masks = [0] * N_S
    for i, s in enumerate(supports):
        mask = 0
        for j in s:
            mask |= (1 << j)
        support_masks[i] = mask
        
    conflict = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in range(i + 1, N_S):
            if bin(support_masks[i] & support_masks[j]).count('1') >= 3:
                conflict[i, j] = True
                conflict[j, i] = True

    # Convert to integer adjacency for vectorized operations
    conflict_int = conflict.astype(np.int8)
    
    # Precompute degree
    degrees = conflict_int.sum(axis=1)
    
    # Greedy MIS with MRV heuristic
    def greedy_mrv(seed=0):
        rng = random.Random(seed)
        remaining = set(range(N_S))
        in_set = set()
        
        while remaining:
            # Score each remaining node by MRV: number of compatible unselected supports it preserves
            # A node v is compatible with remaining node u if not conflict[v,u]
            # We want to pick the node that "preserves" the most remaining options
            # Score(v) = number of u in remaining such that not conflict[v,u] and u != v
            # Equivalently, score(v) = len(remaining) - 1 - (number of conflicts with remaining)
            
            # For efficiency, compute conflicts for each remaining node
            best_node = None
            best_score = -1
            best_tie = -1
            
            # Sample a subset if remaining is large for speed
            if len(remaining) > 50:
                sample = random.Random(seed + len(remaining)).sample(list(remaining), 50)
            else:
                sample = list(remaining)
            
            for v in sample:
                # Count conflicts with remaining
                conflicts_with_rem = 0
                for u in remaining:
                    if u != v and conflict[v, u]:
                        conflicts_with_rem += 1
                score = len(remaining) - 1 - conflicts_with_rem
                tie = rng.random()
                if score > best_score or (score == best_score and tie > best_tie):
                    best_score = score
                    best_tie = tie
                    best_node = v
            
            if best_node is None:
                break
                
            in_set.add(best_node)
            remaining.discard(best_node)
            # Remove all neighbors of best_node from remaining
            for u in list(remaining):
                if conflict[best_node, u]:
                    remaining.discard(u)
        
        return in_set, len(in_set)

    # Multi-start greedy MRV
    best_support_set = set()
    best_size = 0
    rng = np.random.RandomState(42)
    
    # Run many restarts with different seeds
    n_restarts = 200
    for r in range(n_restarts):
        if time.time() > deadline - 30:
            break
        support_set, size = greedy_mrv(seed=42 + r)
        if size > best_size:
            best_size = size
            best_support_set = support_set

    # Construct Vectors from Best Supports
    vectors_D_core = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors_D_core.append(v)
    vectors_D_core = np.array(vectors_D_core, dtype=np.int64)
    
    current_vectors = np.vstack([vectors_A, vectors_D_core])
    current_set_tuples = set(map(tuple, current_vectors))

    # Greedy Fill: Add remaining Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            all_vectors_D.append(v)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    
    # Shuffle for randomness
    indices = np.arange(len(all_vectors_D))
    rng.shuffle(indices)
    
    for i in indices:
        if time.time() > deadline - 10:
            break
        v = all_vectors_D[i]
        if tuple(v) in current_set_tuples:
            continue
        # Check compatibility: dot product <= 2
        dots = current_vectors @ v
        if np.all(dots <= 2):
            current_vectors = np.vstack([current_vectors, v.reshape(1, -1)])
            current_set_tuples.add(tuple(v))

    # Greedy Fill: Add Type B (Weight 2) vectors
    vectors_B = []
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors_B.append(v)
    vectors_B = np.array(vectors_B, dtype=np.int64)
    
    indices_B = np.arange(len(vectors_B))
    rng.shuffle(indices_B)
    
    for i in indices_B:
        if time.time() > deadline - 5:
            break
        v = vectors_B[i]
        if tuple(v) in current_set_tuples:
            continue
        dots = current_vectors @ v
        if np.all(dots <= 1):
            current_vectors = np.vstack([current_vectors, v.reshape(1, -1)])
            current_set_tuples.add(tuple(v))

    return current_vectors

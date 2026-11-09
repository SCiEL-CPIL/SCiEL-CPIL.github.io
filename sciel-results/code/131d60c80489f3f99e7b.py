import numpy as np
import itertools
import time

D = 11

def generate_candidates(Q):
    candidates = []
    if Q == 1:
        for i in range(D):
            for s in (1, -1):
                v = [0]*D
                v[i] = s
                candidates.append(v)
    elif Q == 2:
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0]*D
                    v[i], v[j] = si, sj
                    candidates.append(v)
    elif Q == 3:
        for i, j, k in itertools.combinations(range(D), 3):
            for si in (1, -1):
                for sj in (1, -1):
                    for sk in (1, -1):
                        v = [0]*D
                        v[i], v[j], v[k] = si, sj, sk
                        candidates.append(v)
    elif Q == 4:
        # 4 non-zeros
        for i, j, k, l in itertools.combinations(range(D), 4):
            for si in (1, -1):
                for sj in (1, -1):
                    for sk in (1, -1):
                        for sl in (1, -1):
                            v = [0]*D
                            v[i], v[j], v[k], v[l] = si, sj, sk, sl
                            candidates.append(v)
        # 1 non-zero with value 2
        for i in range(D):
            for s in (2, -2):
                v = [0]*D
                v[i] = s
                candidates.append(v)
    elif Q == 5:
        for idxs in itertools.combinations(range(D), 5):
            for signs in itertools.product((1, -1), repeat=5):
                v = [0]*D
                for idx, s in zip(idxs, signs):
                    v[idx] = s
                candidates.append(v)
    return np.array(candidates, dtype=np.int64)

def find_max_clique(candidates, Q, deadline):
    N = len(candidates)
    if N == 0:
        return []
    
    M = candidates @ candidates.T
    threshold = Q / 2.0
    
    A = (M <= threshold)
    np.fill_diagonal(A, False)
    
    degrees = np.sum(A, axis=1)
    order = np.argsort(-degrees)
    
    remaining = set(range(N))
    clique = []
    
    for v in order:
        if time.time() > deadline:
            break
        if v not in remaining:
            continue
        
        clique.append(v)
        
        non_neigh = np.where(~A[v])[0]
        remaining.difference_update(non_neigh)
        
    return clique

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_vectors = None
    best_score = 0
    
    for Q in [2, 3, 4, 5]:
        if time.time() > deadline:
            break
            
        candidates = generate_candidates(Q)
        if len(candidates) == 0:
            continue
            
        clique_indices = find_max_clique(candidates, Q, deadline)
        
        if len(clique_indices) > best_score:
            best_score = len(clique_indices)
            best_vectors = candidates[clique_indices]
            
    if best_vectors is None:
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return best_vectors

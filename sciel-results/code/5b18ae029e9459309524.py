import numpy as np
import random

N = 50

def construct():
    # Build addition conflict graph: edge (i,j) if sum i+j collides with another pair's sum
    # Find maximum independent set via greedy with random restarts
    
    # Compute all pair sums and identify colliding sums
    pair_sums = {}
    for i in range(N):
        for j in range(i, N):
            s = i + j
            if s not in pair_sums:
                pair_sums[s] = []
            pair_sums[s].append((i, j))
    
    # Build conflict adjacency: conflict[i][j] = True if i and j cannot both be in set
    # i and j conflict if their sum i+j has multiple representations
    conflict = [[False] * N for _ in range(N)]
    for s, pairs in pair_sums.items():
        if len(pairs) > 1:
            for (i, j) in pairs:
                if i != j:
                    conflict[i][j] = True
                    conflict[j][i] = True
                # Also, if i+i collides, i conflicts with all other indices in the colliding pairs
                # But for independent set on vertices, we only need pairwise edges
    
    # Greedy independent set with random restarts
    best_set = []
    best_size = 0
    
    for restart in range(500):
        order = list(range(N))
        random.shuffle(order)
        
        chosen = []
        for i in order:
            can_add = True
            for j in chosen:
                if conflict[i][j]:
                    can_add = False
                    break
            if can_add:
                chosen.append(i)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = chosen[:]
    
    if not best_set:
        best_set = [0]
    
    h = np.zeros(N)
    for i in best_set:
        h[i] = 1.0 / len(best_set)
    
    return list(h)

def compute_ratio(h):
    h = np.asarray(h)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_peak = np.max(conv)
    if max_peak == 0:
        return 0.0
    return l2_sq / (l1 * max_peak)

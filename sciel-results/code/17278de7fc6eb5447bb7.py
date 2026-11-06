import numpy as np
import time
import random
import math

N = 64
TOTAL_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Pre-compute squared distances
    xs = np.repeat(np.arange(N), N)
    ys = np.tile(np.arange(N), N)
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_sq = (dx**2 + dy**2).astype(np.int32)
    
    best_size = 0
    best_set = set()
    
    def check_add_pairs(p, S, dist_sets):
        if p in S:
            return None
            
        S_list = list(S)
        if not S_list:
            return []
            
        d_arr = dist_sq[p, S_list]
        
        seen_dists = set()
        pairs = []
        for i, q in enumerate(S_list):
            d = d_arr[i]
            if d in seen_dists:
                return None
            seen_dists.add(d)
            if d in dist_sets[q]:
                return None
            pairs.append((q, d))
            
        return pairs

    def do_add_with_pairs(p, S, dist_sets, pairs):
        S.add(p)
        dist_sets[p] = {d for _, d in pairs}
        for q, d in pairs:
            dist_sets[q].add(d)

    def do_remove(p, S, dist_sets):
        if p not in S:
            return
        S_list = list(S)
        for q in S_list:
            if q == p:
                continue
            d = dist_sq[p, q]
            dist_sets[q].remove(d)
        del dist_sets[p]
        S.remove(p)

    def greedy_fill(S, dist_sets, indices):
        for p in indices:
            if p in S:
                continue
            pairs = check_add_pairs(p, S, dist_sets)
            if pairs is not None:
                do_add_with_pairs(p, S, dist_sets, pairs)
            if time.time() > deadline - 20:
                break

    def local_search_sa(S, dist_sets):
        current_size = len(S)
        best_size_local = current_size
        best_set_local = S.copy()
        best_dist_local = {k: v.copy() for k, v in dist_sets.items()}
        
        T = 10.0
        min_T = 0.1
        cooling = 0.99
        
        while time.time() < deadline - 5:
            move_type = random.random()
            
            if move_type < 0.7:
                p = random.randint(0, TOTAL_POINTS - 1)
                if p in S:
                    continue
                
                pairs = check_add_pairs(p, S, dist_sets)
                if pairs is not None:
                    do_add_with_pairs(p, S, dist_sets, pairs)
                    current_size += 1
                    if current_size > best_size_local:
                        best_size_local = current_size
                        best_set_local = S.copy()
                        best_dist_local = {k: v.copy() for k, v in dist_sets.items()}
            else:
                if not S:
                    continue
                p = random.choice(list(S))
                prob = math.exp(-1.0 / T)
                if random.random() < prob:
                    do_remove(p, S, dist_sets)
                    current_size -= 1
            
            T *= cooling
            if T < min_T:
                T = min_T
            if random.random() < 0.01:
                T = 5.0

        return best_set_local, best_dist_local

    num_trials = 0
    while time.time() < deadline - 10:
        indices = list(range(TOTAL_POINTS))
        random.shuffle(indices)
        
        S = set()
        dist_sets = {}
        
        greedy_fill(S, dist_sets, indices)
        
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_set = S.copy()
            
        if time.time() < deadline - 20:
            S_imp, dist_sets_imp = local_search_sa(S, dist_sets)
            if len(S_imp) > best_size:
                best_size = len(S_imp)
                best_set = S_imp.copy()
        
        num_trials += 1

    result = []
    for idx in best_set:
        x = idx // N
        y = idx % N
        result.append((x, y))
        
    return result

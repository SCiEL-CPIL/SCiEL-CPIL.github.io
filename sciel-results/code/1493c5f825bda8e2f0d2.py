import numpy as np
import time
import random
import math

N = 64
TOTAL_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    xs = np.repeat(np.arange(N), N)
    ys = np.tile(np.arange(N), N)
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_sq = (dx**2 + dy**2).astype(np.int32)
    
    best_size = 0
    best_set = set()
    
    def try_add(p, S, dist_sets):
        if p in S:
            return False
        S_list = list(S)
        if not S_list:
            S.add(p)
            dist_sets[p] = set()
            return True
        d_arr = dist_sq[p, S_list]
        if len(set(d_arr.tolist())) != len(d_arr):
            return False
        for i, q in enumerate(S_list):
            if d_arr[i] in dist_sets[q]:
                return False
        S.add(p)
        dist_sets[p] = set(d_arr.tolist())
        for i, q in enumerate(S_list):
            dist_sets[q].add(d_arr[i])
        return True

    def remove_point(p, S, dist_sets):
        if p not in S:
            return
        S.remove(p)
        for q in S:
            dist_sets[q].remove(dist_sq[p, q])
        del dist_sets[p]

    def greedy_fill(S, dist_sets, indices):
        for p in indices:
            if p in S:
                continue
            try_add(p, S, dist_sets)
            if time.time() > deadline - 10:
                return

    def local_search(S, dist_sets):
        current_size = len(S)
        best_local_size = current_size
        best_local_set = S.copy()
        best_local_dist_sets = {k: v.copy() for k, v in dist_sets.items()}
        T = 1.0
        
        while time.time() < deadline - 5:
            T *= 0.99
            if T < 0.01: T = 0.01
            
            move_type = random.random()
            accepted = False
            new_size = current_size
            p = -1
            p_remove = -1
            
            if move_type < 0.6:
                complement = [i for i in range(TOTAL_POINTS) if i not in S]
                if complement:
                    p = random.choice(complement)
                    if try_add(p, S, dist_sets):
                        new_size = len(S)
                        accepted = True
            elif move_type < 0.8:
                if S:
                    p = random.choice(list(S))
                    remove_point(p, S, dist_sets)
                    new_size = len(S)
                    accepted = True
            else:
                if S:
                    p_remove = random.choice(list(S))
                    complement = [i for i in range(TOTAL_POINTS) if i not in S]
                    if complement:
                        p_add = random.choice(complement)
                        remove_point(p_remove, S, dist_sets)
                        if try_add(p_add, S, dist_sets):
                            new_size = len(S)
                            accepted = True
                        else:
                            new_size = len(S)
                            accepted = True
                            p = p_remove # For revert logic

            if accepted:
                delta = new_size - current_size
                if delta > 0 or random.random() < math.exp(-delta / T):
                    current_size = new_size
                    if current_size > best_local_size:
                        best_local_size = current_size
                        best_local_set = S.copy()
                        best_local_dist_sets = {k: v.copy() for k, v in dist_sets.items()}
                else:
                    # Revert
                    if move_type >= 0.6 and move_type < 0.8:
                        try_add(p, S, dist_sets)
                    elif move_type >= 0.8:
                        try_add(p_remove, S, dist_sets)
                    current_size = len(S)

        return best_local_set, best_local_dist_sets

    num_trials = 0
    while time.time() < deadline - 10:
        indices = list(range(TOTAL_POINTS))
        random.shuffle(indices)
        S = set()
        dist_sets = {}
        greedy_fill(S, dist_sets, indices)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
        if time.time() < deadline - 20:
            S_imp, dist_sets_imp = local_search(S, dist_sets)
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

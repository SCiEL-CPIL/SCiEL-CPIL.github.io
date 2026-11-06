import random
import time
from itertools import combinations

def construct():
    n = 34
    triples = list(combinations(range(n), 3))
    T = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    k4s = list(combinations(range(n), 4))
    k5s = list(combinations(range(n), 5))
    
    num_k4 = len(k4s)
    num_k5 = len(k5s)
    
    k4_triples = []
    for k4 in k4s:
        t_inds = [triple_to_idx[t] for t in combinations(k4, 3)]
        k4_triples.append(t_inds)
        
    k5_triples = []
    for k5 in k5s:
        t_inds = [triple_to_idx[t] for t in combinations(k5, 3)]
        k5_triples.append(t_inds)
        
    triple_to_hedges = [[] for _ in range(T)]
    for i, t_inds in enumerate(k4_triples):
        for t in t_inds:
            triple_to_hedges[t].append((0, i))
    for i, t_inds in enumerate(k5_triples):
        for t in t_inds:
            triple_to_hedges[t].append((1, i))

    p_red = 0.45
    colors = [0 if random.random() < p_red else 1 for _ in range(T)]
    
    def is_violated_k4(idx):
        return all(colors[t] == 0 for t in k4_triples[idx])
        
    def is_violated_k5(idx):
        return all(colors[t] == 1 for t in k5_triples[idx])

    is_violated_k4_arr = [False] * num_k4
    is_violated_k5_arr = [False] * num_k5
    
    violated_k4_list = []
    violated_k5_list = []
    pos_in_k4_list = {}
    pos_in_k5_list = {}
    
    for i in range(num_k4):
        if is_violated_k4(i):
            is_violated_k4_arr[i] = True
            violated_k4_list.append(i)
            pos_in_k4_list[i] = len(violated_k4_list) - 1
            
    for i in range(num_k5):
        if is_violated_k5(i):
            is_violated_k5_arr[i] = True
            violated_k5_list.append(i)
            pos_in_k5_list[i] = len(violated_k5_list) - 1
            
    total_violations = len(violated_k4_list) + len(violated_k5_list)
    
    start_time = time.time()
    timeout = 115
    
    while total_violations > 0 and time.time() - start_time < timeout:
        # Pick a violated hyperedge
        # Probability proportional to count
        r = random.random() * total_violations
        if r < len(violated_k4_list):
            h_idx = random.choice(violated_k4_list)
            h_type = 0
        else:
            h_idx = random.choice(violated_k5_list)
            h_type = 1
            
        if h_type == 0:
            t_inds = k4_triples[h_idx]
        else:
            t_inds = k5_triples[h_idx]
            
        t_idx = random.choice(t_inds)
        
        # Flip
        colors[t_idx] = 1 - colors[t_idx]
        
        # Update
        for h_type, h_idx in triple_to_hedges[t_idx]:
            if h_type == 0:
                old = is_violated_k4_arr[h_idx]
                new = is_violated_k4(h_idx)
                if old != new:
                    is_violated_k4_arr[h_idx] = new
                    if new:
                        violated_k4_list.append(h_idx)
                        pos_in_k4_list[h_idx] = len(violated_k4_list) - 1
                        total_violations += 1
                    else:
                        pos = pos_in_k4_list[h_idx]
                        last = violated_k4_list[-1]
                        violated_k4_list[pos] = last
                        pos_in_k4_list[last] = pos
                        violated_k4_list.pop()
                        del pos_in_k4_list[h_idx]
                        total_violations -= 1
            else:
                old = is_violated_k5_arr[h_idx]
                new = is_violated_k5(h_idx)
                if old != new:
                    is_violated_k5_arr[h_idx] = new
                    if new:
                        violated_k5_list.append(h_idx)
                        pos_in_k5_list[h_idx] = len(violated_k5_list) - 1
                        total_violations += 1
                    else:
                        pos = pos_in_k5_list[h_idx]
                        last = violated_k5_list[-1]
                        violated_k5_list[pos] = last
                        pos_in_k5_list[last] = pos
                        violated_k5_list.pop()
                        del pos_in_k5_list[h_idx]
                        total_violations -= 1
                        
    if total_violations == 0:
        return {'n': n, 'colors': colors}
    else:
        n_seed = 9
        return {'n': n_seed, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_seed), 3)]}

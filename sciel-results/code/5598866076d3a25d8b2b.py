import numpy as np
import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * 15 - i * (i - 1) // 2 + (j - i - 1)

all_cycles = []
for quad in combinations(range(N), 4):
    a, b, c, d = quad
    all_cycles.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)))
    all_cycles.append((edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)))
    all_cycles.append((edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)))

all_cycles = np.array(all_cycles, dtype=np.int32)

def count_violations_fast(colors):
    cc = colors[all_cycles]
    sorted_cc = np.sort(cc, axis=1)
    return int(np.sum(np.all(sorted_cc[:, 1:] != sorted_cc[:, :-1], axis=1)))

def construct():
    start_time = time.time()
    DEADLINE = 82.0
    
    def initial_coloring():
        c = np.zeros(NUM_EDGES, dtype=np.int32)
        for i in range(N):
            for j in range(i+1, N):
                c[edge_idx(i,j)] = i
        return c
    
    colors = initial_coloring()
    used_colors = set(colors.tolist())
    best_colors = colors.copy()
    best_nc = len(used_colors)
    
    random.seed(42)
    
    # Phase 1: Greedy addition of new colors
    new_colors = list(range(15, 120))
    random.shuffle(new_colors)
    
    for nc in new_colors:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        edges_to_try = list(range(NUM_EDGES))
        random.shuffle(edges_to_try)
        
        placed = False
        for e in edges_to_try:
            if time.time() - start_time > DEADLINE - 10:
                break
            
            old_c = colors[e]
            colors[e] = nc
            
            if count_violations_fast(colors) == 0:
                used_colors.add(nc)
                best_colors = colors.copy()
                best_nc = len(used_colors)
                placed = True
                break
            else:
                colors[e] = old_c
    
    # Phase 2: Local search with multiple restarts
    if time.time() - start_time < DEADLINE - 3:
        for restart in range(20):
            if time.time() - start_time > DEADLINE - 3:
                break
            
            # Perturb current best
            cur = best_colors.copy()
            for _ in range(5):
                e = random.randint(0, NUM_EDGES - 1)
                cur[e] = random.randint(0, 119)
            # Fix violations
            for _ in range(200):
                if time.time() - start_time > DEADLINE - 3:
                    break
                viol = count_violations_fast(cur)
                if viol == 0:
                    break
                # Find a violating cycle and fix it
                cc = cur[all_cycles]
                sorted_cc = np.sort(cc, axis=1)
                is_rainbow = np.all(sorted_cc[:, 1:] != sorted_cc[:, :-1], axis=1)
                if not np.any(is_rainbow):
                    break
                viol_idx = np.where(is_rainbow)[0]
                vi = viol_idx[0]
                cycle = all_cycles[vi]
                # Change one edge in the cycle to match another
                target_edge = cycle[random.randint(0, 3)]
                target_color = cur[cycle[random.randint(0, 3)]]
                cur[target_edge] = target_color
            
            # Now do local search from here
            for _ in range(5000):
                if time.time() - start_time > DEADLINE - 3:
                    break
                
                e = random.randint(0, NUM_EDGES - 1)
                old_c = cur[e]
                new_c = random.randint(0, 119)
                if new_c == old_c:
                    continue
                
                cur[e] = new_c
                if count_violations_fast(cur) == 0:
                    nc = len(set(cur.tolist()))
                    if nc > best_nc:
                        best_colors = cur.copy()
                        best_nc = nc
                else:
                    cur[e] = old_c
    
    # Final verification
    if count_violations_fast(best_colors) != 0:
        return initial_coloring().tolist()
    
    return best_colors.tolist()

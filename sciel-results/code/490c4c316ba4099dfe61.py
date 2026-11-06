import time
import random
import math
from itertools import combinations

N_MAX = 40

def precompute_incidence(n):
    triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    num_edges = len(triples)
    
    k4s = list(combinations(range(n), 4))
    k5s = list(combinations(range(n), 5))
    
    edge_to_k4 = [[] for _ in range(num_edges)]
    edge_to_k5 = [[] for _ in range(num_edges)]
    
    for k4_idx, k4 in enumerate(k4s):
        a, b, c, d = k4
        edges = [
            (a, b, c),
            (a, b, d),
            (a, c, d),
            (b, c, d)
        ]
        for e in edges:
            e_idx = triple_to_idx[e]
            edge_to_k4[e_idx].append(k4_idx)
            
    for k5_idx, k5 in enumerate(k5s):
        edges = list(combinations(k5, 3))
        for e in edges:
            e_idx = triple_to_idx[e]
            edge_to_k5[e_idx].append(k5_idx)
            
    return triples, edge_to_k4, edge_to_k5, len(k4s), len(k5s)

def solve_for_n(n, time_limit):
    if n > N_MAX:
        return None, False
    
    triples, edge_to_k4, edge_to_k5, num_k4, num_k5 = precompute_incidence(n)
    num_edges = len(triples)
    
    start_time = time.time()
    
    for restart in range(100):
        if time.time() - start_time > time_limit:
            break
            
        colors = [random.randint(0, 1) for _ in range(num_edges)]
        
        red_count = [0] * num_k4
        blue_count = [0] * num_k5
        
        energy = 0
        for e_idx, c in enumerate(colors):
            if c == 0:
                for k4_idx in edge_to_k4[e_idx]:
                    red_count[k4_idx] += 1
            else:
                for k5_idx in edge_to_k5[e_idx]:
                    blue_count[k5_idx] += 1
        
        for cnt in red_count:
            if cnt == 4: energy += 1
        for cnt in blue_count:
            if cnt == 10: energy += 1
            
        if energy == 0:
            return colors, True
            
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        for step in range(5000):
            if time.time() - start_time > time_limit:
                break
                
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            
            delta = 0
            
            if old_color == 0:
                for k4_idx in edge_to_k4[e_idx]:
                    if red_count[k4_idx] == 4: delta -= 1
            else:
                for k4_idx in edge_to_k4[e_idx]:
                    if red_count[k4_idx] == 3: delta += 1
            
            if old_color == 1:
                for k5_idx in edge_to_k5[e_idx]:
                    if blue_count[k5_idx] == 10: delta -= 1
            else:
                for k5_idx in edge_to_k5[e_idx]:
                    if blue_count[k5_idx] == 9: delta += 1
            
            # Acceptance
            accept = False
            if delta < 0:
                accept = True
            else:
                # Optimization: if delta is large, exp is small
                if delta > T * 10:
                    accept = False
                else:
                    accept = random.random() < math.exp(-delta / T)
            
            if accept:
                colors[e_idx] = 1 - old_color
                energy += delta
                
                if old_color == 0:
                    for k4_idx in edge_to_k4[e_idx]:
                        red_count[k4_idx] -= 1
                else:
                    for k4_idx in edge_to_k4[e_idx]:
                        red_count[k4_idx] += 1
                        
                if old_color == 1:
                    for k5_idx in edge_to_k5[e_idx]:
                        blue_count[k5_idx] -= 1
                else:
                    for k5_idx in edge_to_k5[e_idx]:
                        blue_count[k5_idx] += 1
                
                if energy == 0:
                    return colors, True
            
            T *= cooling
            if T < min_T:
                T = min_T
                
    return None, False

def construct():
    start_time = time.time()
    max_time = 140.0
    
    best_n = 0
    best_colors = []
    
    targets = [40, 39, 38, 37, 36, 35, 34]
    
    for n in targets:
        if time.time() - start_time > max_time:
            break
        
        time_left = max_time - (time.time() - start_time)
        # Allocate time: 15s for 40, 15s for 39, 10s for 38, 10s for 37, 10s for 36, 5s for 35, 5s for 34
        # Total 70s.
        if n == 40: slice_time = 15.0
        elif n == 39: slice_time = 15.0
        elif n == 38: slice_time = 10.0
        elif n == 37: slice_time = 10.0
        elif n == 36: slice_time = 10.0
        elif n == 35: slice_time = 5.0
        else: slice_time = 5.0
        
        slice_time = min(slice_time, time_left)
        
        if slice_time < 1.0:
            continue
            
        colors, success = solve_for_n(n, slice_time)
        
        if success:
            best_n = n
            best_colors = colors
            break
            
    if best_n == 0:
        best_n = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    return {'n': best_n, 'colors': best_colors}

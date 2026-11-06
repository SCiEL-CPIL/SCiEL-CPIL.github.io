import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time
import random

def _fallback_greedy():
    d = 7
    n = 2 ** d
    edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
                
    def can_add(u, v, chosen_set):
        for i in range(d):
            w = u ^ (1 << i)
            if w == v:
                continue
            if (min(u, w), max(u, w)) in chosen_set:
                for j in range(d):
                    z = v ^ (1 << j)
                    if z == u:
                        continue
                    if (min(v, z), max(v, z)) in chosen_set:
                        if (min(w, z), max(w, z)) in chosen_set:
                            return False
        return True

    current_set = set()
    random.shuffle(edges)
    for e in edges:
        u, v = e
        if can_add(u, v, current_set):
            current_set.add(e)
            
    start_time = time.time()
    max_time = 100.0
    best_size = len(current_set)
    best_set = set(current_set)
    
    while time.time() - start_time < max_time:
        kick_size = 5
        if len(current_set) < kick_size:
            break
        to_remove = random.sample(list(current_set), kick_size)
        for e in to_remove:
            current_set.remove(e)
        random.shuffle(edges)
        for e in edges:
            u, v = e
            if can_add(u, v, current_set):
                current_set.add(e)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
    return [list(e) for e in best_set]

def construct():
    d = 7
    n = 2 ** d
    
    # 1. Generate all edges
    edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
                edge_to_idx[(y, x)] = idx
                
    num_edges = len(edges)
    
    # 2. Generate all 4-cycles
    cycles = []
    
    for i in range(d):
        for j in range(i + 1, d):
            bit_i = 1 << i
            bit_j = 1 << j
            mask = bit_i | bit_j
            
            for v in range(n):
                if (v & mask) != 0:
                    continue
                
                v1 = v
                v2 = v ^ bit_i
                v3 = v ^ bit_i ^ bit_j
                v4 = v ^ bit_j
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v2, v3), max(v2, v3))
                e3 = (min(v3, v4), max(v3, v4))
                e4 = (min(v4, v1), max(v4, v1))
                
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                cycles.append([idx1, idx2, idx3, idx4])
                
    num_cycles = len(cycles)
    
    # 3. Formulate ILP
    c = np.ones(num_edges)
    
    A = np.zeros((num_cycles, num_edges), dtype=np.int8)
    for k, cycle in enumerate(cycles):
        for idx in cycle:
            A[k, idx] = 1
            
    b_lower = np.zeros(num_cycles)
    b_upper = np.full(num_cycles, 3)
    
    constraints = LinearConstraint(A, b_lower, b_upper)
    integrality = np.ones(num_edges)
    bounds = Bounds(0, 1)
    
    start_time = time.time()
    timeout = 110.0
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, 
                   options={"time_limit": max(1.0, timeout - (time.time() - start_time))})
    except Exception:
        return _fallback_greedy()

    if res.status == 0 or res.status == 1:
        x = res.x
        selected_edges = []
        for i in range(num_edges):
            if x[i] > 0.5:
                selected_edges.append(list(edges[i]))
        return selected_edges
    else:
        return _fallback_greedy()

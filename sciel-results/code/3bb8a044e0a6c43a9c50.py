import time
import random
import numpy as np
from itertools import combinations

DEADLINE = 115.0

def construct():
    start = time.time()
    n = 20
    num_colors = 20
    
    all_edges = []
    for u in range(n):
        for v in range(u+1, n):
            all_edges.append((u, v))
    m = len(all_edges)
    
    edge_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i
    
    cycle_edges = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        cycle_edges.append([edge_idx[(a,b)], edge_idx[(b,c)], edge_idx[(c,d)], edge_idx[(a,d)]])
        cycle_edges.append([edge_idx[(a,b)], edge_idx[(b,d)], edge_idx[(c,d)], edge_idx[(a,c)]])
        cycle_edges.append([edge_idx[(a,c)], edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(a,d)]])
    
    num_cycles = len(cycle_edges)
    
    edge_to_cycles = [[] for _ in range(m)]
    for ci, ce in enumerate(cycle_edges):
        for ei in ce:
            edge_to_cycles[ei].append(ci)
    
    # Fast rainbow check: 4 distinct values
    def is_rainbow(c0, c1, c2, c3):
        return c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3
    
    # Initialize: try a structured colouring first
    # Colour by distance or some pattern
    color_of = np.zeros(m, dtype=np.int32)
    for i, (u, v) in enumerate(all_edges):
        color_of[i] = (u + v) % num_colors
    
    # Compute initial conflicts
    conflict = np.zeros(num_cycles, dtype=np.int32)
    for ci in range(num_cycles):
        ce = cycle_edges[ci]
        c0, c1, c2, c3 = int(color_of[ce[0]]), int(color_of[ce[1]]), int(color_of[ce[2]]), int(color_of[ce[3]])
        if not is_rainbow(c0, c1, c2, c3):
            conflict[ci] = 1
    total_conflicts = int(conflict.sum())
    best_conflicts = total_conflicts
    best_colors = color_of.copy()
    
    # If already zero, return
    if total_conflicts == 0:
        result = []
        for i, (u, v) in enumerate(all_edges):
            result.append([u, v, int(color_of[i])])
        return result
    
    # Simulated annealing with fast delta computation
    T = 3.0
    T_min = 0.001
    cooling = 0.999
    
    # For speed, precompute cycle edge arrays as lists
    ce_list = [list(ce) for ce in cycle_edges]
    
    while time.time() - start < DEADLINE:
        # Inner loop
        for _ in range(100):
            if time.time() - start > DEADLINE:
                break
            ei = random.randint(0, m-1)
            old_c = int(color_of[ei])
            new_c = random.randint(0, num_colors-1)
            if old_c == new_c:
                continue
            
            delta = 0
            for ci in edge_to_cycles[ei]:
                ce = ce_list[ci]
                c0 = int(color_of[ce[0]])
                c1 = int(color_of[ce[1]])
                c2 = int(color_of[ce[2]])
                c3 = int(color_of[ce[3]])
                before = 0 if is_rainbow(c0, c1, c2, c3) else 1
                # Replace
                for j in range(4):
                    if ce[j] == ei:
                        if j == 0: c0 = new_c
                        elif j == 1: c1 = new_c
                        elif j == 2: c2 = new_c
                        else: c3 = new_c
                        break
                after = 0 if is_rainbow(c0, c1, c2, c3) else 1
                delta += after - before
            
            if delta <= 0:
                color_of[ei] = new_c
                for ci in edge_to_cycles[ei]:
                    ce = ce_list[ci]
                    c0, c1, c2, c3 = int(color_of[ce[0]]), int(color_of[ce[1]]), int(color_of[ce[2]]), int(color_of[ce[3]])
                    conflict[ci] = 0 if is_rainbow(c0, c1, c2, c3) else 1
                total_conflicts += delta
                if total_conflicts < best_conflicts:
                    best_conflicts = total_conflicts
                    best_colors = color_of.copy()
            else:
                if random.random() < np.exp(-delta / T):
                    color_of[ei] = new_c
                    for ci in edge_to_cycles[ei]:
                        ce = ce_list[ci]
                        c0, c1, c2, c3 = int(color_of[ce[0]]), int(color_of[ce[1]]), int(color_of[ce[2]]), int(color_of[ce[3]])
                        conflict[ci] = 0 if is_rainbow(c0, c1, c2, c3) else 1
                    total_conflicts += delta
            
            T = max(T_min, T * cooling)
        
        if best_conflicts == 0:
            break
    
    if best_conflicts == 0:
        result = []
        for i, (u, v) in enumerate(all_edges):
            result.append([u, v, int(best_colors[i])])
        return result
    
    # Fallback: remove edges to fix
    colors = best_colors.copy()
    active = np.ones(m, dtype=bool)
    
    for _ in range(200):
        if time.time() - start > DEADLINE - 2:
            break
        # Find first conflicting active cycle
        removed = False
        for ci in range(num_cycles):
            ce = ce_list[ci]
            if not all(active[ce[j]] for j in range(4)):
                continue
            c0, c1, c2, c3 = int(colors[ce[0]]), int(colors[ce[1]]), int(colors[ce[2]]), int(colors[ce[3]])
            if not is_rainbow(c0, c1, c2, c3):
                # Remove edge with highest conflict involvement
                # Just remove ce[0]
                active[ce[0]] = False
                removed = True
                break
        if not removed:
            break
    
    result = []
    for i, (u, v) in enumerate(all_edges):
        if active[i]:
            result.append([u, v, int(colors[i])])
    
    return result


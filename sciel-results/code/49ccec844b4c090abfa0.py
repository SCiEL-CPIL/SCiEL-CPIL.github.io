import time
import random
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Try n from 17 down to 14
    for n in [17, 16, 15, 14]:
        if time.time() > deadline - 10.0:
            break
        res = _solve_n(n, deadline)
        if res:
            return res
            
    # Fallback: try n=13 if we have time
    if time.time() < deadline - 5.0:
        res = _solve_n(13, deadline)
        if res:
            return res
            
    # Absolute fallback
    n = 9
    colors = [0] * (n*(n-1)//2)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            colors[idx] = (j-i) % 4
            idx += 1
    return {'n': n, 'colors': colors}

def _solve_n(n, deadline):
    # Precompute cycles
    cycles, edge_to_cycles = _get_cycles(n)
    num_edges = n*(n-1)//2
    num_cycles = len(cycles)
    
    if num_cycles == 0:
         return {'n': n, 'colors': [0]*num_edges}
         
    # Convert cycles to numpy array for fast indexing
    cycles_np = np.array(cycles, dtype=np.int32)
    edge_to_cycles_np = [np.array(lst, dtype=np.int32) for lst in edge_to_cycles]
    
    # Initialize colors
    colors = np.random.randint(0, 4, num_edges)
    
    # Calculate initial cost
    is_mono = _calc_mono(cycles_np, colors)
    cost = np.sum(is_mono)
    
    if cost == 0:
        return {'n': n, 'colors': colors.tolist()}
        
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    best_cost = cost
    best_colors = colors.copy()
    
    # We will run the loop until deadline
    while time.time() < deadline - 2.0:
        e = np.random.randint(num_edges)
        old_c = colors[e]
        new_c = np.random.randint(0, 4)
        if new_c == old_c:
            continue
            
        delta = _delta_cost(e, new_c, colors, is_mono, cycles_np, edge_to_cycles_np)
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            colors[e] = new_c
            # Update is_mono for affected cycles
            affected = edge_to_cycles_np[e]
            if len(affected) > 0:
                affected_cycles = cycles_np[affected]
                affected_colors = colors[affected_cycles]
                is_mono[affected] = np.all(affected_colors == affected_colors[:, 0:1], axis=1)
            cost += delta
            
            if cost < best_cost:
                best_cost = cost
                best_colors = colors.copy()
                if best_cost == 0:
                    return {'n': n, 'colors': best_colors.tolist()}
                    
        T *= cooling
        if T < T_min:
            T = T_min
            
    if best_cost == 0:
        return {'n': n, 'colors': best_colors.tolist()}
        
    return None

def _calc_mono(cycles_np, colors):
    if len(cycles_np) == 0:
        return np.array([], dtype=bool)
    cycle_colors = colors[cycles_np]
    return np.all(cycle_colors == cycle_colors[:, 0:1], axis=1)

def _delta_cost(e, new_c, colors, is_mono, cycles_np, edge_to_cycles_np):
    affected = edge_to_cycles_np[e]
    if len(affected) == 0:
        return 0
        
    affected_cycles = cycles_np[affected]
    
    # Check which were mono
    was_mono = is_mono[affected]
    delta = -np.sum(was_mono)
    
    # Check which will be mono with new_c
    affected_colors = colors[affected_cycles]
    
    mask = (affected_colors == new_c)
    counts = np.sum(mask, axis=1)
    will_be_mono = (counts == 5)
    
    delta += np.sum(will_be_mono)
    
    return delta

def _get_cycles(n):
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_idx[(i, j)] = idx
            idx += 1
    num_edges = idx
            
    # Generate base patterns for K6
    base_patterns = []
    seen = set()
    for perm in itertools.permutations([1, 2, 3, 4, 5]):
        v = [0] + list(perm)
        edges_in_cycle = []
        for i in range(6):
            u, w = v[i], v[(i+1)%6]
            if u > w: u, w = w, u
            edges_in_cycle.append((u, w))
        edge_set = frozenset(edges_in_cycle)
        if edge_set not in seen:
            seen.add(edge_set)
            base_patterns.append(tuple(sorted(edges_in_cycle)))
            
    cycles = []
    edge_to_cycles = [[] for _ in range(num_edges)]
    
    for verts in itertools.combinations(range(n), 6):
        for pat in base_patterns:
            global_e_idxs = []
            for (u_local, w_local) in pat:
                u_g = verts[u_local]
                w_g = verts[w_local]
                e_idx = edge_idx[(u_g, w_g)]
                global_e_idxs.append(e_idx)
                
            cycle_idx = len(cycles)
            cycles.append(global_e_idxs)
            
            for e in global_e_idxs:
                edge_to_cycles[e].append(cycle_idx)
                
    return cycles, edge_to_cycles

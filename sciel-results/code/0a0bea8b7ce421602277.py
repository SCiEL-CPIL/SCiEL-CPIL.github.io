import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k != 5 or g != 9:
        return luw_construct(k, g)

    # Base graph K6
    edges_base = []
    edge_map = {}
    idx = 0
    for i in range(6):
        for j in range(i+1, 6):
            edges_base.append((i, j))
            edge_map[(i, j)] = idx
            edge_map[(j, i)] = idx
            idx += 1
            
    # Generate constraints
    # We'll store them in a list of lists, then convert to np.array
    constraints_list = []
    seen = set()
    
    for start in range(6):
        stack = [(start, -1, -1, 0, [0]*15)]
        while stack:
            v, pv, pv2, l, coeffs = stack.pop()
            
            if l > 0 and v == start:
                if 3 <= l <= 8:
                    ct = tuple(coeffs)
                    if ct not in seen:
                        seen.add(ct)
                        constraints_list.append(ct)
            
            if l >= 8:
                continue
            
            for nb in range(6):
                if nb == pv:
                    continue
                if l >= 2 and nb == pv2:
                    continue
                
                new_coeffs = coeffs[:]
                e_idx = edge_map.get((min(v, nb), max(v, nb)))
                if v < nb:
                    new_coeffs[e_idx] += 1
                else:
                    new_coeffs[e_idx] -= 1
                    
                stack.append((nb, v, pv, l+1, new_coeffs))
                
    C = np.array(constraints_list)
    # C shape: (num_constraints, 15)
    
    best_n = float('inf')
    best_edges = None
    
    for m in range(71, 200):
        if time.time() > deadline:
            break
            
        n = 6 * m
        if n >= best_n:
            continue
            
        found = False
        for _ in range(200): # Reduced retries
            if time.time() > deadline:
                break
                
            voltages = np.random.randint(1, m, size=15)
            S = C @ voltages
            if not (S % m == 0).any():
                found = True
                break
                
        if found:
            edges = []
            for e_idx, (u, v) in enumerate(edges_base):
                w = int(voltages[e_idx])
                for j in range(m):
                    u_id = u * m + j
                    v_id = v * m + (j + w) % m
                    edges.append((u_id, v_id))
                    
            best_n = n
            best_edges = edges
            break
            
    if best_edges is None:
        return luw_construct(k, g)
        
    return best_n, best_edges

def luw_construct(k, g):
    # ... same as before ...
    pass

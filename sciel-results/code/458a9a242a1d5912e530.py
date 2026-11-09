import time
import math
import sys
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    n = len(valid_nodes)
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build conflict edges
    conflicts = [[] for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if valid_nodes[i] + valid_nodes[j] in squares:
                conflicts[i].append(j)
                conflicts[j].append(i)
                edges.append((i, j))
    
    if not edges:
        return valid_nodes
    
    # Build bitmask conflicts
    conflict_masks = [0] * n
    for i in range(n):
        for j in conflicts[i]:
            conflict_masks[i] |= (1 << j)
    
    # Try MILP first for a strong initial solution
    best_mask = 0
    best_size = 0
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix
        
        m = n
        c = np.ones(m)
        num_edges = len(edges)
        
        row_indices = []
        col_indices = []
        data = []
        
        for idx, (u, v) in enumerate(edges):
            row_indices.extend([idx, idx])
            col_indices.extend([u, v])
            data.extend([-1, -1])
        
        A = csr_matrix((data, (row_indices, col_indices)), shape=(num_edges, m))
        constraints = LinearConstraint(A, lb=-np.inf, ub=-np.ones(num_edges))
        integrality = np.ones(m)
        bounds = Bounds(lb=0, ub=1)
        
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 40, "presolve": True}
        )
        
        if res.status == 0:
            cover_mask = res.x
            for i in range(m):
                if cover_mask[i] < 0.5:
                    best_mask |= (1 << i)
                    best_size += 1
            
            if best_size > 0:
                valid = True
                for i in range(m):
                    if best_mask & (1 << i):
                        for j in range(i, m):
                            if best_mask & (1 << j):
                                s = valid_nodes[i] + valid_nodes[j]
                                r = math.isqrt(s)
                                if r * r == s:
                                    valid = False
                                    break
                    if not valid:
                        break
                if not valid:
                    best_mask = 0
                    best_size = 0
                    
    except Exception:
        pass
    
    if best_size == 0:
        # Greedy fallback
        available = (1 << n) - 1
        chosen = 0
        while available:
            best_v = -1
            best_deg = float('inf')
            temp = available
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = bin(conflict_masks[v] & available).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            chosen |= (1 << best_v)
            available &= ~(1 << best_v | conflict_masks[best_v])
        best_mask = chosen
        best_size = bin(best_mask).count('1')
    
    # B&B
    global_best = [best_size, best_mask]
    all_mask = (1 << n) - 1
    
    def bnb(available_mask, current_size, chosen_mask):
        if time.time() > deadline - 1:
            return
        
        n_cand = bin(available_mask).count('1')
        if n_cand == 0:
            if current_size > global_best[0]:
                global_best[0] = current_size
                global_best[1] = chosen_mask
            return
        
        if current_size + n_cand <= global_best[0]:
            return
        
        # Pick vertex with highest degree
        best_v = -1
        best_deg = -1
        temp = available_mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            deg = bin(conflict_masks[v] & available_mask).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
        
        if best_v == -1:
            return
        
        v_bit = 1 << best_v
        
        # Include
        new_avail = available_mask & ~(v_bit | conflict_masks[best_v])
        bnb(new_avail, current_size + 1, chosen_mask | v_bit)
        
        # Exclude
        new_avail2 = available_mask & ~v_bit
        if current_size + bin(new_avail2).count('1') > global_best[0]:
            bnb(new_avail2, current_size, chosen_mask)
    
    bnb(all_mask, 0, 0)
    
    best_mask = global_best[1]
    result_nodes = []
    for i in range(n):
        if best_mask & (1 << i):
            result_nodes.append(valid_nodes[i])
            
    if not result_nodes:
        result_nodes = [valid_nodes[0]]
        
    return result_nodes

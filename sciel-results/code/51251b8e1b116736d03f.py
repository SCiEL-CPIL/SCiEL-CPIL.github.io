import time
import random
import numpy as np
from collections import defaultdict

def construct(k, g):
    # For k=4, g=10, we use a voltage graph lift of K_3,3
    if k != 4 or g != 10:
        return _luw_fallback(k, g)
    
    deadline = time.time() + 110.0
    
    # Base graph K_3,3
    # Vertices: 0,1,2 (left), 3,4,5 (right)
    # Edges:
    # e0: 0-3
    # e1: 0-4
    # e2: 0-5
    # e3: 1-3
    # e4: 1-4
    # e5: 1-5
    # e6: 2-3
    # e7: 2-4
    # e8: 2-5
    
    # We need to find voltages v[0..8] in Z_m such that
    # for every closed walk of length L in {4, 6, 8} in K_3,3,
    # the sum of voltages (with signs) is not 0 mod m.
    
    # Precompute all closed walks of length 4, 6, 8
    # A walk is a sequence of vertices v0, v1, ..., vL=v0
    # such that (vi, vi+1) is an edge.
    # We represent a walk by the sequence of edge indices and directions.
    # Direction: +1 if left->right, -1 if right->left.
    
    def get_walks(max_len):
        walks = []
        # Start from each vertex
        for start in range(6):
            # DFS to generate walks
            def dfs(curr, prev, length, path_edges, path_dirs):
                if length == 0:
                    if curr == start:
                        # Found a closed walk
                        # Store as list of (edge_idx, dir)
                        walks.append((list(path_edges), list(path_dirs)))
                    return
                
                # Determine possible next vertices
                for next_v in range(6):
                    if next_v == prev:
                        continue
                    # Check if edge exists
                    if (curr < 3 and next_v >= 3) or (curr >= 3 and next_v < 3):
                        if curr < 3:
                            # left to right
                            idx = curr * 3 + (next_v - 3)
                            dir = 1
                        else:
                            # right to left
                            idx = next_v * 3 + (curr - 3)
                            dir = -1
                        path_edges.append(idx)
                        path_dirs.append(dir)
                        dfs(next_v, curr, length - 1, path_edges, path_dirs)
                        path_edges.pop()
                        path_dirs.pop()
            
            for first in range(6):
                if first == start: continue
                if (start < 3 and first >= 3) or (start >= 3 and first < 3):
                    if start < 3:
                        idx = start * 3 + (first - 3)
                        dir = 1
                    else:
                        idx = first * 3 + (start - 3)
                        dir = -1
                    dfs(first, start, max_len - 1, [idx], [dir])
        return walks

    walks_4 = get_walks(4)
    walks_6 = get_walks(6)
    walks_8 = get_walks(8)
    
    all_walks = walks_4 + walks_6 + walks_8
    
    # Remove duplicates?
    # A walk and its reverse are different in terms of sum (signs flip), but if sum=0, -sum=0.
    # So we just need to check sum != 0.
    
    m = 41
    n_lift = 6 * m
    
    # Local Search / Simulated Annealing
    # Objective: minimize number of walks with sum == 0
    
    def count_violations(voltages):
        count = 0
        for edges, dirs in all_walks:
            s = 0
            for e, d in zip(edges, dirs):
                s += d * voltages[e]
            if s % m == 0:
                count += 1
        return count

    best_voltages = None
    best_violations = float('inf')
    
    # Try a few random starts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        voltages = [random.randint(1, m-1) for _ in range(9)]
        current_violations = count_violations(voltages)
        
        if current_violations == 0:
            best_voltages = voltages
            break
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_voltages = voltages[:]
            
        # Simulated Annealing
        T = 10.0
        for step in range(1000):
            if time.time() > deadline:
                break
                
            # Pick a random edge to change voltage
            e_idx = random.randint(0, 8)
            old_vol = voltages[e_idx]
            new_vol = random.randint(1, m-1)
            if new_vol == old_vol:
                continue
                
            voltages[e_idx] = new_vol
            new_violations = count_violations(voltages)
            
            delta = new_violations - current_violations
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_violations = new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_voltages = voltages[:]
                    if current_violations == 0:
                        break
            else:
                voltages[e_idx] = old_vol
                
            T *= 0.99 # Cooling
            
        if best_violations == 0:
            break

    if best_voltages is not None and best_violations == 0:
        # Construct the graph
        edges = []
        for i in range(6):
            for j in range(6):
                if (i < 3 and j >= 3) or (i >= 3 and j < 3):
                    if i < 3:
                        idx = i * 3 + (j - 3)
                        vol = best_voltages[idx]
                    else:
                        idx = j * 3 + (i - 3)
                        vol = -best_voltages[idx]
                    for c in range(m):
                        u = i * m + c
                        v_id = j * m + (c + vol) % m
                        edges.append((u, v_id))
        
        # Deduplicate and sort
        unique_edges = set()
        for u, v in edges:
            if u < v:
                unique_edges.add((u, v))
            else:
                unique_edges.add((v, u))
        
        return n_lift, list(unique_edges)
    else:
        # Fallback
        return _luw_fallback(k, g)

def _luw_fallback(k, g):
    # Original LUW code
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

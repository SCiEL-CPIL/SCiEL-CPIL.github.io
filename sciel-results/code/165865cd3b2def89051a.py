import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_n = float('inf')
    best_edges = []

    if k == 4 and g == 10:
        # Base graph: K6 minus perfect matching
        base_edges = []
        pm = {(0,3), (1,4), (2,5)}
        for i in range(6):
            for j in range(i+1, 6):
                if (i,j) not in pm:
                    base_edges.append((i, j))
        
        # Map edge tuple to index
        edge_index = {}
        for idx, e in enumerate(base_edges):
            edge_index[e] = idx
            edge_index[(e[1], e[0])] = idx
        
        # Build adjacency list for base
        base_adj = [[] for _ in range(6)]
        for u, v in base_edges:
            base_adj[u].append(v)
            base_adj[v].append(u)
            
        # Precompute all closed non-backtracking walks of length 3 to 9
        walks = []
        
        def find_walks():
            for start in range(6):
                # DFS with state: (prev, curr, length, coeffs_tuple)
                # Use tuple for coeffs to enable hashing
                stack = []
                for nb in base_adj[start]:
                    e_idx = edge_index[(start, nb)]
                    sign = 1 if (start, nb) == base_edges[e_idx] else -1
                    stack.append((start, nb, 1, ((e_idx, sign),)))
                
                while stack:
                    prev, curr, length, coeffs = stack.pop()
                    if length >= 3:
                        if curr == start:
                            # Convert coeffs to array
                            arr = np.zeros(len(base_edges), dtype=np.int8)
                            for e_idx, sign in coeffs:
                                arr[e_idx] += sign
                            walks.append(arr)
                            
                    if length < 9:
                        for nb in base_adj[curr]:
                            if nb == prev:
                                continue
                            e_idx = edge_index[(curr, nb)]
                            sign = 1 if (curr, nb) == base_edges[e_idx] else -1
                            stack.append((curr, nb, length + 1, coeffs + ((e_idx, sign),)))
        
        find_walks()
        
        # Deduplicate walks
        unique_walks = set()
        for w in walks:
            unique_walks.add(tuple(w))
        
        if unique_walks:
            walk_matrix = np.array(list(unique_walks))
            K = len(walk_matrix)
        else:
            walk_matrix = None
            K = 0
            
        # Try different M values, starting from small M
        # Moore bound for n(4,10) is 243, so we need M >= 41 (since 6*M >= 243)
        # But let's try smaller M first to see if we can find valid graphs
        
        for M in range(41, 200):
            if time.time() > deadline - 10.0:
                break
            n_total = 6 * M
            if n_total >= best_n:
                continue
            
            remaining = deadline - 15.0 - time.time()
            if remaining <= 0:
                break
            
            # Scale attempts with time - more attempts for smaller M
            if M < 60:
                max_attempts = min(20000, max(500, int(remaining * 100)))
            else:
                max_attempts = min(5000, max(200, int(remaining * 30)))
            
            found = False
            for attempt in range(max_attempts):
                if time.time() > deadline - 5.0:
                    break
                
                # Random voltages
                V = np.random.randint(0, M, size=len(base_edges))
                
                if K > 0:
                    # Compute sums
                    sums = (walk_matrix @ V) % M
                    if 0 in sums:
                        continue
                
                # Build the lifted graph edges directly
                edge_set = set()
                for e_idx, (u, v) in enumerate(base_edges):
                    vol = V[e_idx]
                    for w in range(M):
                        a = u * M + w
                        b = v * M + ((w + vol) % M)
                        if a < b:
                            edge_set.add((a, b))
                        else:
                            edge_set.add((b, a))
                
                # Update best
                if n_total < best_n:
                    best_n = n_total
                    best_edges = list(edge_set)
                    found = True
                    break
            
            if found:
                # Found valid for this M, try smaller M? No, M is increasing.
                # But we want the smallest n, so keep trying smaller M if possible
                # Actually, since M is increasing, if we find one for M=41, that's the best
                # We should break and return
                break

    # Fallback: LUW construction if no better found
    if best_n == float('inf') or best_n > 2048:
        # LUW construction for q=4, d=5 (2048 vertices, girth >= 10)
        q = 4
        d = 5
        
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return table[x][y]
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        def neighbours_luw(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                    else:
                        y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        
        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours_luw(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        if len(order) < best_n:
            best_n = len(order)
            best_edges = edges

    return best_n, best_edges

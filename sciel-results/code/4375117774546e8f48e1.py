import numpy as np
import random
import time

def construct(k, g):
    # Construct PG(2,3) incidence graph
    # Points: 13, Lines: 13
    # Total 26 vertices.
    
    # Generate points
    points = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if x == 0 and y == 0 and z == 0: continue
                # Canonicalize: first non-zero is 1
                if x != 0:
                    inv_x = 1 if x == 1 else 2
                    p = (1, (y*inv_x)%3, (z*inv_x)%3)
                elif y != 0:
                    inv_y = 1 if y == 1 else 2
                    p = (0, 1, (z*inv_y)%3)
                else:
                    p = (0, 0, 1)
                if p not in points:
                    points.append(p)
    
    # Generate lines
    lines = []
    line_points = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if a == 0 and b == 0 and c == 0: continue
                # Canonicalize normal
                if a != 0:
                    inv_a = 1 if a == 1 else 2
                    n = (1, (b*inv_a)%3, (c*inv_a)%3)
                elif b != 0:
                    inv_b = 1 if b == 1 else 2
                    n = (0, 1, (c*inv_b)%3)
                else:
                    n = (0, 0, 1)
                
                if n in [tuple(l) for l in lines]: continue
                lines.append(n)
                pts = []
                for p in points:
                    if (n[0]*p[0] + n[1]*p[1] + n[2]*p[2]) % 3 == 0:
                        pts.append(p)
                line_points.append(pts)
    
    # Build base graph
    base_edges = []
    for i, p in enumerate(points):
        for j, lp in enumerate(line_points):
            if p in lp:
                base_edges.append((i, 13+j))
    
    # Build adjacency for base graph
    adj = {i: [] for i in range(26)}
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Pre-compute 6-cycles and 8-cycles
    # A cycle is a tuple of vertices (v0, v1, ..., v_{L-1})
    # We only need to check unique cycles to avoid redundancy, but checking all is fine if fast enough.
    # To avoid checking the same cycle multiple times (reversed or rotated), we can normalize.
    # But for simplicity, we'll just generate all and check.
    # Actually, checking all permutations is wasteful.
    # Let's just generate simple cycles.
    
    cycles_6 = set()
    cycles_8 = set()
    
    # Find 6-cycles
    # Start from each vertex, do DFS up to depth 5, check if closes to start.
    # To avoid duplicates, enforce v0 < v1 < ... is hard for cycles.
    # Standard way: fix v0, then v1 > v0? No, graph is not ordered.
    # We can just collect them in a set of frozensets? No, order matters for voltage sum direction?
    # Voltage sum is directional.
    # However, if a cycle has sum 0 in one direction, it has sum 0 in the other (negated, which is 0 mod m).
    # So we just need to check one orientation.
    
    # Let's use a simple DFS for each start node.
    # To reduce redundancy, only start cycles from the smallest vertex in the cycle?
    # Or just check all and rely on the fact that if sum is 0, it's bad.
    
    # Given small size (26 nodes), we can afford to be a bit inefficient in pre-computation.
    
    def find_cycles(start, length, visited, path):
        if len(path) == length:
            if start in adj[path[-1]]:
                # Cycle found
                # Normalize: rotate so smallest vertex is first
                # And reverse if necessary to ensure a canonical form
                cyc = list(path)
                # Check if start is in path? Yes, path starts with start.
                # So cyc is [start, ..., last]
                # Close it: last -> start.
                # We store the sequence of vertices.
                # To avoid counting the same cycle 2*length times, we can canonicalize.
                # Find index of min vertex in cyc.
                min_idx = cyc.index(min(cyc))
                # Rotate
                rotated = cyc[min_idx:] + cyc[:min_idx]
                # Check reverse
                rev = rotated[::-1]
                # We want the lexicographically smaller of rotated and rev
                # But wait, voltage sum on rev is -sum(rotated).
                # If sum(rotated) != 0, then sum(rev) != 0.
                # So we only need to check one.
                # Let's pick the lexicographically smaller tuple.
                if tuple(rotated) < tuple(rev):
                    cycles_6.add(tuple(rotated)) if length == 6 else cycles_8.add(tuple(rotated))
                else:
                    cycles_6.add(tuple(rev)) if length == 6 else cycles_8.add(tuple(rev))
            return
        
        for nb in adj[path[-1]]:
            if nb in visited: continue
            if len(path) > 1 and nb == start and len(path) < length: continue
            visited.add(nb)
            path.append(nb)
            find_cycles(start, length, visited, path)
            path.pop()
            visited.remove(nb)
            
    # This DFS might be slow if not pruned.
    # 26 nodes, degree 4.
    # Depth 6: 4^5 = 1024 paths per start. 26 * 1024 = 26k paths. Fast.
    # Depth 8: 4^7 = 16k paths per start. 26 * 16k = 416k paths. Still fast.
    
    for s in range(26):
        visited = {s}
        path = [s]
        find_cycles(s, 6, visited, path)
        find_cycles(s, 8, visited, path)
        
    # Convert to lists for faster iteration
    list_cycles_6 = list(cycles_6)
    list_cycles_8 = list(cycles_8)
    
    m = 10
    n_vertices = 26 * m
    
    # Search for voltages
    best_voltages = None
    
    # Local search / Random restart
    max_iters = 5000
    for _ in range(max_iters):
        voltages = {}
        for u, v in base_edges:
            vu = random.randint(0, m-1)
            voltages[(u, v)] = vu
            voltages[(v, u)] = (-vu) % m
        
        valid = True
        
        # Check 6-cycles
        for cyc in list_cycles_6:
            s = 0
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i+1) % len(cyc)]
                s += voltages[(u, v)]
            if s % m == 0:
                valid = False
                break
        
        if not valid:
            continue
            
        # Check 8-cycles
        for cyc in list_cycles_8:
            s = 0
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i+1) % len(cyc)]
                s += voltages[(u, v)]
            if s % m == 0:
                valid = False
                break
                
        if valid:
            # Build lift graph
            lift_edges = []
            for u, v in base_edges:
                vu = voltages[(u, v)]
                for i in range(m):
                    lift_edges.append((u*m + i, v*m + (i + vu) % m))
            
            # Verify girth of lift graph
            adj_lift = {i: [] for i in range(n_vertices)}
            for u, v in lift_edges:
                adj_lift[u].append(v)
                adj_lift[v].append(u)
                
            min_girth = 999
            for start in range(n_vertices):
                # BFS
                dist = {start: 0}
                parent = {start: None}
                queue = [start]
                while queue:
                    cur = queue.pop(0)
                    for nb in adj_lift[cur]:
                        if nb not in dist:
                            dist[nb] = dist[cur] + 1
                            parent[nb] = cur
                            queue.append(nb)
                        elif parent[nb] != cur and parent[cur] != nb:
                            cycle_len = dist[nb] + dist[cur] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                                if min_girth < g:
                                    break
                if min_girth < g:
                    break
                    
            if min_girth >= g:
                return n_vertices, lift_edges
                
    # Fallback to LUW if search fails
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

import numpy as np
import random
import time

def construct(k, g):
    # Construct PG(2,3) incidence graph
    points = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if x == 0 and y == 0 and z == 0: continue
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
    
    lines = []
    line_points = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if a == 0 and b == 0 and c == 0: continue
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
    
    base_edges = []
    adj = {i: [] for i in range(26)}
    for i, p in enumerate(points):
        for j, lp in enumerate(line_points):
            if p in lp:
                base_edges.append((i, 13+j))
                adj[i].append(13+j)
                adj[13+j].append(i)
                
    m = 10
    n_edges = len(base_edges)
    edge_index = {}
    for idx, (u, v) in enumerate(base_edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx

    cycles = []
    edge_to_cycles = [[] for _ in range(n_edges)]
    
    # Find 6-cycles
    for v0 in range(13):
        for v1 in adj[v0]:
            for v2 in adj[v1]:
                if v2 == v0: continue
                for v3 in adj[v2]:
                    if v3 == v0 or v3 == v1: continue
                    for v4 in adj[v3]:
                        if v4 in (v0, v1, v2): continue
                        for v5 in adj[v4]:
                            if v5 in (v0, v1, v2, v3): continue
                            if v0 in adj[v5]:
                                cycle_edges = []
                                verts = [v0, v1, v2, v3, v4, v5]
                                for i in range(6):
                                    u, v = verts[i], verts[(i+1)%6]
                                    idx = edge_index[(u, v)]
                                    sign = 1 if base_edges[idx][0] == u else -1
                                    cycle_edges.append((idx, sign))
                                c_idx = len(cycles)
                                cycles.append(cycle_edges)
                                for idx, sign in cycle_edges:
                                    edge_to_cycles[idx].append(c_idx)
                                    
    # Find 8-cycles
    for v0 in range(13):
        for v1 in adj[v0]:
            for v2 in adj[v1]:
                if v2 == v0: continue
                for v3 in adj[v2]:
                    if v3 in (v0, v1): continue
                    for v4 in adj[v3]:
                        if v4 in (v0, v1, v2): continue
                        for v5 in adj[v4]:
                            if v5 in (v0, v1, v2, v3): continue
                            for v6 in adj[v5]:
                                if v6 in (v0, v1, v2, v3, v4): continue
                                for v7 in adj[v6]:
                                    if v7 in (v0, v1, v2, v3, v4, v5): continue
                                    if v0 in adj[v7]:
                                        cycle_edges = []
                                        verts = [v0, v1, v2, v3, v4, v5, v6, v7]
                                        for i in range(8):
                                            u, v = verts[i], verts[(i+1)%8]
                                            idx = edge_index[(u, v)]
                                            sign = 1 if base_edges[idx][0] == u else -1
                                            cycle_edges.append((idx, sign))
                                        c_idx = len(cycles)
                                        cycles.append(cycle_edges)
                                        for idx, sign in cycle_edges:
                                            edge_to_cycles[idx].append(c_idx)

    # Precompute cycle sums structure for fast update
    # cycle_sums[i] = sum of voltages in cycle i
    # We maintain an array `cycle_sums`
    
    best_volt = None
    best_bad = float('inf')
    
    for restart in range(50):
        voltages = [random.randint(0, m-1) for _ in range(n_edges)]
        cycle_sums = [0] * len(cycles)
        for i, c in enumerate(cycles):
            s = 0
            for idx, sign in c:
                s += sign * voltages[idx]
            cycle_sums[i] = s % m
            
        bad_count = sum(1 for s in cycle_sums if s == 0)
        
        if bad_count == 0:
            best_volt = voltages
            break
            
        T = 1.0
        for step in range(2000):
            idx = random.randint(0, n_edges - 1)
            old_val = voltages[idx]
            new_val = random.randint(0, m-1)
            if new_val == old_val: continue
            
            delta = (new_val - old_val)
            
            new_bad = bad_count
            for c_idx in edge_to_cycles[idx]:
                c = cycles[c_idx]
                # Find sign for this edge in this cycle
                # This is slow. Precompute signs per edge per cycle?
                # Or just recompute sum?
                # Given small number of cycles per edge, recompute is okay.
                s = 0
                for e_idx, sign in c:
                    if e_idx == idx:
                        s += sign * new_val
                    else:
                        s += sign * voltages[e_idx]
                old_s = cycle_sums[c_idx]
                new_s = s % m
                if old_s == 0 and new_s != 0: new_bad -= 1
                elif old_s != 0 and new_s == 0: new_bad += 1
                cycle_sums[c_idx] = new_s
                
            if new_bad < bad_count or (random.random() < np.exp((bad_count - new_bad) / T)):
                voltages[idx] = new_val
                bad_count = new_bad
                if bad_count == 0:
                    best_volt = voltages
                    break
            else:
                # Revert cycle_sums
                for c_idx in edge_to_cycles[idx]:
                    c = cycles[c_idx]
                    s = 0
                    for e_idx, sign in c:
                        if e_idx == idx:
                            s += sign * old_val
                        else:
                            s += sign * voltages[e_idx]
                    cycle_sums[c_idx] = s % m
                
            T *= 0.999
            if T < 0.01: break
            
        if best_volt is not None:
            break
            
    if best_volt is not None:
        n_vertices = 26 * m
        lift_edges = []
        for u, v in base_edges:
            vu = best_volt[edge_index[(u, v)]]
            for i in range(m):
                lift_edges.append((u*m + i, v*m + (i + vu) % m))
                
        adj_lift = {i: [] for i in range(n_vertices)}
        for u, v in lift_edges:
            adj_lift[u].append(v)
            adj_lift[v].append(u)
            
        min_girth = 999
        for start in range(n_vertices):
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
                            
        if min_girth >= g:
            return n_vertices, lift_edges

    # Fallback
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

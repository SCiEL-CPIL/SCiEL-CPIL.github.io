import random
import time
import math

def construct(k, g):
    # Target: 3-regular, girth >= g (16)
    
    if k != 3:
        # Fallback for other k
        # Use LUW
        return _luw_fallback(k, g)

    # Base Graph 1: Heawood (14 vertices, girth 6)
    # Base Graph 2: Petersen (10 vertices, girth 5)
    
    # Heawood Graph Edges
    # Vertices 0..13
    # Bipartition: 0..6 and 7..13
    # 0-7, 1-8, 2-9, 3-10, 4-11, 5-12, 6-13? 
    # And cycles?
    # Standard Heawood:
    # Outer: 0-1-2-3-4-5-0
    # Inner: 6-7-8-9-10-11-6
    # Spokes: 0-6, 1-7, 2-8, 3-9, 4-10, 5-11
    # Wait, this is the prism graph $C_6 \times K_2$? No, that's 12 vertices.
    # Heawood has 14 vertices.
    
    # Let's use a known edge list for Heawood.
    # One representation:
    # 0: 1, 6, 7
    # 1: 0, 2, 8
    # 2: 1, 3, 9
    # 3: 2, 4, 10
    # 4: 3, 5, 11
    # 5: 4, 0, 12  (Wait, 0-5? No, 0-1-2-3-4-5-0 is a 6-cycle)
    # Let's check degrees.
    # If 0 is connected to 1, 5, and 6?
    
    # Correct Heawood construction:
    # It is the incidence graph of the Fano plane.
    # Points: 0,1,2,3,4,5,6
    # Lines: 7,8,9,10,11,12,13
    # Incidence:
    # 0 is on lines 7, 8, 9?
    # Fano plane lines:
    # L1: 0,1,2
    # L2: 0,3,4
    # L3: 0,5,6
    # L4: 1,3,5
    # L5: 1,4,6
    # L6: 2,3,6
    # L7: 2,4,5
    # Map lines to vertices 7..13.
    # 7 -> L1 (0,1,2)
    # 8 -> L2 (0,3,4)
    # 9 -> L3 (0,5,6)
    # 10 -> L4 (1,3,5)
    # 11 -> L5 (1,4,6)
    # 12 -> L6 (2,3,6)
    # 13 -> L7 (2,4,5)
    
    heawood_edges = []
    # Point 0 is on L1, L2, L3 -> 7, 8, 9
    heawood_edges.extend([(0,7), (0,8), (0,9)])
    # Point 1 is on L1, L4, L5 -> 7, 10, 11
    heawood_edges.extend([(1,7), (1,10), (1,11)])
    # Point 2 is on L1, L6, L7 -> 7, 12, 13
    heawood_edges.extend([(2,7), (2,12), (2,13)])
    # Point 3 is on L2, L4, L6 -> 8, 10, 12
    heawood_edges.extend([(3,8), (3,10), (3,12)])
    # Point 4 is on L2, L5, L7 -> 8, 11, 13
    heawood_edges.extend([(4,8), (4,11), (4,13)])
    # Point 5 is on L3, L4, L7 -> 9, 10, 13
    heawood_edges.extend([(5,9), (5,10), (5,13)])
    # Point 6 is on L3, L5, L6 -> 9, 11, 12
    heawood_edges.extend([(6,9), (6,11), (6,12)])
    
    n_base_h = 14
    adj_h = [[] for _ in range(n_base_h)]
    for i, (u, v) in enumerate(heawood_edges):
        adj_h[u].append((v, i))
        adj_h[v].append((u, i))

    # Generate NB walks of length < g in Heawood
    walks_h = _get_nb_walks(heawood_edges, adj_h, n_base_h, g)
    walk_coeffs_h = _get_walk_coeffs(walks_h, len(heawood_edges))

    # Base Graph 2: Petersen
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base_p = 10
    adj_p = [[] for _ in range(n_base_p)]
    for i, (u, v) in enumerate(petersen_edges):
        adj_p[u].append((v, i))
        adj_p[v].append((u, i))
        
    walks_p = _get_nb_walks(petersen_edges, adj_p, n_base_p, g)
    walk_coeffs_p = _get_walk_coeffs(walks_p, len(petersen_edges))

    start_time = time.time()
    deadline = start_time + 140.0

    best_n = float('inf')
    best_edges = None

    # Try Heawood first
    # Moore bound 512. n = 14m.
    # m >= ceil(512/14) = 37.
    for m in range(37, 60):
        if time.time() > deadline - 10.0:
            break
        candidate_n = 14 * m
        if candidate_n >= best_n:
            continue
            
        # Try random voltages
        found = False
        # More tries for smaller m?
        tries = 5000
        for _ in range(tries):
            if time.time() > deadline - 5.0:
                break
            volts = [random.randint(1, m-1) for _ in range(len(heawood_edges))]
            # Check connectivity: gcd(volts, m) == 1
            g_val = m
            for v in volts:
                g_val = math.gcd(g_val, v)
            if g_val != 1:
                continue
                
            valid = True
            for coeffs in walk_coeffs_h:
                s = 0
                for i, c in enumerate(coeffs):
                    if c != 0:
                        s += c * volts[i]
                if s % m == 0:
                    valid = False
                    break
            
            if valid:
                found = True
                break
        
        if found:
            best_n = candidate_n
            best_edges = _construct_lift(heawood_edges, volts, m)
            # We found a valid Heawood lift.
            # Can we do better?
            # Next m is 38 -> 532.
            # If we found m=37, n=518.
            # We should break if we found the minimum possible m for this base.
            # Since m starts at 37, if we found it, it's the best for Heawood.
            break

    # If Heawood failed or didn't find a small enough m, try Petersen
    if best_n > 520 or best_edges is None:
        for m in range(52, 100):
            if time.time() > deadline - 5.0:
                break
            candidate_n = 10 * m
            if candidate_n >= best_n:
                continue
                
            found = False
            tries = 5000
            for _ in range(tries):
                if time.time() > deadline - 2.0:
                    break
                volts = [random.randint(1, m-1) for _ in range(len(petersen_edges))]
                g_val = m
                for v in volts:
                    g_val = math.gcd(g_val, v)
                if g_val != 1:
                    continue
                    
                valid = True
                for coeffs in walk_coeffs_p:
                    s = 0
                    for i, c in enumerate(coeffs):
                        if c != 0:
                            s += c * volts[i]
                    if s % m == 0:
                        valid = False
                        break
                
                if valid:
                    found = True
                    break
            
            if found:
                if candidate_n < best_n:
                    best_n = candidate_n
                    best_edges = _construct_lift(petersen_edges, volts, m)
                break

    if best_edges is None:
        return _luw_fallback(k, g)

    return best_n, best_edges

def _get_nb_walks(edges, adj, n, g):
    walks = []
    for start_v in range(n):
        queue = []
        for next_v, e_id in adj[start_v]:
            u, v = edges[e_id]
            sign = 1 if u == start_v else -1
            queue.append((next_v, e_id, 1, [(e_id, sign)]))
            
        while queue:
            curr_v, prev_e, length, path = queue.pop(0)
            if length >= g:
                continue
            if curr_v == start_v and length >= 3:
                walks.append(path)
            for next_v, e_id in adj[curr_v]:
                if e_id == prev_e:
                    continue
                u, v = edges[e_id]
                sign = 1 if u == curr_v else -1
                new_path = path + [(e_id, sign)]
                queue.append((next_v, e_id, length + 1, new_path))
                
    unique_walks = set()
    for w in walks:
        unique_walks.add(tuple(w))
    return list(unique_walks)

def _get_walk_coeffs(walks, num_edges):
    coeffs_list = []
    for w in walks:
        coeffs = [0] * num_edges
        for e_id, sign in w:
            coeffs[e_id] += sign
        coeffs_list.append(coeffs)
    return coeffs_list

def _construct_lift(edges, volts, m):
    lift_edges = []
    for i, (u, v) in enumerate(edges):
        vol = volts[i]
        for t in range(m):
            u_lift = u * m + t
            v_lift = v * m + (t + vol) % m
            lift_edges.append((u_lift, v_lift))
    return lift_edges

def _luw_fallback(k, g):
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    return len(order), edges_luw

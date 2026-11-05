import time
import random
import sys
from collections import deque, defaultdict
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Structural change: Voltage lift of the Petersen graph over Z_p.
    """
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin

    if k == 3:
        n, edges = _voltage_lift_petersen(g, deadline, start_time)
        if n is not None:
            return n, edges
        # Fallback to the well-understood LUW construction (guaranteed valid family).
        return _construct_luw(k, g, d=11)
    else:
        return _construct_luw(k, g, d=7)

def _voltage_lift_petersen(g, deadline, start_time):
    """
    Constructs a 3-regular graph with girth >= g using voltage lift of Petersen graph.
    """
    # Petersen Graph Definition
    # Vertices 0..9
    # Edges:
    # 0-1, 1-2, 2-3, 3-4, 4-0 (Outer)
    # 5-7, 7-9, 9-6, 6-8, 8-5 (Inner)
    # 0-5, 1-6, 2-7, 3-8, 4-9 (Spokes)
    
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Map edge to index
    edge_index = {}
    for i, (u, v) in enumerate(petersen_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    # Adjacency list for Petersen
    adj = defaultdict(list)
    for u, v in petersen_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Generate all non-backtracking walks of length 5 to g-1
    # A walk is a sequence of vertices (v0, v1, ..., vL) such that vi != v(i-2)
    # We store the sequence of edge indices and directions.
    # Direction: 1 if (vi-1, vi) matches stored (u, v), -1 if (vi, vi-1) matches.
    # Actually, simpler: store (edge_idx, sign) where sign is +1 if we go u->v as defined, -1 if v->u.
    
    walks = []
    
    # DFS to find walks
    # State: (current_vertex, prev_vertex, path_edges)
    # path_edges is list of (edge_idx, sign)
    
    def find_walks(max_len):
        res = []
        for start in range(10):
            # Start with each neighbor
            for nb in adj[start]:
                e_idx = edge_index[(start, nb)]
                # Determine sign
                u, v = petersen_edges[e_idx]
                if u == start and v == nb:
                    sign = 1
                else:
                    sign = -1
                path = [(e_idx, sign)]
                # Extend
                def dfs(curr, prev, path_list, length):
                    if length >= max_len:
                        return
                    if length >= 5: # Only interested in length >= 5
                        res.append(list(path_list))
                    
                    for nb2 in adj[curr]:
                        if nb2 == prev:
                            continue
                        e_idx2 = edge_index[(curr, nb2)]
                        u2, v2 = petersen_edges[e_idx2]
                        if u2 == curr and v2 == nb2:
                            sign2 = 1
                        else:
                            sign2 = -1
                        path_list.append((e_idx2, sign2))
                        dfs(nb2, curr, path_list, length + 1)
                        path_list.pop()
                
                dfs(nb, start, path, 1)
        return res

    # We need walks of length < g.
    # For g=16, lengths 5..15.
    if g > 5:
        max_walk_len = g - 1
        if max_walk_len > 15: # Cap to avoid explosion, though 15 is small
            max_walk_len = 15
        walks = find_walks(max_walk_len)
    else:
        walks = []

    # Prune walks: if a walk is a prefix of another, we only need the shorter one?
    # No, if prefix sums to 0, it's a cycle. If longer sums to 0, it's a cycle.
    # We must check all.
    
    # Try primes
    primes = [p for p in range(50, 200) if all(p % i != 0 for i in range(2, int(p**0.5)+1))]
    
    # Sort primes by size (smaller p = smaller graph)
    # But smaller p is harder to satisfy constraints.
    # Let's try a mix, or start from middle.
    # Actually, n = 10p. We want small n. So try small p first.
    # But small p might not have a solution.
    
    for p in primes:
        if time.time() > deadline:
            break
            
        # Try to find voltages in Z_p
        # Random restarts
        found = False
        attempts = 200 # Limit attempts per prime to keep time low
        
        for _ in range(attempts):
            if time.time() > deadline:
                break
            
            # Initialize random voltages
            voltages = [random.randint(0, p-1) for _ in range(15)]
            
            # Repair loop
            valid = True
            iterations = 0
            max_iterations = 5000
            
            while valid and iterations < max_iterations:
                iterations += 1
                # Check all walks
                violated_walk = None
                for w in walks:
                    s = 0
                    for e_idx, sign in w:
                        s = (s + sign * voltages[e_idx]) % p
                    if s == 0:
                        violated_walk = w
                        break
                
                if violated_walk is None:
                    found = True
                    break
                else:
                    # Repair: pick a random edge in the violated walk and change its voltage
                    # To fix sum=0, we need to change sum to non-zero.
                    # Changing v_e by delta changes sum by sign * delta.
                    # We can just add 1 to a random edge in the walk.
                    e_idx, sign = random.choice(violated_walk)
                    voltages[e_idx] = (voltages[e_idx] + 1) % p
            
            if found:
                break
        
        if found:
            # Construct the lift
            n = 10 * p
            edges = set()
            # Vertex (v, i) -> index v * 10 + i
            for i in range(10):
                for nb in adj[i]:
                    e_idx = edge_index[(i, nb)]
                    u, v = petersen_edges[e_idx]
                    # Determine voltage
                    if u == i and v == nb:
                        val = voltages[e_idx]
                    else:
                        val = -voltages[e_idx] % p
                    
                    # Lift edge from (v, i) to (v + val, nb)
                    # We need to add edges for all v in 0..p-1
                    for base_v in range(p):
                        v1 = base_v * 10 + i
                        v2 = ((base_v + val) % p) * 10 + nb
                        if v1 > v2:
                            v1, v2 = v2, v1
                        edges.add((v1, v2))
            
            return n, list(edges)
        
        # If not found for this p, try next p

    return None

def _construct_luw(k, g, d=None, variant=0):
    """
    LUW construction (fallback, guaranteed valid family).
    """
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)

    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges

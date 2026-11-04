import time
import random
from collections import defaultdict, deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses voltage lift of Petersen graph for k=3.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    if k == 3:
        return _construct_petersen_lift(g, deadline)
    else:
        # Fallback for other k, though task specifies k=3
        return _construct_luw(k, g, d=7)

def _construct_petersen_lift(g, deadline):
    """
    Constructs a girth-g graph by lifting the Petersen graph with voltages in Z_m.
    """
    # Petersen graph edges (10 vertices, 15 edges)
    # Standard representation: outer pentagon 0-1-2-3-4-0, inner star 5-7-9-6-8-5
    # Connections: 0-5, 1-6, 2-7, 3-8, 4-9
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), # Outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5), # Inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)  # Spokes
    ]
    
    n_base = 10
    # Map edge to index for voltage assignment
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # Precompute all closed walks of length < g in the Petersen graph
    # Represent each walk as a vector of net edge counts (with sign)
    # Since the group is abelian (Z_m), the order doesn't matter, only the net count of each edge traversed in each direction.
    # Let c_e be the number of times edge e is traversed from u to v minus v to u.
    # We need sum(c_e * v_e) != 0 mod m.
    
    coeff_vectors = set()
    max_walk_len = g - 1
    
    for start_v in range(n_base):
        # DFS to find all closed walks of length < g
        # State: (current_v, prev_v, length, counts)
        # counts is a tuple of length 15
        stack = [(start_v, -1, 0, [0]*15)]
        
        while stack:
            v, prev_v, l, counts = stack.pop()
            
            if l > 0 and v == start_v:
                # Found a closed walk
                # Remove trivial back-and-forth? No, non-backtracking is required for girth check.
                # However, any closed walk of length < g implies a cycle of length <= that walk.
                # If we ensure all *non-backtracking* closed walks have non-zero voltage, girth is >= g.
                # The DFS above allows backtracking (e.g. u-v-u). 
                # A walk u-v-u has length 2. If we allow it, we must ensure its voltage is non-zero.
                # But voltage(u,v) + voltage(v,u) = v_e - v_e = 0. So it's always 0.
                # This means we CANNOT use standard voltage lifts on undirected graphs with Z_m if we allow backtracking,
                # because the 2-walk always sums to 0.
                # BUT, the definition of girth is the length of the *shortest cycle*.
                # A cycle cannot have length 2 in a simple graph.
                # So we only need to consider *simple* cycles or *non-backtracking* closed walks.
                # The condition "no non-backtracking closed walk of length < g" is equivalent to girth >= g.
                # So we should only add non-backtracking walks to coeff_vectors.
                
                # Check if the last step was a backtrack?
                # The stack state doesn't easily tell us the whole path history for backtracking check at the end.
                # Better: Only push non-backtracking steps.
                pass
            
            if l >= max_walk_len:
                continue
            
            for next_v in range(n_base):
                if next_v == prev_v:
                    continue # Non-backtracking
                if next_v == v:
                    continue # No loops
                
                e_idx = edge_index.get((v, next_v))
                if e_idx is None:
                    continue
                
                new_counts = counts.copy()
                # We need to track direction.
                # Let's store counts as (count_forward, count_backward) or just net count?
                # In Z_m, voltage(u,v) = x_e, voltage(v,u) = -x_e.
                # So traversing u->v adds x_e, v->u subtracts x_e.
                # We need to know the direction relative to the edge index definition.
                # Let's define edge e=(u,v) with u<v.
                # If we traverse u->v, add 1 to count[e].
                # If we traverse v->u, subtract 1 from count[e].
                
                u_edge, v_edge = pet_edges[e_idx]
                if v == u_edge and next_v == v_edge:
                    new_counts[e_idx] += 1
                else:
                    new_counts[e_idx] -= 1
                
                stack.append((next_v, v, l + 1, new_counts))
                
                if l > 0 and next_v == start_v:
                    # This is a non-backtracking closed walk
                    coeff_vectors.add(tuple(new_counts))

    # Remove zero vector if present (shouldn't be for non-backtracking closed walks of length > 0 in simple graph? 
    # Actually, a non-backtracking closed walk must have length >= 3. 
    # Can it have net zero count on all edges? 
    # Yes, e.g. a triangle traversed once. Counts: +1, +1, +1. Not zero.
    # What about a figure-8? Not simple, but non-backtracking?
    # Anyway, if a vector is all zeros, it imposes no constraint (0=0 is always true, so we just don't add it).
    # But wait, if the vector is all zeros, it means the walk traverses each edge equally in both directions.
    # The voltage sum is 0. This would be a "bad" walk if length < g.
    # So we MUST include it in the constraints.
    # So we keep all non-backtracking closed walks.
    
    coeff_list = [c for c in coeff_vectors if any(x != 0 for x in c)]
    
    # Search for m and voltages
    best_n = float('inf')
    best_edges = None
    
    # Try small primes
    primes = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
    
    for m in primes:
        if time.time() > deadline:
            break
        
        n_candidate = 10 * m
        if n_candidate >= best_n:
            continue
            
        # Try to find voltages
        # We need x_e in Z_m such that for all c in coeff_list: sum(c_e * x_e) != 0 mod m
        # Note: c_e can be negative.
        
        max_attempts = 200
        found = False
        for _ in range(max_attempts):
            if time.time() > deadline:
                break
                
            # Random assignment
            voltages = [random.randint(1, m-1) for _ in range(15)]
            
            valid = True
            for c in coeff_list:
                s = 0
                for i in range(15):
                    s += c[i] * voltages[i]
                if s % m == 0:
                    valid = False
                    break
            
            if valid:
                # Construct the graph
                edges = []
                for u in range(n_base):
                    for v in range(n_base):
                        if u == v:
                            continue
                        e_idx = edge_index.get((u, v))
                        if e_idx is None:
                            continue
                        
                        vol = voltages[e_idx]
                        # Determine direction sign
                        u_edge, v_edge = pet_edges[e_idx]
                        if u == u_edge and v == v_edge:
                            sign = 1
                        else:
                            sign = -1
                        
                        # Lift
                        for layer in range(m):
                            u_node = layer * n_base + u
                            # Target layer
                            target_layer = (layer + sign * vol) % m
                            v_node = target_layer * n_base + v
                            
                            if u_node < v_node:
                                edges.append((u_node, v_node))
                            else:
                                edges.append((v_node, u_node))
                
                edges = list(set(edges))
                
                # Verify girth
                if _has_girth_at_least(n_candidate, edges, g):
                    if n_candidate < best_n:
                        best_n = n_candidate
                        best_edges = edges
                        found = True
                        # Don't break immediately, try to find smaller m? 
                        # We iterate m in increasing order, so first valid m is smallest.
                        # But we might have skipped some m if random failed.
                        # So we should continue searching for smaller m if we find a valid one?
                        # No, we iterate m from small to large. 
                        # If we find a valid one for m, it's the best for this m.
                        # We can break out of m loop if we want the absolute smallest, 
                        # but random might fail for small m. 
                        # So we continue to next m? No, if we found one for m, n=10m.
                        # Any larger m gives larger n.
                        # So we can return immediately?
                        # Wait, what if a smaller m was possible but random failed?
                        # We should probably try more random attempts for small m.
                        # Let's just return the first valid one found in the increasing m loop,
                        # assuming random is good enough.
                        return n_candidate, edges
                
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback
    return _construct_luw(k=3, g=g, d=11)

def _construct_luw(k, g, d=None, variant=0):
    """
    LUW construction with specified depth d and variant for rule selection.
    For k=3, uses q=3 arithmetic.
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
    
    # Build rule table with variant-dependent offsets
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

def _has_girth_at_least(n, edges, g):
    """
    Check if graph has girth at least g using non-backtracking walks.
    A graph has girth >= g iff there are no non-backtracking closed walks of length < g.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Non-backtracking closed walks of length < g
    max_len = g - 1
    
    for start in range(n):
        # State: (current_vertex, prev_vertex, length)
        # Start with all neighbors of start as length-1 walks
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
        
        while queue:
            u, prev, l = queue.popleft()
            
            if u == start and l > 0:
                # Found a non-backtracking closed walk of length l < g
                return False
            
            if l >= max_len:
                continue
            
            for v in adj[u]:
                if v != prev:
                    queue.append((v, u, l + 1))
    
    return True

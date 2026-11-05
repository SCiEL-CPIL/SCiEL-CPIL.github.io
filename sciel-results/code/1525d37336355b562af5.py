import random
import time
import math
from collections import deque

def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)
    if g != 16:
        return _luw_fallback(k, g)

    # Heawood graph (14 vertices, girth 6) - incidence graph of Fano plane
    heawood_edges = [
        (0,7),(0,8),(0,9),
        (1,7),(1,10),(1,11),
        (2,7),(2,12),(2,13),
        (3,8),(3,10),(3,12),
        (4,8),(4,11),(4,13),
        (5,9),(5,10),(5,13),
        (6,9),(6,11),(6,12),
    ]
    n_base = 14
    num_edges = len(heawood_edges)
    
    adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(heawood_edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Enumerate all non-backtracking walks of length < g
    # Represent each walk as a coefficient vector over edges
    walk_coeffs = _get_nb_walk_coeffs(heawood_edges, adj, n_base, g)
    num_walks = len(walk_coeffs)
    
    start_time = time.time()
    deadline = start_time + 130.0
    
    best_n = float('inf')
    best_edges = None
    
    # Moore bound for (3,16): 2 + 3*(1+2+4+8+16+32+64) = 2+3*127 = 383
    # Actually: 2 + 3*(2^0 + 2^1 + ... + 2^7) = 2 + 3*255 = 767? 
    # Moore bound for girth g (even): 2*(k-1)^{g/2 - 1} + 2*(k-1)^{g/2 - 2} ... 
    # For k=3, g=16: n >= 2 + 3*(1+2+4+8+16+32+64) = 2 + 3*127 = 383
    # Wait, that's for girth 14. For girth 16:
    # n >= 2 + 3*(1+2+4+8+16+32+64) = 383? No...
    # Moore bound: n >= 2 + k*((k-1)^{g/2-1} - 1)/(k-2) for even g
    # For k=3, g=16: n >= 2 + 3*(2^7 - 1)/1 = 2 + 3*127 = 383
    # Actually the correct Moore bound for (3,16) is 2+3*(2^7-1) = 383
    # But the problem says best lower bound is 512. Let me just use 512.
    
    # n = 14*m, so m >= ceil(512/14) = 37
    for m in range(37, 80):
        if time.time() > deadline - 15.0:
            break
        candidate_n = 14 * m
        if candidate_n >= best_n:
            continue
        
        # Try to find voltage assignment
        # Use randomized search with constraint propagation
        found, volts = _find_voltage(heawood_edges, walk_coeffs, num_edges, m, deadline)
        
        if found:
            best_n = candidate_n
            best_edges = _construct_lift(heawood_edges, volts, m)
            # If we found m=37, that's optimal for Heawood base
            if m == 37:
                break
    
    # If Heawood didn't give a good result, try Petersen
    if best_n > 550 or best_edges is None:
        petersen_edges = [
            (0,1),(1,2),(2,3),(3,4),(4,0),
            (5,7),(7,9),(9,6),(6,8),(8,5),
            (0,5),(1,6),(2,7),(3,8),(4,9)
        ]
        n_base_p = 10
        num_edges_p = len(petersen_edges)
        adj_p = [[] for _ in range(n_base_p)]
        for i, (u, v) in enumerate(petersen_edges):
            adj_p[u].append((v, i))
            adj_p[v].append((u, i))
        walk_coeffs_p = _get_nb_walk_coeffs(petersen_edges, adj_p, n_base_p, g)
        
        for m in range(52, 120):
            if time.time() > deadline - 5.0:
                break
            candidate_n = 10 * m
            if candidate_n >= best_n:
                continue
            found, volts = _find_voltage(petersen_edges, walk_coeffs_p, num_edges_p, m, deadline)
            if found:
                if candidate_n < best_n:
                    best_n = candidate_n
                    best_edges = _construct_lift(petersen_edges, volts, m)
                break
    
    if best_edges is None:
        return _luw_fallback(k, g)
    
    return best_n, best_edges


def _get_nb_walk_coeffs(edges, adj, n, g):
    """Get coefficient vectors for all non-backtracking walks of length < g."""
    coeffs_set = set()
    
    for start_v in range(n):
        # BFS/DFS for NB walks
        # State: (current_vertex, prev_edge, length, coeff_vector)
        # Use iterative DFS to avoid recursion depth issues
        
        stack = []
        for next_v, e_id in adj[start_v]:
            u, v = edges[e_id]
            sign = 1 if u == start_v else -1
            coeff = [0] * len(edges)
            coeff[e_id] = sign
            stack.append((next_v, e_id, 1, tuple(coeff)))
        
        while stack:
            curr_v, prev_e, length, coeff_tuple = stack.pop()
            
            if length >= g:
                continue
            
            # Check if this is a cycle (returned to start)
            if curr_v == start_v and length >= 3:
                coeffs_set.add(coeff_tuple)
            
            for next_v, e_id in adj[curr_v]:
                if e_id == prev_e:
                    continue
                u, v = edges[e_id]
                sign = 1 if u == curr_v else -1
                new_coeff = list(coeff_tuple)
                new_coeff[e_id] += sign
                stack.append((next_v, e_id, length + 1, tuple(new_coeff)))
    
    return [list(c) for c in coeffs_set]


def _find_voltage(edges, walk_coeffs, num_edges, m, deadline):
    """Find a voltage assignment that avoids all short cycles."""
    start_time = time.time()
    max_time = min(deadline - start_time - 2.0, 30.0)
    
    # Precompute: for each walk, which edges have nonzero coefficients
    walk_info = []
    for coeffs in walk_coeffs:
        nonzero = [(i, c) for i, c in enumerate(coeffs) if c != 0]
        if nonzero:
            walk_info.append(nonzero)
    
    # Randomized search
    best_volts = None
    
    # Strategy: try many random assignments, but use smarter initialization
    # Start with a base assignment and perturb
    
    num_tries = 50000
    for attempt in range(num_tries):
        if time.time() > start_time + max_time:
            break
        
        # Generate random voltages
        volts = [random.randint(1, m-1) for _ in range(num_edges)]
        
        # Quick connectivity check
        g_val = m
        for v in volts:
            g_val = math.gcd(g_val, v)
        if g_val != 1:
            continue
        
        # Check all walks
        valid = True
        for nonzero in walk_info:
            s = 0
            for i, c in nonzero:
                s += c * volts[i]
            if s % m == 0:
                valid = False
                break
        
        if valid:
            best_volts = volts
            break
    
    if best_volts is not None:
        return True, best_volts
    
    # If pure random failed, try more structured approach
    # Use local search: start from a random assignment and fix conflicts
    
    for attempt in range(100):
        if time.time() > start_time + max_time:
            break
        
        volts = [random.randint(1, m-1) for _ in range(num_edges)]
        
        for iteration in range(200):
            if time.time() > start_time + max_time:
                break
            
            # Find a violated constraint
            violated = -1
            for w_idx, nonzero in enumerate(walk_info):
                s = 0
                for i, c in nonzero:
                    s += c * volts[i]
                if s % m == 0:
                    violated = w_idx
                    break
            
            if violated == -1:
                return True, volts
            
            # Try to fix by changing one of the edges in the violated walk
            nonzero = walk_info[violated]
            # Pick a random edge in this walk and change its voltage
            idx, c = nonzero[random.randint(0, len(nonzero)-1)]
            old_val = volts[idx]
            new_val = random.randint(1, m-1)
            if new_val == old_val:
                new_val = (new_val + 1) % m
                if new_val == 0:
                    new_val = 1
            volts[idx] = new_val
    
    return False, None


def _construct_lift(edges, volts, m):
    """Construct the lifted graph from base graph and voltage assignment."""
    lift_edges = []
    for i, (u, v) in enumerate(edges):
        vol = volts[i]
        for t in range(m):
            u_lift = u * m + t
            v_lift = v * m + (t + vol) % m
            lift_edges.append((u_lift, v_lift))
    return lift_edges


def _luw_fallback(k, g):
    """LUW construction as fallback."""
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

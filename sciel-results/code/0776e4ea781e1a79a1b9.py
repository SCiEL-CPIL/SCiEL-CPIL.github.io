import random
import time
import math
import sys

def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)
    
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s kill
    
    # Petersen Graph as base
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer cycle
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
    ]
    n_base = 10
    adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Generate all closed non-backtracking walks of length 3 to g-1 from every vertex
    walks = []
    for start_v in range(n_base):
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
    walks = list(unique_walks)
    
    # Pre-compute coefficient vectors for each walk (sparse)
    walk_coeffs = []
    for w in walks:
        coeffs = [0] * len(edges)
        for e_id, sign in w:
            coeffs[e_id] += sign
        nonzero = [(i, c) for i, c in enumerate(coeffs) if c != 0]
        if nonzero:
            walk_coeffs.append(nonzero)
    
    n_walks = len(walk_coeffs)
    
    # Moore bound for n(3,16) is 512, so n = 10*m >= 512 means m >= 52
    best_n = float('inf')
    best_edges = None
    best_m = None
    
    # Check if a voltage assignment is valid
    def check_volts(volts, m):
        for nonzero in walk_coeffs:
            s = 0
            for idx, c in nonzero:
                s += c * volts[idx]
            if s % m == 0:
                return False
        return True
    
    # Try to find valid voltages for a given m using multiple strategies
    def find_volts(m, time_left):
        if time_left <= 5:
            return None
        
        n_edges = len(edges)
        
        # Strategy 1: Structured patterns (fast, deterministic)
        # Try arithmetic progressions with different multipliers
        for mult in range(1, min(m, 20)):
            if time.time() > start_time + deadline - 5:
                return None
            volts = [(i + 1) * mult % m for i in range(n_edges)]
            # Ensure no zero
            for i in range(n_edges):
                if volts[i] == 0:
                    volts[i] = 1
            g_val = m
            for v in volts:
                g_val = math.gcd(g_val, v)
                if g_val == 1:
                    break
            if g_val != 1:
                continue
            if check_volts(volts, m):
                return volts
        
        # Strategy 2: Quadratic residue patterns
        for offset in range(min(m, 10)):
            if time.time() > start_time + deadline - 5:
                return None
            volts = []
            for i in range(n_edges):
                val = ((i + offset) * (i + offset + 1)) % m
                if val == 0:
                    val = 1
                volts.append(val)
            g_val = m
            for v in volts:
                g_val = math.gcd(g_val, v)
                if g_val == 1:
                    break
            if g_val != 1:
                continue
            if check_volts(volts, m):
                return volts
        
        # Strategy 3: Greedy construction with backtracking
        # Assign voltages one edge at a time, checking constraints incrementally
        # For speed, we only check constraints that involve the newly assigned edge
        
        # Pre-compute for each edge which walks involve it
        edge_to_walks = [[] for _ in range(n_edges)]
        for w_idx, nonzero in enumerate(walk_coeffs):
            for idx, c in nonzero:
                edge_to_walks[idx].append((w_idx, c))
        
        # Sort edges by number of constraints (most constrained first)
        edge_order = sorted(range(n_edges), key=lambda e: len(edge_to_walks[e]), reverse=True)
        
        max_greedy_tries = min(500, max(50, int(time_left / 0.001)))
        
        for trial in range(max_greedy_tries):
            if time.time() > start_time + deadline - 3:
                return None
            
            volts = [None] * n_edges
            valid = True
            
            for pos, e_idx in enumerate(edge_order):
                # Try random values for this edge
                found_val = False
                for _ in range(20):
                    val = random.randint(1, m - 1)
                    volts[e_idx] = val
                    
                    # Check all walks that involve this edge
                    ok = True
                    for w_idx, c in edge_to_walks[e_idx]:
                        nonzero = walk_coeffs[w_idx]
                        s = 0
                        for idx, coeff in nonzero:
                            if volts[idx] is None:
                                # Can't check yet if other edges not assigned
                                ok = False
                                break
                            s += coeff * volts[idx]
                        if ok and s % m == 0:
                            ok = False
                            break
                    
                    if ok:
                        found_val = True
                        break
                
                if not found_val:
                    valid = False
                    break
            
            if valid:
                # Final verification
                if check_volts(volts, m):
                    return volts
        
        # Strategy 4: Pure random with more tries
        max_random_tries = min(1000, max(100, int(time_left / 0.0005)))
        
        for trial in range(max_random_tries):
            if time.time() > start_time + deadline - 2:
                return None
            
            volts = [random.randint(1, m - 1) for _ in range(n_edges)]
            
            g_val = m
            for v in volts:
                g_val = math.gcd(g_val, v)
                if g_val == 1:
                    break
            if g_val != 1:
                continue
            
            if check_volts(volts, m):
                return volts
        
        return None
    
    # Search for smallest m
    for m in range(52, 5000):
        if time.time() > deadline - 10.0:
            break
        
        candidate_n = 10 * m
        time_left = deadline - time.time()
        
        if time_left <= 5:
            break
        
        volts = find_volts(m, time_left)
        
        if volts is not None:
            best_m = m
            best_n = candidate_n
            lift_edges = []
            for i, (u, v) in enumerate(edges):
                vol = volts[i]
                for t in range(m):
                    u_lift = u * m + t
                    v_lift = v * m + (t + vol) % m
                    lift_edges.append((u_lift, v_lift))
            best_edges = lift_edges
            break
    
    if best_edges is None:
        return _luw_fallback(k, g)
    
    return best_n, best_edges


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


if __name__ == "__main__":
    n, e = construct(3, 16)
    print(f"n={n}")

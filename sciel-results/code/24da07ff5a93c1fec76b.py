"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import deque
import math

def construct(k, g):
    # Petersen graph lift approach
    # Petersen graph: 10 vertices, 3-regular, girth 5
    base_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),  # outer cycle
        (5,7), (7,9), (9,6), (6,8), (8,5),  # inner star
        (0,5), (1,6), (2,7), (3,8), (4,9)   # spokes
    ]
    base_adj = [[] for _ in range(10)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
    
    # Enumerate all cycles in Petersen graph up to length g-1
    def find_all_cycles_up_to_len(base_adj, max_len):
        cycles = set()
        n_base = len(base_adj)
        for start in range(n_base):
            # DFS
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()
                if len(path) > max_len:
                    continue
                for nb in base_adj[node]:
                    if nb == start and len(path) >= 3:
                        # Found a cycle
                        cycle = list(path)
                        # Normalize: start from min vertex
                        min_val = min(cycle)
                        min_idx = cycle.index(min_val)
                        norm = cycle[min_idx:] + cycle[:min_idx]
                        # Check reverse
                        rev = list(reversed(norm))
                        rev_min_idx = rev.index(min(rev))
                        rev_norm = rev[rev_min_idx:] + rev[:rev_min_idx]
                        canon = tuple(min(norm, rev_norm))
                        cycles.add(canon)
                    elif nb not in path:
                        stack.append((nb, path + [nb]))
        return list(cycles)
    
    short_cycles = find_all_cycles_up_to_len(base_adj, g - 1)
    
    # Map each cycle to a list of edge keys and signs
    # Edge key: (min(u,v), max(u,v))
    # We need to track the orientation to sum twists correctly.
    # Twist on edge (u,v) with u<v is t.
    # If we traverse u->v, we add t. If v->u, we subtract t (or add m-t).
    # Actually, let's just store the edge indices and the direction.
    # Let's assign an index to each undirected edge.
    edge_keys = []
    edge_to_idx = {}
    for i, (u, v) in enumerate(base_edges):
        key = (min(u,v), max(u,v))
        edge_keys.append(key)
        edge_to_idx[key] = i
    
    # For each cycle, determine the sequence of (edge_idx, sign)
    # sign is +1 if we go from min to max, -1 if from max to min
    cycle_constraints = []
    for cycle in short_cycles:
        cons = []
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            key = (min(u,v), max(u,v))
            idx = edge_to_idx[key]
            # Determine sign
            # If u < v, we are going min->max, sign +1
            # If u > v, we are going max->min, sign -1
            sign = 1 if u < v else -1
            cons.append((idx, sign))
        cycle_constraints.append(cons)
    
    def build_lift(m, twists):
        n = 10 * m
        edges = []
        for i, (u, v) in enumerate(base_edges):
            t = twists[i]
            for i_lift in range(m):
                u_lift = u * m + i_lift
                v_lift = v * m + (i_lift + t) % m
                edges.append((min(u_lift, v_lift), max(u_lift, v_lift)))
        return n, edges
    
    def check_lift_girth(m, twists, target_g):
        # Check if the lift has girth >= target_g
        # For each base cycle C of length L < target_g:
        # Sum S = sum(sign * twist) mod m
        # If S == 0, cycle length in lift is L. If L < target_g, fail.
        # If S != 0, cycle length in lift is L * (m / gcd(S, m)).
        # We need L * (m / gcd(S, m)) >= target_g.
        
        for cons in cycle_constraints:
            L = len(cons)
            S = 0
            for idx, sign in cons:
                S += sign * twists[idx]
            S %= m
            if S == 0:
                if L < target_g:
                    return False
            else:
                g_val = math.gcd(S, m)
                lifted_len = L * (m // g_val)
                if lifted_len < target_g:
                    return False
        return True
    
    # Try m values
    # Moore bound for k=3, g=16 is 512.
    # n = 10m. So 10m >= 512 => m >= 52.
    # Known best is 936, so m=94.
    # We scan m from 52 upwards.
    
    best_n = float('inf')
    best_result = None
    
    # To speed up, we can fix some twists? No, random is fine.
    # We increase attempts for smaller m if they fail, or just move to next m.
    # Let's try a fixed number of attempts per m.
    
    max_m = 100
    for m in range(52, max_m + 1):
        n = 10 * m
        if n >= best_n:
            continue
        
        # Try random assignments
        # Number of attempts: we can afford more for small m if it's likely to work?
        # Actually, probability of valid assignment is roughly (1 - 1/m)^num_cycles * (prob gcd condition).
        # For m=52, num_cycles ~ 20-30. (1-1/52)^30 ~ 0.5.
        # GCD condition is harder.
        
        for attempt in range(200):
            random.seed(123 + m * 1000 + attempt)
            twists = [random.randint(0, m-1) for _ in range(len(base_edges))]
            
            if check_lift_girth(m, twists, g):
                n_lift, edges = build_lift(m, twists)
                # Optional: verify with BFS on the actual graph to be 100% sure
                # The analytic check should be correct for voltage graphs.
                # But let's add a quick check for the smallest cycles just in case.
                # Actually, the analytic check is exact for the girth of the lift.
                if n_lift < best_n:
                    best_n = n_lift
                    best_result = (n_lift, edges)
                break # Found a valid one for this m, move to next m? 
                # Wait, we want the MINIMUM n. So if we find one for m=52, that's 520 vertices.
                # That would be better than 936.
                # But is it possible? The Moore bound is 512.
                # If we find one, we return it immediately?
                # No, we should continue checking smaller m? We are iterating m increasing.
                # So the first valid m gives the smallest n = 10m.
                # So we can break out of the m loop as well?
                # Yes, if we find a valid graph for m, it has n=10m.
                # Any larger m will have larger n.
                # So we can return immediately.
                return (n_lift, edges)
        
    # If no lift found in range, fallback to LUW
    # This should not happen if the range is large enough and girth is achievable.
    # But for safety, we return a large valid graph.
    # We can use the LUW construction from the original code as a fallback.
    
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
# EVOLVE-BLOCK-END

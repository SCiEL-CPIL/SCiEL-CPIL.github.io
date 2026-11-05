"""Voltage-lift over Petersen using integer linear algebra + greedy repair."""
import time
import sys
import random
import numpy as np

def construct(k, g):
    deadline = time.time() + 140  # safety margin before 150s kill

    # Base graph: Petersen graph (10 vertices, 15 edges, girth 5, 3-regular)
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer 5-cycle
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner 5-cycle (star)
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
    ]
    n_base = 10
    n_e = len(pet_edges)  # 15

    # Build adjacency
    adj = [[] for _ in range(n_base)]
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        adj[u].append(v)
        adj[v].append(u)
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    # Enumerate all non-backtracking closed walks of length 5..g-1
    # For each, compute the signed incidence vector over the 15 edges.
    # Constraint: for voltage vector v in Z_m^15, sum_i c_i * v_i != 0 mod m
    # where c is the signed count vector.
    
    max_L = g - 1
    constraints = set()
    
    # DFS from each vertex, tracking (current, prev, length, counts_list)
    # To reduce memory, we only keep the counts as a tuple when we find a closed walk.
    # We normalize by making the first non-zero entry positive (since c.v=0 iff (-c).v=0).
    
    for start_v in range(n_base):
        # stack entries: (current, prev, length, counts_tuple_or_list)
        # Use list for mutable counts during walk, convert to tuple only at closure
        stack = [(start_v, -1, 0, [0] * n_e)]
        while stack:
            current, prev, length, counts = stack.pop()
            
            if length > max_L:
                continue
            
            for next_v in adj[current]:
                if next_v == prev:
                    continue
                
                e_idx = edge_index[(current, next_v)]
                u, v = pet_edges[e_idx]
                d = 1 if current == u else -1
                
                new_counts = counts.copy()
                new_counts[e_idx] += d
                new_l = length + 1
                
                if next_v == start_v and new_l >= 5:
                    c_tuple = tuple(new_counts)
                    # Normalize
                    first_nz = -1
                    for i in range(n_e):
                        if c_tuple[i] != 0:
                            first_nz = i
                            break
                    if first_nz != -1:
                        if c_tuple[first_nz] < 0:
                            c_tuple = tuple(-x for x in c_tuple)
                        constraints.add(c_tuple)
                
                if new_l < max_L:
                    stack.append((next_v, current, new_l, new_counts))
    
    if not constraints:
        # Fallback: return a known large cage or trivial construction
        # For safety, return the Lazebnik-Ustimenko construction
        return _luw_fallback(k, g)
    
    C_list = list(constraints)
    C = np.array(C_list, dtype=np.int64)
    n_constraints = C.shape[0]
    print(f"Found {n_constraints} unique non-backtracking closed walk constraints", file=sys.stderr)
    
    # We need v in Z_m^15 such that C @ v != 0 mod m for all rows.
    # Strategy: 
    # 1. Pick a prime m such that 10*m is as small as possible but >= Moore bound.
    # 2. Use simulated annealing with efficient numpy cost evaluation.
    # 3. If SA fails, try a different m.
    
    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Moore bound for (3,16) is 765? No, task says lower bound 512.
    # Actually, the Moore bound for cubic girth 16 is 1 + 3*(2^7 - 1) / (2-1) ? 
    # For even girth g=2r, Moore bound is 3*(2^r - 1). r=8 -> 3*255 = 765.
    # But the task says "best lower bound is 512". This might be a stronger lower bound.
    # In any case, we want n = 10*m to be small. Let's try m starting from 52.
    
    best_result = None
    
    # Try m values: 52, 53, 54, ... up to maybe 100
    # Prime m is better for linear algebra properties, but not strictly required.
    # However, if m is prime, Z_m is a field, which might help with some algorithms.
    # But SA works over any modulus.
    
    m_candidates = list(range(52, 80))
    # Prioritize primes
    m_candidates.sort(key=lambda x: (not is_prime(x), x))
    
    for m in m_candidates:
        if time.time() > deadline - 30:
            break
        
        n_lift = 10 * m
        
        # Simulated Annealing
        # Cost function: number of rows in C where (C @ v) % m == 0
        v = np.random.randint(1, m, size=n_e).astype(np.int64)
        
        def cost(v):
            residues = np.mod(C @ v, m)
            return np.sum(residues == 0)
        
        best_v = v.copy()
        best_cost = cost(best_v)
        current_v = v.copy()
        current_cost = best_cost
        
        T = 1.0
        cooling = 0.999
        steps = 50000
        step = 0
        
        while step < steps and time.time() < deadline - 30:
            step += 1
            j = random.randint(0, n_e - 1)
            old_val = current_v[j]
            new_val = random.randint(1, m - 1)
            if new_val == old_val:
                continue
            
            current_v[j] = new_val
            # Efficient cost update:
            # Only rows where C[row, j] != 0 are affected.
            # But for simplicity, recompute cost with numpy (fast for 15 vars, ~10k constraints)
            new_cost = cost(current_v)
            delta = new_cost - current_cost
            
            if delta < 0 or (T > 0.001 and random.random() < np.exp(-delta / T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_v = current_v.copy()
                if new_cost == 0:
                    break
            else:
                current_v[j] = old_val
            
            T *= cooling
        
        if best_cost == 0:
            # Valid assignment found!
            lift_voltages = best_v
            lift_edges = []
            for i, (u, v_base) in enumerate(pet_edges):
                vol = int(lift_voltages[i])
                for layer in range(m):
                    vu = u * m + layer
                    vv = v_base * m + (layer + vol) % m
                    lift_edges.append((vu, vv))
            
            # Quick validation
            deg = [0] * n_lift
            for u, v in lift_edges:
                deg[u] += 1
                deg[v] += 1
            if all(d == k for d in deg):
                # Check for duplicate edges or loops (shouldn't happen with proper voltage)
                # But let's be safe.
                edge_set = set()
                valid = True
                for u, v in lift_edges:
                    if u == v:
                        valid = False
                        break
                    if (u, v) in edge_set or (v, u) in edge_set:
                        valid = False
                        break
                    edge_set.add((u, v))
                if valid:
                    best_result = (n_lift, lift_edges)
                    break
    
    if best_result:
        return best_result
    
    # Fallback: Lazebnik-Ustimenko construction
    return _luw_fallback(k, g)


def _luw_fallback(k, g):
    """Lazebnik-Ustimenko construction for cubic graphs."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

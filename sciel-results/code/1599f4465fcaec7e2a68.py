import time
import math
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0

    def remaining():
        return deadline - time.time()

    # ---------------- fast girth verification (non-backtracking BFS with early exit) ----------------
    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        # Non-backtracking BFS from each vertex
        # State: (node, prev_node)
        # We track shortest distance to each (node, prev) state
        min_cycle = float('inf')

        for s in range(n):
            if time.time() > time_limit:
                return -1
            # dist[(node, prev)] = shortest length of non-backtracking walk from s to (node, prev)
            dist = {}
            # Start: from s, go to each neighbor
            for nb in adj[s]:
                dist[(nb, s)] = 1
                if nb == s:
                    min_cycle = min(min_cycle, 1)
                    return min_cycle

            queue = deque([(nb, s) for nb in adj[s]])
            while queue:
                if time.time() > time_limit:
                    return -1
                node, prev = queue.popleft()
                d = dist[(node, prev)]
                if d >= g - 1:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == s:
                        cycle_len = d + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        state = (nb, node)
                        if state not in dist:
                            dist[state] = d + 1
                            queue.append((nb, node))
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------------- LUW construction (guaranteed fallback) ----------------
    def _luw_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: x * y % q))

    def luw_construct():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        add, sub, mul = _luw_ops(q)
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

    # ---------------- Petersen voltage lift with vectorized constraint checking ----------------
    def petersen_lift_construct():
        if k != 3 or g != 16:
            return None

        # Petersen graph edges (10 vertices, 15 edges)
        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
        ]
        
        n_base = 10
        edge_index = {}
        for i, (u, v) in enumerate(pet_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i
        
        # Precompute all non-backtracking closed walks of length < g in the Petersen graph
        # Each walk corresponds to a linear constraint on voltages
        coeff_vectors = set()
        max_walk_len = g - 1
        
        for start_v in range(n_base):
            stack = [(start_v, -1, 0, [0] * 15)]
            
            while stack:
                v, prev_v, l, counts = stack.pop()
                
                if l >= max_walk_len:
                    continue
                
                for next_v in range(n_base):
                    if next_v == prev_v:
                        continue
                    if next_v == v:
                        continue
                    
                    e_idx = edge_index.get((v, next_v))
                    if e_idx is None:
                        continue
                    
                    new_counts = counts.copy()
                    u_edge, v_edge = pet_edges[e_idx]
                    if v == u_edge and next_v == v_edge:
                        new_counts[e_idx] += 1
                    else:
                        new_counts[e_idx] -= 1
                    
                    new_l = l + 1
                    if next_v == start_v and new_l > 0:
                        coeff_vectors.add(tuple(new_counts))
                    
                    stack.append((next_v, v, new_l, new_counts))
        
        # Normalize constraints
        unique_coeffs = set()
        for c in coeff_vectors:
            if all(x == 0 for x in c):
                continue
            for i in range(15):
                if c[i] != 0:
                    if c[i] < 0:
                        c = tuple(-x for x in c)
                    unique_coeffs.add(c)
                    break
        coeff_list = list(unique_coeffs)
        
        # Build numpy matrix for vectorized evaluation
        if len(coeff_list) > 0:
            coeff_matrix = np.array(coeff_list, dtype=np.int64)
            n_constraints = len(coeff_matrix)
        else:
            coeff_matrix = np.zeros((0, 15), dtype=np.int64)
            n_constraints = 0

        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True

        # Generate primes in ascending order
        primes = []
        candidate = 2
        while len(primes) < 200:
            if is_prime(candidate):
                primes.append(candidate)
            candidate += 1
        
        best_n = float('inf')
        best_edges = None
        
        # Try primes in ascending order to find smallest valid m
        for m in primes:
            if time.time() > deadline - 15:
                break
            
            n_candidate = 10 * m
            if n_candidate >= best_n:
                continue
            
            # More attempts for smaller primes since they're more valuable
            max_attempts = 600 if m < 100 else 300
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 15:
                    break
                
                # Random initialization
                voltages = np.array([random.randint(1, m - 1) for _ in range(15)], dtype=np.int64)
                
                # Local Repair Loop with vectorized checking
                for repair_step in range(100):
                    if time.time() > deadline - 15:
                        break
                    
                    # Check validity using numpy
                    if n_constraints > 0:
                        # Compute all constraint sums at once
                        sums = (coeff_matrix @ voltages) % m
                        violated_indices = np.where(sums == 0)[0]
                        
                        if len(violated_indices) == 0:
                            # All constraints satisfied
                            edges = _build_lifted_graph(pet_edges, voltages.tolist(), m)
                            return n_candidate, edges
                        
                        # Pick a random violated constraint
                        viol_idx = random.choice(violated_indices.tolist())
                        c = coeff_list[viol_idx]
                        
                        # Find edges with non-zero coefficient
                        candidates = [i for i in range(15) if c[i] != 0]
                        if not candidates:
                            break
                        j = random.choice(candidates)
                        
                        # Compute required change: we need c[j] * delta != 0 mod m
                        # Since m is prime and c[j] != 0, any delta != 0 works
                        # But we want to minimize disruption, so pick a small delta
                        delta = random.randint(1, m - 1)
                        voltages[j] = (voltages[j] + delta) % m
                        if voltages[j] == 0:
                            voltages[j] = 1  # Avoid zero voltage
                    else:
                        # No constraints, any assignment works
                        edges = _build_lifted_graph(pet_edges, voltages.tolist(), m)
                        return n_candidate, edges
        
        return None

    def _build_lifted_graph(pet_edges, voltages, m):
        n_base = 10
        edge_index = {}
        for i, (u, v) in enumerate(pet_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i
        
        edges = set()
        for u in range(n_base):
            for v in range(n_base):
                if u == v:
                    continue
                e_idx = edge_index.get((u, v))
                if e_idx is None:
                    continue
                
                vol = voltages[e_idx]
                u_edge, v_edge = pet_edges[e_idx]
                if u == u_edge and v == v_edge:
                    sign = 1
                else:
                    sign = -1
                
                for layer in range(m):
                    u_node = layer * n_base + u
                    target_layer = (layer + sign * vol) % m
                    v_node = target_layer * n_base + v
                    
                    a, b = min(u_node, v_node), max(u_node, v_node)
                    edges.add((a, b))
        
        return list(edges)

    # ---------------- Margulis-LUB Ramanujan graph construction ----------------
    def lub_construct():
        if k != 3 or g != 16:
            return None

        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(math.isqrt(n)) + 1, 2):
                if n % i == 0:
                    return False
            return True

        # Find primes p, q such that p*q >= 512 (Moore bound)
        # and the graph has girth >= 16
        best_result = None

        # Generate candidate prime pairs
        primes = []
        for n in range(2, 500):
            if is_prime(n):
                primes.append(n)

        # We want p*q >= 512, minimize p*q
        # Try p <= q to avoid duplicates
        candidates = []
        for i, p in enumerate(primes):
            for q in primes[i:]:
                n = p * q
                if n >= 512 and n <= 2000:
                    candidates.append((n, p, q))

        candidates.sort()

        for n, p, q in candidates:
            if remaining() < 30:
                break

            # Build the Cayley graph on Z_p x Z_q
            # Generators: (1,0), (-1,0), (0,1)
            # Vertex (x,y) -> index x*q + y
            edges = []
            for x in range(p):
                for y in range(q):
                    idx = x * q + y
                    # Neighbor (x+1, y)
                    nx = (x + 1) % p
                    nidx = nx * q + y
                    if idx < nidx:
                        edges.append((idx, nidx))
                    # Neighbor (x-1, y)
                    nx = (x - 1) % p
                    nidx = nx * q + y
                    if idx < nidx:
                        edges.append((idx, nidx))
                    # Neighbor (x, y+1)
                    ny = (y + 1) % q
                    nidx = x * q + ny
                    if idx < nidx:
                        edges.append((idx, nidx))

            # Verify degree
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)

            valid = True
            for i in range(n):
                if len(adj[i]) != 3:
                    valid = False
                    break
            if not valid:
                continue

            # Check girth with early exit
            gc = check_girth(n, edges, g, deadline - 10)
            if gc >= g:
                return (n, edges)
            elif gc > 0:
                # Found a cycle shorter than g, try next pair
                continue

        return None

    # Try Petersen voltage lift first (often gives smaller graphs)
    result = petersen_lift_construct()
    if result is not None:
        return result

    # Try Margulis-LUB construction
    result = lub_construct()
    if result is not None:
        return result

    # Fallback to the LUW construction.
    return luw_construct()

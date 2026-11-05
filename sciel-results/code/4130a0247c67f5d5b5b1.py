import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses voltage lift of Petersen graph for k=3.
    """
    start_time = time.time()
    deadline = start_time + 115.0

    if k == 3:
        return _construct_petersen_lift(g, deadline, start_time)
    else:
        return _construct_luw(k, g, d=7)

def _construct_petersen_lift(g, deadline, start_time):
    """
    Constructs a girth-g graph by lifting the Petersen graph with voltages in Z_m.
    Uses vectorized constraint checking and simulated annealing local repair.
    """
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
    
    adj_base = [[] for _ in range(n_base)]
    for u, v in pet_edges:
        adj_base[u].append(v)
        adj_base[v].append(u)

    # Precompute all non-backtracking closed walks of length < g in the Petersen graph
    # We need to avoid cycles of length 1 to g-1. Since the graph is 3-regular and simple,
    # min cycle in base is 5. But we need to ensure the lift has no short cycles.
    # A cycle in the lift projects to a closed walk in the base.
    # The voltage sum around the walk must be non-zero mod m.
    coeff_vectors = set()
    max_walk_len = g - 1
    
    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0] * 15)]
        
        while stack:
            v, prev_v, l, counts = stack.pop()
            
            if l >= max_walk_len:
                continue
            
            for next_v in adj_base[v]:
                if next_v == prev_v:
                    continue
                
                e_idx = edge_index[(v, next_v)]
                u_edge, v_edge = pet_edges[e_idx]
                
                new_counts = counts.copy()
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
        # Normalize sign: make first non-zero element positive
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

    def check_girth_fast(n, edges, min_girth, time_limit):
        """
        BFS-based girth check. Returns min_cycle_len found, or -1 if timeout.
        """
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        min_cycle = float('inf')
        
        for s in range(n):
            if time.time() > time_limit:
                return -1
            
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < min_girth:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            
            if min_cycle < min_girth:
                return min_cycle
        
        return min_cycle

    def solve_voltage(m, max_iters=5000):
        """
        Simulated annealing to find voltage assignment v such that
        C @ v != 0 (mod m) for all rows of C.
        """
        # Initialize with random values in [1, m-1] to avoid trivial zero issues
        # though 0 is allowed for voltages, non-zero is often better for mixing
        v = np.random.randint(1, m, size=15).astype(np.int64)

        def cost(v):
            if n_constraints == 0:
                return 0
            residues = np.mod(coeff_matrix @ v, m)
            return np.sum(residues == 0)

        best_v = v.copy()
        best_cost = cost(v)

        if best_cost == 0:
            return v, 0

        # Simulated annealing parameters
        T = 1.0
        cooling = 0.9999
        min_T = 0.01

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Pick a random variable to change
            j = random.randint(0, 14)
            old_val = v[j]
            
            # Try a random new value
            v[j] = random.randint(1, m - 1)

            new_cost = cost(v)
            delta = new_cost - best_cost

            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_v = v.copy()
                if new_cost == 0:
                    return v, 0
            else:
                v[j] = old_val

            T *= cooling

        return best_v, best_cost

    # Try primes in increasing order
    # Start from a reasonable size to balance search space and graph size
    # Moore bound for k=3, g=16 is roughly 2 * (3^7 - 1)/2 + 1 = 3^7 + 1 = 2188? No.
    # Moore bound: 1 + 3 + 3*2 + ... + 3*2^(g/2 - 1) for even g.
    # g=16 -> radius 8.
    # 1 + 3 + 6 + 12 + 24 + 48 + 96 + 192 + 384 = 766.
    # So n must be >= 766? Wait, the prompt says best lower bound is 512.
    # Moore bound formula: 1 + k * (k-1)^(d-1) / 2 * 2 ...
    # For k=3, g=16 (radius 8): 1 + 3 * (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 1 + 3*255 = 766.
    # So n >= 766.
    # Smallest known is 936.
    # We want to find a solution near the lower bound.
    # Let's try m such that 10*m is close to 766 but >= 766.
    # m >= 77.
    
    primes_to_try = []
    candidate = 2
    while len(primes_to_try) < 50:
        if is_prime(candidate):
            primes_to_try.append(candidate)
        candidate += 1
    
    # Filter primes that give n >= 766 (Moore bound)
    # Actually, let's just iterate primes starting from 79 upwards.
    # 10 * 79 = 790.
    
    best_n = float('inf')
    best_edges = None

    # We will try primes starting from 79
    # But we should also check if a smaller m works? No, Moore bound is hard.
    # Wait, is the Moore bound tight for all g? No.
    # But n cannot be less than the Moore bound.
    # So 10*m >= 766 => m >= 76.6 => m >= 77.
    # Smallest prime >= 77 is 79.
    
    start_m = 79
    
    m = start_m
    while m < 500 and time.time() < deadline - 20:
        if is_prime(m):
            n_candidate = 10 * m
            
            # Try multiple restarts
            for restart in range(20):
                if time.time() > deadline - 15:
                    break
                
                v, cost = solve_voltage(m, max_iters=5000)
                
                if cost == 0:
                    # Construct graph
                    edges = _build_lifted_graph(pet_edges, v.tolist(), m)
                    
                    # Verify girth
                    gc = check_girth_fast(n_candidate, edges, g, deadline - 5)
                    
                    if gc >= g:
                        if n_candidate < best_n:
                            best_n = n_candidate
                            best_edges = edges
                            return best_n, best_edges
                        # If we found a valid one, we can return it immediately
                        # since we are iterating m in increasing order
                        return n_candidate, edges

        m += 1

    # Fallback: LUW
    return _construct_luw(k=3, g=g, d=11)

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

def _construct_luw(k, g, d=None, variant=0):
    """
    LUW construction with specified depth d and variant for rule selection.
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

def _has_girth_at_least(n, edges, g):
    """
    Check if graph has girth at least g using non-backtracking walks.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    max_len = g - 1
    
    for start in range(n):
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
        
        while queue:
            u, prev, l = queue.popleft()
            
            if u == start and l > 0:
                return False
            
            if l >= max_len:
                continue
            
            for v in adj[u]:
                if v != prev:
                    queue.append((v, u, l + 1))
    
    return True

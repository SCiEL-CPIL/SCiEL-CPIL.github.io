import time
import random
import numpy as np

def construct(k, g):
    deadline = time.time() + 115.0
    rng = random.Random(42)
    
    # Moore bound
    if g % 2 == 1:
        half = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(half))
    else:
        half = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(half))
        
    # Special case for k=5, g=9 using Clebsch lift
    if k == 5 and g == 9:
        result = try_clebsch_lift(5, 9, deadline, rng)
        if result:
            return result

    best_n = None
    best_edges = None
    
    # Helper to check girth >= g, returns True/False
    def has_girth_at_least_g(adj, n, g):
        visit_id = np.zeros(n * n, dtype=np.int32)
        current_id = 1
        
        for start in range(n):
            queue = []
            for nb in adj[start]:
                idx = start * n + nb
                if visit_id[idx] != current_id:
                    visit_id[idx] = current_id
                    queue.append((start, nb, 1))
            
            while queue:
                new_queue = []
                for prev, curr, depth in queue:
                    if depth >= g - 1:
                        continue
                    for nxt in adj[curr]:
                        if nxt == prev:
                            continue
                        if nxt == start:
                            if depth + 1 < g:
                                return False
                        else:
                            idx = curr * n + nxt
                            if visit_id[idx] != current_id:
                                visit_id[idx] = current_id
                                new_queue.append((curr, nxt, depth + 1))
                queue = new_queue
            current_id += 1
            if current_id > 2**31:
                visit_id[:] = 0
                current_id = 1
        return True

    # Returns (min_cycle_length, list_of_edges_in_shortest_cycle)
    def find_shortest_cycle(adj, n, max_girth):
        visit_id = np.zeros(n * n, dtype=np.int32)
        current_id = 1
        min_cycle = max_girth + 1
        best_cycle_edges = None
        
        for start in range(n):
            queue = []
            for nb in adj[start]:
                idx = start * n + nb
                if visit_id[idx] != current_id:
                    visit_id[idx] = current_id
                    queue.append((start, nb, 1, [(start, nb)]))
            
            while queue:
                new_queue = []
                for prev, curr, depth, path in queue:
                    if depth >= max_girth - 1:
                        continue
                    for nxt in adj[curr]:
                        if nxt == prev:
                            continue
                        if nxt == start:
                            cycle_len = depth + 1
                            if cycle_len < min_cycle:
                                min_cycle = cycle_len
                                # Reconstruct cycle edges: path + (curr, start)
                                cycle_edges = list(path) + [(curr, start)]
                                best_cycle_edges = cycle_edges
                        else:
                            idx = curr * n + nxt
                            if visit_id[idx] != current_id:
                                visit_id[idx] = current_id
                                new_path = path + [(curr, nxt)]
                                new_queue.append((curr, nxt, depth + 1, new_path))
                queue = new_queue
            current_id += 1
            if current_id > 2**31:
                visit_id[:] = 0
                current_id = 1
            if min_cycle == 3:
                break
        return min_cycle, best_cycle_edges

    def random_regular_graph(n, k, rng):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        rng.shuffle(stubs)
        
        adj = [[] for _ in range(n)]
        seen = [set() for _ in range(n)]
        
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None
            if v in seen[u]:
                return None
            adj[u].append(v)
            adj[v].append(u)
            seen[u].add(v)
            seen[v].add(u)
            
        return adj

    def get_all_edges(adj, n):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return edges

    def try_swap(adj, n, k, rng, prefer_edges=None):
        edges = get_all_edges(adj, n)
        if len(edges) < 2:
            return False
            
        if prefer_edges and len(prefer_edges) >= 2:
            # Try to pick one edge from prefer_edges and one random edge
            e1 = rng.choice(prefer_edges)
            e2 = rng.choice(edges)
            if e1 == e2:
                e2 = rng.choice(edges)
                if e1 == e2:
                    return False
        else:
            e1_idx = rng.randint(0, len(edges) - 1)
            e2_idx = rng.randint(0, len(edges) - 2)
            if e2_idx >= e1_idx:
                e2_idx += 1
            e1 = edges[e1_idx]
            e2 = edges[e2_idx]
            
        u1, v1 = e1
        u2, v2 = e2
        
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            return False
            
        adj[u1].remove(v1)
        adj[v1].remove(u1)
        adj[u2].remove(v2)
        adj[v2].remove(u2)
        
        if u2 not in adj[u1] and v2 not in adj[v1]:
            adj[u1].append(u2)
            adj[u2].append(u1)
            adj[v1].append(v2)
            adj[v2].append(v1)
            return True
        
        if v2 not in adj[u1] and u2 not in adj[v1]:
            adj[u1].append(v2)
            adj[v2].append(u1)
            adj[v1].append(u2)
            adj[u2].append(v1)
            return True
            
        adj[u1].append(v1)
        adj[v1].append(u1)
        adj[u2].append(v2)
        adj[v2].append(u2)
        
        return False

    # Main search loop
    n_start = moore
    if (n_start * k) % 2 != 0:
        n_start += 1
        
    max_n_to_try = 500
    
    for n in range(n_start, max_n_to_try + 1, 2):
        if time.time() > deadline - 10:
            break
            
        attempts = 0
        max_attempts = 30
        
        while attempts < max_attempts and time.time() < deadline - 5:
            adj = random_regular_graph(n, k, rng)
            if adj is None:
                attempts += 1
                continue
                
            if has_girth_at_least_g(adj, n, g):
                edges = get_all_edges(adj, n)
                return n, edges
            
            # Local search: repeatedly find shortest cycle, try to break it
            for _ in range(500):
                if time.time() > deadline - 5:
                    break
                
                min_cycle, cycle_edges = find_shortest_cycle(adj, n, g)
                
                if min_cycle >= g:
                    edges = get_all_edges(adj, n)
                    return n, edges
                
                # Try to break the shortest cycle by swapping one of its edges
                # with a random edge
                if cycle_edges:
                    # Normalize cycle edges to canonical form
                    canon_cycle = []
                    for a, b in cycle_edges:
                        canon_cycle.append((min(a,b), max(a,b)))
                    
                    # Try multiple times to break this cycle
                    for _attempt in range(10):
                        if time.time() > deadline - 5:
                            break
                        if try_swap(adj, n, k, rng, prefer_edges=canon_cycle):
                            break
                else:
                    try_swap(adj, n, k, rng)
                
                # Check girth periodically
                if _ % 20 == 0:
                    if has_girth_at_least_g(adj, n, g):
                        edges = get_all_edges(adj, n)
                        return n, edges
            
            attempts += 1
            
    # Fallback: LUW construction
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
        if time.time() > deadline - 2:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def try_clebsch_lift(k, g, deadline, rng):
    # Clebsch graph: 16 vertices, 5-regular
    n_base = 16
    base_adj = [[] for _ in range(n_base)]
    # Edge if xor has weight 1 or 4
    for u in range(n_base):
        for v in range(u+1, n_base):
            x = u ^ v
            # popcount
            pc = bin(x).count('1')
            if pc == 1 or pc == 4:
                base_adj[u].append(v)
                base_adj[v].append(u)
                
    # Check if it is 5-regular
    for i in range(n_base):
        if len(base_adj[i]) != 5:
            return None

    # Generate non-backtracking walks of length 5 to 8
    walks = []
    # Start from each vertex, each neighbor
    for start in range(n_base):
        for nb in base_adj[start]:
            # Walk: start -> nb
            current = nb
            prev = start
            path = [start, nb]
            # Extend to length 8
            for depth in range(2, 9):
                new_paths = []
                for nxt in base_adj[current]:
                    if nxt == prev:
                        continue
                    new_path = path + [nxt]
                    if depth + 1 >= 5: # Record walks of length >= 5
                        walks.append(new_path)
                    new_paths.append((nxt, current, new_path))
                # We need to branch, so we can't just keep one path
                # This simple DFS is inefficient for generating all walks.
                # Let's do a proper BFS/DFS generation.
                break # This logic is flawed for generation, need full recursion
        
    # Proper walk generation
    walks = []
    def gen_walks(curr, prev, path, depth, max_depth):
        if depth > max_depth:
            return
        if depth >= 5:
            walks.append(path[:])
        for nxt in base_adj[curr]:
            if nxt == prev:
                continue
            gen_walks(nxt, curr, path + [nxt], depth + 1, max_depth)
            
    for start in range(n_base):
        for nb in base_adj[start]:
            gen_walks(nb, start, [start, nb], 2, 8)
            
    # Number of walks
    # 16 * 5 * (4^0 + 4^1 + 4^2 + 4^3) = 80 * (1+4+16+64) = 80 * 85 = 6800 walks.
    # This is small enough to handle in Python.

    # Map edges to indices
    edge_list = []
    edge_index = {}
    for u in range(n_base):
        for v in base_adj[u]:
            if u < v:
                idx = len(edge_list)
                edge_list.append((u, v))
                edge_index[(u, v)] = idx
                edge_index[(v, u)] = idx
                
    num_edges = len(edge_list)
    
    # Convert walks to edge index lists
    walk_edges = []
    for w in walks:
        edges_in_walk = []
        for i in range(len(w)-1):
            u, v = w[i], w[i+1]
            edges_in_walk.append(edge_index[(u, v)])
        walk_edges.append(edges_in_walk)
        
    # Try different group sizes m
    # We want n = 16*m to be small, but >= Moore bound (426)
    # So m >= 27
    primes_to_try = [29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for m in primes_to_try:
        if time.time() > deadline - 10:
            break
            
        # Local search for voltages
        best_voltages = None
        best_score = float('inf')
        
        for restart in range(50):
            voltages = [rng.randint(0, m-1) for _ in range(num_edges)]
            
            # Calculate initial score
            score = 0
            for we in walk_edges:
                s = 0
                for e in we:
                    s = (s + voltages[e]) % m
                if s == 0:
                    score += 1
                    
            for step in range(1000):
                if time.time() > deadline - 10:
                    break
                if score == 0:
                    best_voltages = voltages[:]
                    break
                    
                # Pick a random walk that is zero
                # To speed up, we can just pick a random edge and change it
                # and recalculate score? No, too slow.
                # Pick a random zero walk and change one of its edges.
                zero_walks = []
                for i, we in enumerate(walk_edges):
                    s = 0
                    for e in we:
                        s = (s + voltages[e]) % m
                    if s == 0:
                        zero_walks.append(i)
                        
                if not zero_walks:
                    best_voltages = voltages[:]
                    break
                    
                wi = rng.choice(zero_walks)
                we = walk_edges[wi]
                ei = rng.choice(we)
                
                old_val = voltages[ei]
                new_val = rng.randint(0, m-1)
                while new_val == old_val:
                    new_val = rng.randint(0, m-1)
                    
                voltages[ei] = new_val
                
                # Recalculate score
                new_score = 0
                for w in walk_edges:
                    s = 0
                    for e in w:
                        s = (s + voltages[e]) % m
                    if s == 0:
                        new_score += 1
                            
                if new_score < score:
                    score = new_score
                else:
                    voltages[ei] = old_val
                    
            if best_voltages is None and score < best_score:
                best_score = score
                best_voltages = voltages[:]
                
        if best_voltages is not None and best_score == 0:
            # Build lifted graph
            n_lifted = n_base * m
            adj_lifted = [[] for _ in range(n_lifted)]
            edges_lifted = []
            
            for u in range(n_base):
                for v in base_adj[u]:
                    if u < v:
                        e_idx = edge_index[(u, v)]
                        vol = best_voltages[e_idx]
                        # Edge (u, a) to (v, a+vol)
                        # And (v, b) to (u, b-vol)
                        for a in range(m):
                            u1 = u * m + a
                            v1 = v * m + (a + vol) % m
                            adj_lifted[u1].append(v1)
                            adj_lifted[v1].append(u1)
                            if u1 < v1:
                                edges_lifted.append((u1, v1))
                                
            return n_lifted, edges_lifted
            
    return None

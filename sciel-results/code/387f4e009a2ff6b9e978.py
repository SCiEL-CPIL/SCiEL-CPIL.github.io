import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety

    best_n = float('inf')
    best_edges = None

    def verify_graph(n, edges, k, g, time_limit=10.0):
        """
        Verify that the graph is:
        1. Simple (no loops, no multi-edges)
        2. k-regular
        3. Has girth >= g
        """
        if time.time() > start_time + time_limit:
            return False

        if n == 0:
            return False
        
        # Check degree and build adjacency list
        adj = [[] for _ in range(n)]
        seen_edges = set()
        
        for u, v in edges:
            if u == v:
                return False
            if u < 0 or u >= n or v < 0 or v >= n:
                return False
            edge_key = (min(u, v), max(u, v))
            if edge_key in seen_edges:
                return False
            seen_edges.add(edge_key)
            adj[u].append(v)
            adj[v].append(u)
        
        # Check regularity
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Optimized Girth Check
        max_depth = g // 2
        
        for u in range(n):
            if time.time() > start_time + time_limit:
                return False
                
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            
            while q:
                curr = q.popleft()
                if dist[curr] >= max_depth:
                    continue
                
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        parent[v] = curr
                        q.append(v)
                    else:
                        # Ignore tree edges (parent-child)
                        if v == parent[curr]:
                            continue
                        # If v is a child (dist[v] == dist[curr] + 1) and not parent, 
                        # it was discovered by another node in the same level.
                        # This forms a cycle of length dist[curr] + 1 + dist[v].
                        cycle_len = dist[curr] + 1 + dist[v]
                        if cycle_len < g:
                            return False
                            
        return True

    def build_luw_graph(q_val, d_val):
        # Original LUW construction
        if q_val == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q_val), (lambda x, y: (x - y) % q_val), (lambda x, y: x * y % q_val)

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        
        def get_rule(d_v):
            r = dict(rule)
            for c in range(5, d_v + 1):
                rc, c0 = (c - 5) % 4, c - (c - 5) % 4
                r[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][rc]
            return r

        r = get_rule(d_val)
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0] * (d_val - 1)
                for c in range(2, d_val + 1):
                    a, b = r[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d_val)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # 1. Try LUW family as fallback
    q = k
    min_d = max(3, g - 5)
    
    # Try a wider range of d values
    for d_val in range(min_d, min_d + 15):
        if time.time() > deadline - 30:
            break
        try:
            n, edges = build_luw_graph(q, d_val)
            if n > 50000:
                continue
            if n < best_n:
                t_limit = 5.0 if n > 10000 else 20.0
                if verify_graph(n, edges, k, g, time_limit=t_limit):
                    best_n = n
                    best_edges = edges
                    break
        except Exception:
            pass

    # 2. Try Cayley graph on A_6 (order 360)
    # This is a specific search for a smaller graph (n=360 < 1024)
    if time.time() < deadline - 20:
        try:
            n_cayley, edges_cayley = search_cayley_a6(k, g, start_time, deadline)
            if n_cayley is not None and n_cayley < best_n:
                best_n = n_cayley
                best_edges = edges_cayley
        except Exception:
            pass

    if best_edges is None:
        return 0, []

    return best_n, best_edges

def search_cayley_a6(k, g, start_time, deadline):
    """
    Search for a k-regular Cayley graph on A_6 (order 360) with girth >= g.
    """
    if k != 4:
        return None, None
    
    # Generate A_6 elements as tuples of permutations of 0..5
    # We only need even permutations.
    # Total 720 permutations in S_6. Half are in A_6.
    
    import itertools
    
    # Generate all permutations
    perms = list(itertools.permutations(range(6)))
    
    # Filter even permutations
    def is_even(perm):
        inv = 0
        for i in range(6):
            for j in range(i+1, 6):
                if perm[i] > perm[j]:
                    inv += 1
        return inv % 2 == 0
        
    a6_perms = [p for p in perms if is_even(p)]
    n = len(a6_perms) # 360
    
    # Map perm to index
    perm_to_idx = {p: i for i, p in enumerate(a6_perms)}
    
    # Identity
    identity = (0, 1, 2, 3, 4, 5)
    idx_identity = perm_to_idx[identity]
    
    # Precompute multiplication table? 360x360 is 129600.
    # Might be too slow to precompute fully in Python.
    # Let's do on-the-fly multiplication.
    
    def multiply(p1, p2):
        # p1 * p2
        # Apply p2 then p1? Or p1 then p2?
        # Standard: (p1 * p2)(i) = p1(p2(i))
        return tuple(p1[p2[i]] for i in range(6))
        
    def inverse(p):
        inv = [0]*6
        for i in range(6):
            inv[p[i]] = i
        return tuple(inv)
        
    # We need 4 generators.
    # Generators should not be identity or inverses of each other.
    # Random search for 4 generators.
    
    max_attempts = 500
    for _ in range(max_attempts):
        if time.time() > deadline - 10:
            break
            
        # Pick 4 random distinct generators
        # Ensure they are not identity
        candidates = [p for i, p in enumerate(a6_perms) if i != idx_identity]
        
        # Shuffle and pick 4
        random.shuffle(candidates)
        gens = candidates[:4]
        
        # Check if any generator is inverse of another
        valid = True
        for i in range(4):
            for j in range(i+1, 4):
                if multiply(gens[i], gens[j]) == identity:
                    valid = False
                    break
                if multiply(gens[j], gens[i]) == identity:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Build edges
        edges = []
        # For each vertex, add edges to vertex * gen
        # We only need to add each edge once.
        # Since it's a Cayley graph, it's vertex-transitive.
        # We can just iterate all vertices and add (v, v*gen).
        # To avoid duplicates, we can check if v < v*gen?
        # Or just add all and let verify_graph handle it?
        # verify_graph checks for multi-edges.
        # It's better to generate unique edges.
        
        # For each generator, the edges are (v, v*g).
        # Since g is not identity, v != v*g.
        # If g and g^-1 are both in gens, we get duplicates.
        # We ensured no pair is inverse.
        # But what if g1 * g2 = identity? Checked.
        # What if g1 = g2? We picked distinct candidates.
        
        # So (v, v*g) is unique for a fixed g.
        # Across different g, can (v, v*g1) == (v', v'*g2)?
        # Yes, if v*g1 = v'*g2.
        # This implies v' = v * g1 * g2^-1.
        # So we might add the same edge twice if we iterate all gens.
        # To be safe, use a set of edges.
        
        edge_set = set()
        for v_idx in range(n):
            v_perm = a6_perms[v_idx]
            for g in gens:
                u_perm = multiply(v_perm, g)
                u_idx = perm_to_idx[u_perm]
                if v_idx < u_idx:
                    edge_set.add((v_idx, u_idx))
                else:
                    edge_set.add((u_idx, v_idx))
                    
        edges = list(edge_set)
        
        # Verify
        # This is expensive (360 vertices).
        # But we have 500 attempts.
        # 500 * (build + verify) might be too slow.
        # Let's limit attempts.
        
        # Quick check: Girth.
        # We can do a fast BFS check for girth < g.
        # If it passes, return.
        
        # To speed up, we can check girth first without full verification?
        # verify_graph does both.
        # Let's just call verify_graph.
        
        if verify_graph(n, edges, k, g, time_limit=5.0):
            return n, edges
            
    return None, None

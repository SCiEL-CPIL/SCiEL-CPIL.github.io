"""Voltage-lift construction of a 3-regular graph with large girth."""
import time
import random
import numpy as np

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    Uses voltage lifting of the Petersen graph over Z_p with random voltage assignments,
    then verifies girth via non-backtracking walks.
    """
    deadline = time.time() + 115.0  # safety margin before 120s limit
    
    if k != 3:
        # Fallback for non-3-regular: use LUW construction
        return _luw_fallback(k, g)
    
    # Petersen graph: 10 vertices, 3-regular
    # Standard Petersen graph edges
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer pentagon
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),  # inner star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
    ]
    n_base = 10
    # Adjacency list for Petersen
    adj = [[] for _ in range(n_base)]
    for u, v in petersen_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # We need girth >= 16. For a voltage lift over Z_p, the total vertices = 10*p.
    # We search for a voltage assignment that gives girth >= 16.
    # The girth of the lift is determined by the shortest non-backtracking closed walk
    # in the base whose voltage sum is 0 mod p.
    
    # Enumerate all non-backtracking closed walks of length < 16 in the Petersen graph.
    # For each such walk, we get a linear constraint on the voltages.
    # We want to assign voltages so that no such walk has sum = 0 mod p.
    
    # First, let's find all non-backtracking closed walks of length up to g-1 = 15 in Petersen.
    # Since Petersen has girth 5, there are walks of length 5, 6, 7, ...
    
    def find_nonbacktracking_closed_walks(adj, max_len):
        """Find all non-backtracking closed walks of length <= max_len."""
        walks = []
        n = len(adj)
        # A non-backtracking walk is a sequence v0, v1, ..., vl where vi != v_{i-2} for i >= 2
        # and it's closed means v0 == vl.
        # We'll use DFS with state (current_vertex, previous_vertex, length).
        
        def dfs(v, prev, length, path):
            if time.time() > deadline:
                return
            if length == 0:
                return
            if length >= max_len:
                if v == path[0]:
                    walks.append(list(path))
                return
            for nb in adj[v]:
                if nb == prev and length >= 2:
                    continue  # would backtrack
                if length == 1 and nb == path[0]:
                    # This would close a walk of length 1 (self-loop), not possible in simple graph
                    continue
                path.append(nb)
                dfs(nb, v, length + 1, path)
                path.pop()
        
        for start in range(n):
            dfs(start, -1, 0, [start])
        
        return walks
    
    # For efficiency, let's be smarter. We only need walks of length 5 to 15 (girth 5 is the min).
    # Actually, we need to ensure NO closed non-backtracking walk of length < g has voltage sum 0.
    
    # Let's collect all non-backtracking closed walks of length 5 to g-1.
    all_walks = find_nonbacktracking_closed_walks(adj, g - 1)
    
    # For each walk, extract the sequence of directed edges.
    # A walk v0, v1, ..., vl=v0 corresponds to directed edges (v0,v1), (v1,v2), ..., (v_{l-1}, v0).
    # The voltage of the lift is the sum of voltages on these directed edges.
    
    # There are 20 directed edges in Petersen (10 undirected * 2).
    # Let's index them.
    directed_edges = []
    for u, v in petersen_edges:
        directed_edges.append((u, v))
        directed_edges.append((v, u))
    edge_index = {}
    for i, (u, v) in enumerate(directed_edges):
        edge_index[(u, v)] = i
    n_edges = len(directed_edges)  # 20
    
    # For each walk, create a vector of edge multiplicities (0 or 1 for each directed edge)
    walk_vectors = []
    for walk in all_walks:
        vec = [0] * n_edges
        for i in range(len(walk) - 1):
            e = (walk[i], walk[i + 1])
            vec[edge_index[e]] = 1
        walk_vectors.append(vec)
    
    # Now we want to assign voltages v[0..19] in Z_p such that for each walk vector w,
    # sum(w[i] * v[i]) != 0 mod p.
    
    # Strategy: try small primes p, and for each p, try random voltage assignments.
    # Check if all walk constraints are satisfied.
    
    best_result = None
    best_n = float('inf')
    
    # Try primes from small to larger
    primes_to_try = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83]
    
    for p in primes_to_try:
        if time.time() > deadline - 10:
            break
        
        n_total = n_base * p
        if n_total >= best_n:
            continue
        
        # Try random voltage assignments
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random voltages in Z_p, avoid all-zero
            voltages = [random.randint(0, p - 1) for _ in range(n_edges)]
            if all(v == 0 for v in voltages):
                continue
            
            # Check all walk constraints
            valid = True
            for vec in walk_vectors:
                s = 0
                for i in range(n_edges):
                    if vec[i]:
                        s = (s + voltages[i]) % p
                if s == 0:
                    valid = False
                    break
            
            if valid:
                # Construct the lifted graph
                # Vertex (v, t) where v in 0..9, t in 0..p-1
                # Edge from (u, t) to (v, t + voltage(u,v) mod p) for each directed edge (u,v)
                edges = []
                seen = set()
                for u, v in petersen_edges:
                    for t in range(p):
                        t1 = (t + voltages[edge_index[(u, v)]]) % p
                        # Vertex ids: u*p + t, v*p + t1
                        vid1 = u * p + t
                        vid2 = v * p + t1
                        if vid1 < vid2:
                            key = (vid1, vid2)
                        else:
                            key = (vid2, vid1)
                        if key not in seen:
                            seen.add(key)
                            edges.append(key)
                
                # Verify: check degree and girth
                # Degree check
                degree = [0] * n_total
                for u, v in edges:
                    degree[u] += 1
                    degree[v] += 1
                if any(d != k for d in degree):
                    continue
                
                # Girth check via non-backtracking BFS from each vertex
                girth = _check_girth(n_total, edges, k, g)
                if girth >= g:
                    best_n = n_total
                    best_result = (n_total, edges)
                    break  # Found valid for this p, try smaller p next
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: LUW construction
    return _luw_fallback(k, g)


def _check_girth(n, edges, k, target_girth):
    """Check if graph has girth at least target_girth using non-backtracking walks."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Non-backtracking BFS from each vertex
    # State: (current_vertex, previous_vertex)
    # We want to find the shortest cycle through each vertex.
    
    for start in range(n):
        # BFS with state (v, prev)
        visited = {}  # (v, prev) -> distance
        queue = []
        for nb in adj[start]:
            visited[(nb, start)] = 1
            queue.append((nb, start, 1))
        
        idx = 0
        while idx < len(queue):
            v, prev, dist = queue[idx]
            idx += 1
            
            if dist >= target_girth:
                continue
            
            for nb in adj[v]:
                if nb == prev:
                    continue
                if nb == start:
                    # Found a cycle of length dist + 1
                    return dist + 1
                if (nb, v) not in visited:
                    visited[(nb, v)] = dist + 1
                    queue.append((nb, v, dist + 1))
    
    return target_girth  # No cycle shorter than target_girth found


def _luw_fallback(k, g):
    """Fallback LUW construction for non-3-regular or when voltage lift fails."""
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

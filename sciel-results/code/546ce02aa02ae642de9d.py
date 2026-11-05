"""Rewritten program: construct k-regular graph of girth g using random Cayley graphs."""
# EVOLVE-BLOCK-START
import random
import itertools
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: Try to build a Cayley graph on a non-abelian group using involutory generators.
    If that fails within time, fall back to the Lazebnik-Ustimenko-Woldar construction.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    start_time = time.time()
    time_limit = 110  # leave 10s margin
    
    # Moore bound calculation
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    # L-U-W construction as reliable fallback
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q)"""
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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
    
    # Girth check via BFS from identity for Cayley graphs
    def check_girth_cayley(n, generators, group_mul, identity=0, required_girth=9):
        """Check if Cayley graph has girth >= required_girth."""
        # BFS from identity, find shortest cycle through identity
        # For vertex-transitive graphs, this gives the girth
        dist = [-1] * n
        dist[identity] = 0
        queue = deque([identity])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= required_girth - 1:
                continue  # no need to go deeper
            for s in generators:
                v = group_mul(u, s)
                if v == identity:
                    # Found a cycle of length dist[u] + 1
                    if dist[u] + 1 < required_girth:
                        return False
                elif dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        
        return True
    
    # Helper: check girth of a general graph
    def check_girth_general(n, adj, required_girth):
        """Check if graph has girth >= required_girth using BFS from each vertex."""
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= required_girth - 1:
                    continue
                for v in adj[u]:
                    if v == start:
                        if dist[u] + 1 < required_girth:
                            return False
                    elif dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
        return True
    
    # Try to find a good Cayley graph on S_6 x Z/2
    def try_cayley_s6z2(k, g):
        """Try to find a Cayley graph on S_6 x Z/2 with girth >= g."""
        if time.time() - start_time > time_limit:
            return None
        
        # Build S_6 as permutations (0-indexed)
        # Use tuple representation
        def perm_mul(p, q):
            """Compose permutations p and q (apply q first, then p)."""
            return tuple(p[q[i]] for i in range(len(p)))
        
        def perm_inv(p):
            inv = [0] * len(p)
            for i in range(len(p)):
                inv[p[i]] = i
            return tuple(inv)
        
        # Generate all permutations of 6 elements
        from itertools import permutations
        all_perms = list(permutations(range(6)))
        perm_to_idx = {p: i for i, p in enumerate(all_perms)}
        
        # S_6 x Z/2: elements are (perm, bit)
        # Identity: (id_perm, 0)
        id_perm = tuple(range(6))
        identity = (id_perm, 0)
        
        # Build group multiplication
        def group_mul(e1, e2):
            p1, b1 = e1
            p2, b2 = e2
            return (perm_mul(p1, p2), b1 ^ b2)
        
        # Find all involutions
        involutions = []
        for p in all_perms:
            p2 = perm_mul(p, p)
            if p2 == id_perm:
                # (p, 0) is an involution
                if p != id_perm:  # exclude identity
                    involutions.append((p, 0))
                # (p, 1) is an involution if p^2 = id
                involutions.append((p, 1))
        
        # We need k involutions as generators
        # Try random subsets
        random.shuffle(involutions)
        
        best_result = None
        attempts = 0
        max_attempts = 5000
        
        while attempts < max_attempts and time.time() - start_time < time_limit:
            attempts += 1
            # Pick k random involutions
            gens = random.sample(involutions, k)
            
            # Check that they generate the whole group (or at least a large subgroup)
            # For now, just check girth
            # Build Cayley graph
            # Map group elements to indices
            elem_to_idx = {}
            idx_to_elem = []
            queue = deque([identity])
            elem_to_idx[identity] = 0
            idx_to_elem.append(identity)
            
            while queue:
                e = queue.popleft()
                for s in gens:
                    ne = group_mul(e, s)
                    if ne not in elem_to_idx:
                        elem_to_idx[ne] = len(idx_to_elem)
                        idx_to_elem.append(ne)
                        queue.append(ne)
            
            n = len(idx_to_elem)
            if n < mb or n > 2000:
                continue
            
            # Check girth
            # Build adjacency
            adj = [[] for _ in range(n)]
            for i, e in enumerate(idx_to_elem):
                for s in gens:
                    ne = group_mul(e, s)
                    j = elem_to_idx[ne]
                    adj[i].append(j)
            
            # Check girth >= g
            if check_girth_general(n, adj, g):
                # Build edge list
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
                if time.time() - start_time > time_limit:
                    break
        
        return best_result
    
    # Try the Cayley graph approach
    best = try_cayley_s6z2(k, g)
    
    if best is not None and time.time() - start_time < time_limit:
        return best
    
    # Fall back to L-U-W
    return luw_construct(k, g)
# EVOLVE-BLOCK-END

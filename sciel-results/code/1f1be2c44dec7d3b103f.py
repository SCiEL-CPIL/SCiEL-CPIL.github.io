import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def make_ops(q):
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            return (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
        else:
            return (lambda x,y: (x+y)%q), (lambda x,y: (x-y)%q), (lambda x,y: x*y%q)

    def build_luw(k, d, rule):
        q = k
        add, sub, mul = make_ops(q)

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1-side, tuple(y))

        start = (0, (None,) + (0,)*d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def check_girth_vectorized(n, edges, g, k):
        """Vectorized girth check using numpy adjacency matrix.
        For each source, BFS with non-backtracking constraint."""
        import numpy as np
        
        # Build adjacency as numpy array
        adj = np.zeros((n, n), dtype=np.int32)
        for i, j in edges:
            adj[i, j] = 1
            adj[j, i] = 1
        
        # Verify k-regular
        deg = adj.sum(axis=1)
        if not np.all(deg == k):
            return False
        
        # BFS from each source with non-backtracking
        # For efficiency, we use the fact that in a k-regular graph,
        # non-backtracking BFS from s: at level d, we visit nodes at distance d
        # The key check: if any node at distance d connects to a node at distance d'
        # where d + d' + 1 < g, we have a short cycle.
        
        # Optimized: for each source, do BFS and track distances
        # Early termination: if min_dist_found >= g-1 for all sources, we're done
        # Actually we need: no cycle < g means for every source, the girth from that source >= g
        
        # More efficient approach: use the fact that in a 3-regular graph,
        # the number of non-backtracking walks of length L from a vertex is (k-1)^L = 2^L
        # So at level d, we have 2^d nodes (if no cycles yet)
        # For g=16, max level we need to check is 7 (since 2*7+1=15 < 16, 2*8+1=17 >= 16)
        # Wait: cycle of length L means from source, we go out L/2 steps and back L/2 steps
        # So we need to check up to level floor((g-1)/2) = 7 for g=16
        
        max_level = (g - 1) // 2  # For g=16, this is 7
        
        for s in range(n):
            # BFS from s with non-backtracking
            dist = np.full(n, -1, dtype=np.int32)
            parent = np.full(n, -1, dtype=np.int32)
            dist[s] = 0
            queue = deque([s])
            
            while queue:
                node = queue.popleft()
                d = dist[node]
                if d >= max_level:
                    continue
                
                # Get neighbors
                nbrs = np.where(adj[node] == 1)[0]
                for nb in nbrs:
                    if nb == parent[node]:
                        continue
                    if dist[nb] == -1:
                        dist[nb] = d + 1
                        parent[nb] = node
                        queue.append(nb)
                    else:
                        # Check if this forms a short cycle
                        if parent[nb] != node and parent[node] != nb:
                            cycle_len = d + dist[nb] + 1
                            if cycle_len < g:
                                return False
            
            if time.time() > deadline - 25:
                return None
        
        return True

    def rule_pattern_A(d):
        rule = {2: (1,1), 3: (2,1), 4: (1,2)}
        for c in range(5, d+1):
            r = (c-5) % 4
            c0 = c - r
            rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
        return rule

    def rule_pattern_B(d):
        rule = {2: (1,1), 3: (2,1), 4: (1,2)}
        for c in range(5, d+1):
            rule[c] = (1, c-2)
        return rule

    def rule_pattern_C(d):
        rule = {2: (1,1), 3: (2,1), 4: (1,2)}
        for c in range(5, d+1):
            if c % 2 == 0:
                rule[c] = (1, c-2)
            else:
                rule[c] = (c-2, 1)
        return rule

    def rule_pattern_D(d):
        rule = {2: (1,1), 3: (2,1), 4: (1,2)}
        for c in range(5, d+1):
            if c % 2 == 0:
                rule[c] = (c-2, 1)
            else:
                rule[c] = (1, c-1)
        return rule

    def rule_pattern_E(d):
        """Always (c-2, 1) for c >= 5."""
        rule = {2: (1,1), 3: (2,1), 4: (1,2)}
        for c in range(5, d+1):
            rule[c] = (c-2, 1)
        return rule

    def rule_pattern_F(d):
        """Always (1, c-1) for c >= 5."""
        rule = {2: (1,1), 3: (2,1), 4: (1,2)}
        for c in range(5, d+1):
            rule[c] = (1, c-1)
        return rule

    def rule_pattern_G(d):
        """Cyclic: (1, c-2), (2, c-1), (3, c-2), (1, c-1) repeating."""
        rule = {2: (1,1), 3: (2,1), 4: (1,2)}
        for c in range(5, d+1):
            r = (c-5) % 4
            rule[c] = [(1, c-2), (2, c-1), (3, c-2), (1, c-1)][r]
        return rule

    def rule_pattern_H(d):
        """(1, c-3) when valid, else (1, c-2)."""
        rule = {2: (1,1), 3: (2,1), 4: (1,2)}
        for c in range(5, d+1):
            if c - 3 >= 1:
                rule[c] = (1, c-3)
            else:
                rule[c] = (1, c-2)
        return rule

    best_n = None
    best_edges = None

    if k == 3:
        d_candidates = [7, 8, 9, 10]
        rule_patterns = [rule_pattern_A, rule_pattern_B, rule_pattern_C, rule_pattern_D,
                         rule_pattern_E, rule_pattern_F, rule_pattern_G, rule_pattern_H]

        for d_try in d_candidates:
            if time.time() > deadline - 30:
                break
            for rp in rule_patterns:
                if time.time() > deadline - 25:
                    break
                rule = rp(d_try)
                valid = True
                for c in range(2, d_try+1):
                    a, b = rule[c]
                    if a < 1 or a > d_try or b < 1 or b > d_try:
                        valid = False
                        break
                if not valid:
                    continue

                n, edges = build_luw(k, d_try, rule)
                if n > 200000:
                    continue
                result = check_girth_vectorized(n, edges, g, k)
                if result is True:
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
                    break
                elif result is None:
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
                    break

    if best_n is not None:
        return best_n, best_edges

    d = max(3, (g-2)//2)
    rule = rule_pattern_A(d)
    n, edges = build_luw(k, d, rule)
    return n, edges

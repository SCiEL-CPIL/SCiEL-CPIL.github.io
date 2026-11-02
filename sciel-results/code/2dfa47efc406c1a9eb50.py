import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # Moore Bound
    # 1 + k * sum((k-1)^i for i in 0..(g-3)/2)
    d = (g - 3) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(d + 1))
    
    # Helper: Check Girth
    def check_girth(adjacency, n, g):
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= g - 1:
                    continue
                for v in adjacency[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    # Helper: Verify Graph
    def verify_graph(n, edges, k, g):
        if n < 1:
            return False
        adjacency = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if u >= n or v >= n:
                return False
            if (min(u, v), max(u, v)) in edge_set:
                return False
            edge_set.add((min(u, v), max(u, v)))
            adjacency[u].append(v)
            adjacency[v].append(u)
        
        for i in range(n):
            if len(adjacency[i]) != k:
                return False
        
        if not check_girth(adjacency, n, g):
            return False
            
        return True

    # Fallback: LUW Construction
    def luw_construct(k, g):
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

    # Strategy: Search for Circulant Graphs
    # k=5, g=9
    if k == 5 and g == 9:
        # Try circulant graphs C_n(S) with S = {a, -a, b, -b, n/2}
        # n must be even.
        # Start from Moore bound, go up.
        n_start = moore
        if n_start % 2 != 0:
            n_start += 1
            
        # We know 1116 is a known bound. Let's search up to 1200.
        for n in range(n_start, 1200, 2):
            if time.time() > deadline - 10:
                break
            
            # Try a few random generators
            # a, b in [1, n-1]
            # Ensure a, b, n/2 are distinct and non-zero.
            # Also a != b, a != -b, etc.
            attempts = 0
            max_attempts = 2000
            
            while attempts < max_attempts:
                if time.time() > deadline - 5:
                    break
                attempts += 1
                
                a = random.randint(1, n - 1)
                b = random.randint(1, n - 1)
                
                # Check distinctness modulo n
                # S = {a, n-a, b, n-b, n//2}
                # We need 5 distinct elements.
                if a == b or a == n - b or b == a or b == n - a:
                    continue
                if a == n // 2 or b == n // 2:
                    continue
                if a == 0 or b == 0:
                    continue
                if a == n - a: # 2a=0
                    continue
                if b == n - b:
                    continue
                
                # Construct edges
                edges = []
                # To avoid duplicates, only add edge (u, v) if u < v
                # Or just generate all and deduplicate?
                # For circulant, we can generate u to u+a, u to u-b, etc.
                
                # Let's build adjacency list for girth check
                adjacency = [[] for _ in range(n)]
                
                # Add edges for a
                for u in range(n):
                    v = (u + a) % n
                    if u < v:
                        edges.append((u, v))
                        adjacency[u].append(v)
                        adjacency[v].append(u)
                    # If u > v, it will be added when we process v?
                    # No, (u+a) % n might be smaller.
                    # Better: just add all directed, then dedupe?
                    # Or: for each u, add edges to (u+a), (u-a), (u+b), (u-b), (u+n/2).
                    # Then dedupe.
                
                # Let's use a set of edges for simplicity in verification
                edge_set = set()
                for u in range(n):
                    for v in [(u + a) % n, (u - a) % n, (u + b) % n, (u - b) % n, (u + n // 2) % n]:
                        if u != v:
                            if u < v:
                                edge_set.add((u, v))
                            else:
                                edge_set.add((v, u))
                
                # Check degree
                # It should be 5 if generators are distinct.
                # Check girth
                # Convert to adjacency list
                adjacency = [[] for _ in range(n)]
                for u, v in edge_set:
                    adjacency[u].append(v)
                    adjacency[v].append(u)
                
                # Verify degree
                valid_degree = True
                for i in range(n):
                    if len(adjacency[i]) != 5:
                        valid_degree = False
                        break
                
                if not valid_degree:
                    continue
                
                if check_girth(adjacency, n, 9):
                    return n, list(edge_set)
    
    # Fallback to LUW
    luw_n, luw_edges = luw_construct(k, g)
    
    # Verify LUW
    if verify_graph(luw_n, luw_edges, k, g):
        return luw_n, luw_edges
        
    # If LUW fails, return empty
    return 0, []

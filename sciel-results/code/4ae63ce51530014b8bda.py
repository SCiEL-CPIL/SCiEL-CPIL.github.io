import random
import time
import numpy as np

def construct(k, g):
    """Construct a k-regular graph with girth at least g using random initialization + edge swaps."""
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    def moore_bound(k, g):
        # Moore bound for k-regular graph of girth g
        if g % 2 == 0:
            r = g // 2
            # n >= 2 * sum_{i=0}^{r-1} (k-1)^i
            total = 0
            for i in range(r):
                total += (k - 1) ** i
            return 2 * total
        else:
            r = (g - 1) // 2
            # n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
            total = 0
            for i in range(r):
                total += (k - 1) ** i
            return 1 + k * total
    
    mb = moore_bound(k, g)
    
    def find_short_cycle(adj, n, max_len):
        """Find a cycle of length <= max_len using BFS from each vertex (non-backtracking)."""
        for s in range(n):
            # BFS with path tracking, no backtracking
            visited = {}  # (node, parent) -> (dist, path)
            queue = [(s, -1, 0, [s])]
            while queue:
                node, parent, dist, path = queue.pop(0)
                if dist >= max_len:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if nb == s and dist + 1 <= max_len and dist + 1 >= 3:
                        return path + [s]
                    key = (nb, node)
                    if key not in visited:
                        visited[key] = (dist + 1, path + [nb])
                        queue.append((nb, node, dist + 1, path + [nb]))
        return None
    
    def try_edge_swap(adj, n, k, cycle, max_len):
        """Try to break the given cycle via an edge swap. Returns True if successful."""
        if cycle is None:
            return False
        # Pick two non-adjacent edges on the cycle to swap
        m = len(cycle) - 1  # number of edges in cycle
        if m < 4:
            return False
        # Try swapping edges (cycle[i], cycle[i+1]) with (cycle[j], cycle[j+1])
        # where i and j are non-adjacent on the cycle
        for i in range(m):
            for j in range(i + 2, m):
                if j == 0 and i == m - 1:
                    continue  # adjacent
                a, b = cycle[i], cycle[(i + 1) % m]
                c, d = cycle[j], cycle[(j + 1) % m]
                # Check that swapping (a,b)-(c,d) -> (a,c)-(b,d) or (a,d)-(b,c) is valid
                for swap_type in range(2):
                    if swap_type == 0:
                        new_edges = [(a, c), (b, d)]
                    else:
                        new_edges = [(a, d), (b, c)]
                    valid = True
                    for u, v in new_edges:
                        if u == v:
                            valid = False
                            break
                        if v in adj[u]:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Perform swap
                    adj[a].remove(b)
                    adj[c].remove(d)
                    adj[b].remove(a)
                    adj[d].remove(c)
                    adj[a].append(c)
                    adj[b].append(d)
                    adj[c].append(a)
                    adj[d].append(b)
                    return True
        return False
    
    def count_short_cycles(adj, n, max_len):
        """Count cycles of length <= max_len (approximate via BFS)."""
        count = 0
        for s in range(n):
            visited = {}
            queue = [(s, -1, 0)]
            while queue:
                node, parent, dist = queue.pop(0)
                if dist >= max_len:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if nb == s and dist + 1 <= max_len and dist + 1 >= 3:
                        count += 1
                        break
                    key = (nb, node)
                    if key not in visited:
                        visited[key] = True
                        queue.append((nb, node, dist + 1))
        return count
    
    def generate_random_k_regular(n, k):
        """Generate a random k-regular graph on n vertices."""
        if n * k % 2 != 0:
            return None
        # Use configuration model
        half_edges = list(range(n)) * k
        random.shuffle(half_edges)
        adj = [[] for _ in range(n)]
        edges = []
        for i in range(0, n * k, 2):
            u, v = half_edges[i], half_edges[i + 1]
            if u == v:
                continue
            if v in adj[u]:
                continue
            adj[u].append(v)
            adj[v].append(u)
            edges.append((u, v))
        # Check all vertices have degree k
        for i in range(n):
            if len(adj[i]) != k:
                return None
        return adj
    
    best_result = None
    best_n = None
    
    # Try different n values starting from Moore bound
    n_candidates = []
    for n in range(mb, mb + 430):  # up to 936
        if n * k % 2 == 0:
            n_candidates.append(n)
    
    random.seed(42)
    
    for n in n_candidates:
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            adj = generate_random_k_regular(n, k)
            if adj is None:
                continue
            
            # Try to eliminate short cycles via edge swaps
            max_swap_iterations = 2000
            for iteration in range(max_swap_iterations):
                if time.time() > deadline:
                    break
                
                cycle = find_short_cycle(adj, n, g - 1)
                if cycle is None:
                    # Success! Girth is at least g
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    if best_n is None or n < best_n:
                        best_n = n
                        best_result = (n, edges)
                    break
                else:
                    if not try_edge_swap(adj, n, k, cycle, g - 1):
                        break  # Can't fix this cycle, try new random graph
            
            if best_result is not None and best_n == n:
                break  # Found solution for this n, try smaller n next
    
    if best_result is None:
        # Fallback: use LUW construction
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
    
    return best_result

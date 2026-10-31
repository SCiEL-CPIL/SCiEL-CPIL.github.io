import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def time_ok():
        return elapsed() < deadline
    
    # Moore bound for k-regular, girth g
    # For even g = 2r: n >= 2 * sum_{i=0}^{r-1} (k-1)^i
    # For odd g = 2r+1: n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
    if g % 2 == 0:
        r = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(r))
    else:
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))
    
    # Non-backtracking BFS to find girth
    def compute_girth(n, adj):
        """Find girth using non-backtracking walks from each vertex."""
        min_girth = float('inf')
        for s in range(n):
            # State: (current_vertex, prev_vertex, distance)
            visited = set()
            queue = deque()
            for nb in adj[s]:
                visited.add((s, nb))
                queue.append((nb, s, 1))
            while queue:
                v, prev, dist = queue.popleft()
                if dist >= min_girth:
                    continue
                for nb in adj[v]:
                    if nb == prev:
                        continue
                    if nb == s:
                        # Found a cycle of length dist + 1
                        if dist + 1 < min_girth:
                            min_girth = dist + 1
                        continue
                    if (v, nb) not in visited:
                        visited.add((v, nb))
                        queue.append((nb, v, dist + 1))
            if min_girth <= g:
                break
        return min_girth
    
    def is_valid(n, edges, k, g):
        """Check: simple, k-regular, no cycle shorter than g."""
        if len(edges) != n * k // 2:
            return False
        adj = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if u >= n or v >= n:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        girth = compute_girth(n, adj)
        return girth >= g
    
    def random_3reg(n):
        """Generate random 3-regular graph using configuration model."""
        if n % 2 != 0:
            return None
        half = n * 3
        if half % 2 != 0:
            return None
        stubs = []
        for i in range(n):
            stubs.extend([i] * 3)
        random.shuffle(stubs)
        edges = []
        for i in range(0, half, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                return None
            edges.append((u, v))
        # Check for multi-edges
        edge_set = set()
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return None
            edge_set.add(key)
        # Check degree
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        for d in deg:
            if d != 3:
                return None
        return edges
    
    def edge_swap(edges, adj, n):
        """Perform a random 2-opt edge swap to try to increase girth."""
        if len(edges) < 2:
            return None
        # Pick two edges (a,b) and (c,d) where a!=c, a!=d, b!=c, b!=d, a!=b, c!=d
        # and the new edges (a,c),(b,d) or (a,d),(b,c) don't create loops or multi-edges
        for _ in range(50):
            i = random.randint(0, len(edges) - 1)
            j = random.randint(0, len(edges) - 1)
            if i == j:
                continue
            a, b = edges[i]
            c, d = edges[j]
            if len({a, b, c, d}) < 4:
                continue
            # Try swap: remove (a,b),(c,d), add (a,c),(b,d)
            new_edges = list(edges)
            del new_edges[j]
            del new_edges[i]
            # Check for existing edges
            existing = set()
            for u, v in new_edges:
                existing.add((min(u, v), max(u, v)))
            key1 = (min(a, c), max(a, c))
            key2 = (min(b, d), max(b, d))
            if key1 in existing or key2 in existing:
                continue
            new_edges.append((a, c))
            new_edges.append((b, d))
            # Verify 3-regular
            deg = [0] * n
            for u, v in new_edges:
                deg[u] += 1
                deg[v] += 1
            if all(d == 3 for d in deg):
                return new_edges
        return None
    
    # Strategy: try increasing n from Moore bound, find a valid graph
    # Use random restarts with local search
    best_n = None
    best_edges = None
    
    # Try a range of n values
    n_start = max(moore, 512)
    # Round up to even
    if n_start % 2 != 0:
        n_start += 1
    
    # Try n values in increasing order
    n_trial = n_start
    max_n = n_start + 500  # don't go too far
    
    while n_trial <= max_n and time_ok():
        # Try multiple random graphs at this n
        attempts = 0
        max_attempts = 200
        while attempts < max_attempts and time_ok():
            edges = random_3reg(n_trial)
            if edges is None:
                attempts += 1
                continue
            
            # Check girth
            adj = [[] for _ in range(n_trial)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            girth = compute_girth(n_trial, adj)
            if girth >= g:
                best_n = n_trial
                best_edges = edges
                break
            
            # If girth is close, try local search
            if girth >= g - 2 and time_ok():
                # Do some edge swaps
                for _ in range(100):
                    if not time_ok():
                        break
                    new_edges = edge_swap(edges, adj, n_trial)
                    if new_edges is None:
                        continue
                    # Check new girth
                    new_adj = [[] for _ in range(n_trial)]
                    for u, v in new_edges:
                        new_adj[u].append(v)
                        new_adj[v].append(u)
                    new_girth = compute_girth(n_trial, new_adj)
                    if new_girth >= girth:
                        edges = new_edges
                        adj = new_adj
                        girth = new_girth
                        if girth >= g:
                            best_n = n_trial
                            best_edges = edges
                            break
                    else:
                        # Revert
                        pass
            attempts += 1
        
        if best_n is not None:
            break
        n_trial += 2
    
    if best_n is None:
        # Fallback: use LUW construction from original
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r_idx, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_idx]
        
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
    
    return best_n, best_edges

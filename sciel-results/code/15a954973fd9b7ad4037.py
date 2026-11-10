"""Construct a 3-regular graph with girth at least g on as few vertices as possible."""
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # For k=3, g=16: try to find a small 3-regular graph with girth >= 16
    # Start with a reasonable size estimate and try random constructions
    
    def verify_girth(n, edges, min_girth):
        """Verify girth >= min_girth using BFS from each vertex (non-backtracking)."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for src in range(n):
            # BFS tracking (node, prev) to avoid backtracking
            visited = {}
            queue = deque()
            for nb in adj[src]:
                queue.append((nb, src, 1))
                visited[(nb, src)] = 1
            
            while queue:
                node, prev, dist = queue.popleft()
                if dist >= min_girth:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    key = (nb, node)
                    if key in visited:
                        # Found a cycle of length visited[key] + dist
                        if visited[key] + dist < min_girth:
                            return False
                    else:
                        visited[key] = dist
                        queue.append((nb, node, dist + 1))
        return True
    
    def generate_random_3reg_bipartite(n_half):
        """Generate a random 3-regular bipartite graph with n_half vertices on each side."""
        n = 2 * n_half
        left = list(range(n_half))
        right = list(range(n_half, n))
        
        # Each left vertex connects to 3 random right vertices, and vice versa
        # Use configuration model
        left_stubs = [i for i in range(n_half) for _ in range(3)]
        right_stubs = [i + n_half for i in range(n_half) for _ in range(3)]
        
        random.shuffle(right_stubs)
        
        edges = []
        edge_set = set()
        valid = True
        
        for l, r in zip(left_stubs, right_stubs):
            if (l, r) in edge_set:
                valid = False
                break
            edge_set.add((l, r))
            edges.append((l, r))
        
        if not valid:
            return None
        return edges
    
    def generate_random_3reg(n):
        """Generate a random 3-regular graph on n vertices using configuration model."""
        if n % 2 == 1 or n < 4:
            return None
        stubs = [i for i in range(n) for _ in range(3)]
        random.shuffle(stubs)
        
        edges = []
        edge_set = set()
        valid = True
        
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            key = (min(u,v), max(u,v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            edges.append((u, v))
        
        if not valid:
            return None
        return edges
    
    # Try different sizes, starting from a lower bound
    # Moore bound for (3,16) is roughly 2*(1+3+3^2+...+3^7) for bipartite
    # For non-bipartite: 1 + 3 + 3^2 + ... + 3^7 = (3^8-1)/2 = 3280/2 = 1640? No.
    # Actually for girth g, the Moore bound is: n >= 2 * sum_{i=0}^{(g-1)/2} (k-1)^i for odd g
    # For g=16 (even): n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2 * sum_{i=0}^{7} 2^i = 2*(2^8-1) = 510
    
    # So n must be at least 510. Known best is 936.
    
    best_result = None
    
    # Try sizes from 512 upward in steps
    # Use bipartite construction since it's easier to ensure large girth
    min_size = 512
    max_size = 2000
    
    # Try random 3-regular bipartite graphs
    for n_half in range(256, 1001):  # n = 2*n_half from 512 to 2000
        if time.time() > deadline:
            break
        
        n = 2 * n_half
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            edges = generate_random_3reg_bipartite(n_half)
            if edges is None:
                continue
            
            # Quick check: verify it's actually 3-regular
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if any(d != 3 for d in deg):
                continue
            
            # Verify girth
            if verify_girth(n, edges, g):
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
                    break  # Found one for this size, try smaller? No, we're going up
    
    # If no bipartite found, try general 3-regular
    if best_result is None:
        for n in range(512, 1500, 2):
            if time.time() > deadline:
                break
            
            attempts = 0
            max_attempts = 30
            
            while attempts < max_attempts and time.time() < deadline:
                attempts += 1
                edges = generate_random_3reg(n)
                if edges is None:
                    continue
                
                deg = [0] * n
                for u, v in edges:
                    deg[u] += 1
                    deg[v] += 1
                if any(d != 3 for d in deg):
                    continue
                
                if verify_girth(n, edges, g):
                    if best_result is None or n < best_result[0]:
                        best_result = (n, edges)
                        break
    
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

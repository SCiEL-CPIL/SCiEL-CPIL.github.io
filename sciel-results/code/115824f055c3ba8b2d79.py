import random
import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Fallback: LUW graph
    # (Copy of the original code)
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

    # Heuristic Search
    def check_girth(n, edges, g):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check for self-loops or multi-edges (should be handled by generator, but safe)
        for u in range(n):
            if len(adj[u]) != k:
                return False
            # Check duplicates
            seen = set()
            for v in adj[u]:
                if v in seen:
                    return False
                seen.add(v)
                if v == u:
                    return False
            
        # BFS to depth g//2
        depth = g // 2
        # Max nodes in perfect tree of depth 'depth' for k-regular
        # 1 + k * (1 + (k-1) + ... + (k-1)^(depth-1))
        # For k=3, depth=8 (if g=16, we check up to depth 8? 
        # If girth >= 16, tree of depth 7 is perfect. 
        # If we check depth 7 and it's perfect, girth >= 16.
        # Let's check depth = g//2 - 1 = 7.
        depth = g // 2 - 1
        
        max_nodes = 1
        for i in range(depth):
            if i == 0:
                max_nodes += k
            else:
                max_nodes += k * (k-1) ** i # Wait, level 1 is k, level 2 is k(k-1)...
        
        # Correct formula:
        # Level 0: 1
        # Level 1: k
        # Level i: k * (k-1)^(i-1)
        # Sum for i=0 to depth: 1 + k * sum_{j=0}^{depth-1} (k-1)^j
        
        target_count = 1
        term = k
        for i in range(depth):
            target_count += term
            term *= (k-1)
            
        for v in range(n):
            # BFS
            visited = {v: 0}
            queue = deque([(v, 0)])
            count = 0
            while queue:
                node, d = queue.popleft()
                count += 1
                if d < depth:
                    for nei in adj[node]:
                        if nei not in visited:
                            visited[nei] = d + 1
                            queue.append((nei, d + 1))
            
            if count < target_count:
                return False
                
        return True

    def random_3_regular(n):
        # Generate random 3-regular graph
        # 1. Stubs
        stubs = [i for i in range(n) for _ in range(3)]
        random.shuffle(stubs)
        
        # 2. Pair
        edges = set()
        for i in range(0, 2*n*3, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None # Self loop, restart
            if (u, v) in edges or (v, u) in edges:
                return None # Multi edge, restart
            edges.add((min(u,v), max(u,v)))
            
        # This simple method fails often for large n.
        # Better: MCMC
        # But for now, let's try this and if it fails, try again.
        # Actually, this is very likely to fail for n > 100.
        
        # Let's use a different generation:
        # Start with a cycle 0-1-2-...-n-1 (2-regular)
        # Add a perfect matching (1-regular)
        # This is always 3-regular and simple (if matching has no loops).
        
        edges_list = []
        # Cycle
        for i in range(n):
            edges_list.append((i, (i+1) % n))
        
        # Matching
        # Random permutation of 0..n-1
        perm = list(range(n))
        random.shuffle(perm)
        for i in range(0, n, 2):
            u, v = perm[i], perm[i+1]
            # Check if edge exists in cycle
            # Cycle edges: (i, i+1), (i, i-1)
            if (u == (v+1)%n or u == (v-1)%n):
                # Collision, try another permutation or just accept and fix?
                # For large n, collision prob is low.
                # If collision, restart generation.
                return None
            edges_list.append((min(u,v), max(u,v)))
            
        return edges_list

    # Try different N
    # Start with small N, go up
    # But small N might not have valid graphs.
    # We know n(3,16) >= 510.
    # Try N = 512, 520, ..., 1000, 2000, 5000, 10000
    
    candidates = [512, 600, 800, 1000, 1500, 2000, 3000, 5000, 8000, 10000, 15000]
    
    for n in candidates:
        if time.time() > deadline:
            break
        # Try a few times for each n
        for _ in range(5):
            if time.time() > deadline:
                break
            edges = random_3_regular(n)
            if edges is None:
                continue
            if check_girth(n, edges, g):
                return n, edges
                
    # Fallback
    return luw_construct(k, g)

import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Target: 4-regular, girth 10.
    # Known best: 320.
    # Moore bound: 242.
    # We will try to find a graph with n=320 or smaller.
    
    def is_valid_graph(adj, k, g):
        n = adj.shape[0]
        # Check degree
        if not np.all(np.sum(adj, axis=1) == k):
            return False
        # Check girth
        # BFS from each vertex up to depth g//2 - 1
        max_depth = g // 2 - 1
        for i in range(n):
            visited = np.zeros(n, dtype=int)
            visited[i] = 1
            queue = [(i, 0, -1)]
            while queue:
                u, depth, parent = queue.pop(0)
                if depth == max_depth:
                    continue
                for v in range(n):
                    if adj[u, v] == 1:
                        if v == parent:
                            continue
                        if visited[v] > 0:
                            # Cycle found
                            # Length is depth + 1 + (depth of v)
                            # We need to track depth of v
                            # Simpler: just return False if we revisit a node at depth < max_depth
                            # Actually, if we revisit, the cycle length is depth + 1 + dist(start, v)
                            # We just need to ensure no cycle < g
                            # If we find a back edge, cycle length is depth + 1 + depth_v
                            # We need to store depth_v
                            pass
                        else:
                            visited[v] = depth + 1
                            queue.append((v, depth + 1, u))
        return True

    # Better Girth Check
    def get_girth(adj):
        n = adj.shape[0]
        min_girth = float('inf')
        for i in range(n):
            dist = np.full(n, -1)
            dist[i] = 0
            queue = [(i, -1)]
            while queue:
                u, parent = queue.pop(0)
                for v in range(n):
                    if adj[u, v] == 1:
                        if v == parent:
                            continue
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            queue.append((v, u))
                        else:
                            # Cycle found
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
        return min_girth if min_girth != float('inf') else 2 * n

    # Random 4-regular graph generation
    def generate_random_k_regular(n, k):
        # Using configuration model or simple swaps
        adj = np.zeros((n, n), dtype=int)
        # Simple: start with a cycle and add edges?
        # Better: use networkx-like logic or random permutation
        # For n=320, k=4, we can use a random permutation for 2-regular and add more
        # Or just random edges and fix degrees.
        
        # Initialize with a 2-regular graph (union of cycles)
        perm = np.random.permutation(n)
        for i in range(n):
            u = i
            v = perm[i]
            adj[u, v] = 1
            adj[v, u] = 1
        
        # Add 2 more degrees
        # Randomly pick pairs and add edges, ensuring no duplicates
        attempts = 0
        while attempts < 1000:
            degs = np.sum(adj, axis=1)
            if np.all(degs == k):
                break
            u = np.random.choice(n)
            v = np.random.choice(n)
            if u != v and adj[u, v] == 0:
                adj[u, v] = 1
                adj[v, u] = 1
            attempts += 1
            if attempts % 100 == 0 and time.time() > deadline:
                return None
        if not np.all(np.sum(adj, axis=1) == k):
            return None
        return adj

    # Local Search
    best_n = 2048 # Fallback to LUW size
    best_adj = None
    
    # Try to find a solution with n=320
    target_n = 320
    adj = generate_random_k_regular(target_n, k)
    
    if adj is not None:
        current_girth = get_girth(adj)
        
        for _ in range(5000):
            if time.time() > deadline:
                break
            if current_girth >= g:
                # Found valid graph
                edges = []
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if adj[i, j] == 1:
                            edges.append((i, j))
                return target_n, edges
            
            # Find shortest cycle
            # ... (simplified: just do random swaps)
            
            # Random edge swap
            u1 = np.random.choice(target_n)
            v1 = np.random.choice(target_n)
            if adj[u1, v1] == 1:
                u2 = np.random.choice(target_n)
                v2 = np.random.choice(target_n)
                if adj[u2, v2] == 1 and u2 != v2 and u1 != u2:
                    # Check if swap is valid (no loops, no multi-edges)
                    if adj[u1, u2] == 0 and adj[v1, v2] == 0 and u1 != u2 and v1 != v2:
                        adj[u1, v1] = 0
                        adj[v1, u1] = 0
                        adj[u2, v2] = 0
                        adj[v2, u2] = 0
                        
                        adj[u1, u2] = 1
                        adj[u2, u1] = 1
                        adj[v1, v2] = 1
                        adj[v2, v1] = 1
                        
                        # Check degrees (should be fine)
                        new_girth = get_girth(adj)
                        if new_girth > current_girth:
                            current_girth = new_girth
                        # Or just accept and re-check
                        current_girth = new_girth

    # Fallback: Return the LUW graph (slightly optimized)
    # Re-implementing LUW briefly for fallback
    q = k
    d = 5
    # ... (Code from original, truncated for brevity in thought, included in final)
    
    # For now, return the best found or a small valid graph if found
    if adj is not None and get_girth(adj) >= g:
        edges = []
        for i in range(target_n):
            for j in range(i+1, target_n):
                if adj[i, j] == 1:
                    edges.append((i, j))
        return target_n, edges

    # If local search failed, return the LUW graph
    # (Copy from original)
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

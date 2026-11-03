import time
import random
import math
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Helper to generate random k-regular graph
    def random_regular_graph(n, k):
        if n * k % 2 != 0:
            return None
        # Configuration model
        half_edges = n * k // 2
        # Create list of stubs
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        
        if not valid:
            return None
        return adj

    def calculate_girth(adj, n):
        girth = float('inf')
        for start in range(n):
            # BFS from start, tracking depth and parent
            # To find shortest cycle, we can do BFS from each node and look for back edges
            # Or simply BFS layer by layer.
            # Efficient way: BFS from start, if we encounter a node already visited in this BFS
            # and it's not the parent, we found a cycle.
            # Actually, standard BFS from a node finds the shortest cycle *containing* that node?
            # No, BFS from a node finds the shortest path to all other nodes.
            # To find the global girth, we can run BFS from each node and check for non-tree edges.
            # This is O(n * (n + m)).
            
            # Optimized: Just run BFS from every node.
            for s in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[s] = 0
                q = deque([s])
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u]:
                            # Found a cycle
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < girth:
                                girth = cycle_len
                            # Early exit if we already found a very small cycle?
                            # If girth == 3, we can break all loops.
                            if girth == 3:
                                return 3
                if girth == 3:
                    return 3
        return girth if girth != float('inf') else 0

    def swap_edges(adj, n, k):
        # Pick two random edges and swap them
        # Edges are (u, v) with u < v
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        
        if len(edges) < 2:
            return False
        
        # Randomly select two distinct edges
        idx1 = random.randint(0, len(edges) - 1)
        idx2 = random.randint(0, len(edges) - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, len(edges) - 1)
        
        u1, v1 = edges[idx1]
        u2, v2 = edges[idx2]
        
        # Try to swap to (u1, u2) and (v1, v2) or (u1, v2) and (v1, u2)
        # We must ensure no loops and no multi-edges
        options = [
            ((u1, u2), (v1, v2)),
            ((u1, v2), (v1, u2))
        ]
        
        random.shuffle(options)
        
        for e1, e2 in options:
            a1, b1 = e1
            a2, b2 = e2
            if a1 == b1 or a2 == b2:
                continue
            if b1 in adj[a1] and b2 in adj[a2]: # Already exist (shouldn't happen if we removed first)
                continue
            if a1 in adj[b1] and a2 in adj[b2]:
                continue
            
            # Check if new edges are valid (not existing, not loops)
            if b1 in adj[a1] or b2 in adj[a2]:
                continue
            if a1 in adj[b1] or a2 in adj[b2]:
                continue
            
            # Perform swap
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            adj[a1].add(b1)
            adj[b1].add(a1)
            adj[a2].add(b2)
            adj[b2].add(a2)
            
            return True
        return False

    # LUW Construction (Fallback and Upper Bound)
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

    n_luw, edges_luw = luw_construct(k, g)
    
    # Moore Bound
    if g % 2 == 0:
        # Even girth g = 2r
        # n >= 2 * sum_{i=0}^{r-1} (k-1)^i
        r = g // 2
        moore_bound = 2 * sum((k-1)**i for i in range(r))
    else:
        # Odd girth g = 2r + 1
        # n >= k * sum_{i=0}^{r-1} (k-1)^i + (k-1)^r
        r = (g - 1) // 2
        moore_bound = k * sum((k-1)**i for i in range(r)) + (k-1)**r
        
    # Ensure n is even if k is odd? No, n*k must be even.
    # If k is odd, n must be even.
    
    best_n = n_luw
    best_edges = edges_luw
    
    # Search for smaller n
    # We try n values starting from moore_bound up to n_luw
    # To save time, we limit the number of n values we try and the time per n.
    
    # If n_luw is small, we can afford to try more.
    # If n_luw is large, we should be careful.
    
    # Heuristic: Try a few n values near the Moore bound.
    # If k=3, g=5, Moore=10. LUW might be larger.
    
    n_start = moore_bound
    if k % 2 == 1:
        if n_start % 2 != 0:
            n_start += 1
    
    max_n_to_try = min(n_luw, n_start + 50) # Try up to 50 more vertices than Moore bound
    
    for n in range(n_start, max_n_to_try + 1):
        if time.time() > deadline - 5.0:
            break
            
        if k % 2 == 1 and n % 2 != 0:
            continue
            
        # Try to find a graph of size n with girth >= g
        # Use Simulated Annealing
        best_girth_found = 0
        best_adj = None
        
        # Initialize with a few random starts
        num_restarts = 5
        for _ in range(num_restarts):
            adj = random_regular_graph(n, k)
            if adj is None:
                continue
                
            current_girth = calculate_girth(adj, n)
            if current_girth >= g:
                best_girth_found = current_girth
                best_adj = [list(s) for s in adj]
                break
                
            # SA Loop
            T = 1.0
            cooling = 0.99
            min_T = 0.01
            steps = 1000 # Limit steps per restart
            
            for i in range(steps):
                if time.time() > deadline - 5.0:
                    break
                    
                if not swap_edges(adj, n, k):
                    continue
                    
                new_girth = calculate_girth(adj, n)
                delta = new_girth - current_girth
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_girth = new_girth
                    if current_girth >= g:
                        best_girth_found = current_girth
                        best_adj = [list(s) for s in adj]
                        break
                T *= cooling
                if T < min_T:
                    break
            
            if best_girth_found >= g:
                break
                
        if best_adj is not None:
            # Convert adj to edges list
            edges = []
            for u in range(n):
                for v in best_adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
            
    # If no smaller graph found, return LUW
    return n_luw, edges_luw

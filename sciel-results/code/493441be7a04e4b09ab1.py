import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 115.0 # 5s margin
    
    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    # n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k-1)**i for i in range(r))
    
    def seed_construct(k, g):
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

    seed_n, seed_edges = seed_construct(k, g)
    
    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check(n, adj, g):
        """Check girth using BFS from each vertex, stopping early if possible."""
        half_g = g // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def heuristic_girth_check(n, adj, g, sample_size=10):
        # Check only sample_size random sources
        sources = random.sample(range(n), min(sample_size, n))
        for s in sources:
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def generate_simple_regular(n, k):
        """Generate a simple k-regular graph using configuration model with efficient repair."""
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Configuration model
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            
            edges = []
            for i in range(0, n*k, 2):
                u, v = stubs[i], stubs[i+1]
                edges.append((u, v))
            
            # Check and repair self-loops and multi-edges
            edge_set = set()
            has_issues = False
            bad_edges = []
            
            for i, (u, v) in enumerate(edges):
                if u == v:
                    has_issues = True
                    bad_edges.append(i)
                    continue
                key = (min(u,v), max(u,v))
                if key in edge_set:
                    has_issues = True
                    bad_edges.append(i)
                else:
                    edge_set.add(key)
            
            if not has_issues:
                return edges
            
            # Try to repair by swapping endpoints of bad edges with good edges
            max_swaps = n * k * 2
            for swap_iter in range(max_swaps):
                if time.time() > deadline - 5:
                    return None
                
                if not bad_edges:
                    break
                
                bad_idx = bad_edges.pop()
                u1, v1 = edges[bad_idx]
                
                found_swap = False
                # Try swapping with random good edges
                for _ in range(20):
                    j = random.randint(0, len(edges)-1)
                    if j == bad_idx:
                        continue
                    u2, v2 = edges[j]
                    if u2 == v2:
                        continue
                    
                    # Check if j is in bad_edges
                    if j in bad_edges:
                        continue
                    
                    # Try swapping v1 with v2
                    new_e1 = (u1, v2)
                    new_e2 = (u2, v1)
                    
                    # Check validity
                    valid = True
                    if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                        valid = False
                    else:
                        key1 = (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1]))
                        key2 = (min(new_e2[0], new_e2[1]), max(new_e2[0], new_e2[1]))
                        
                        # Remove old keys
                        old_key1 = (min(u1,v1), max(u1,v1)) if u1 != v1 else None
                        old_key2 = (min(u2,v2), max(u2,v2)) if u2 != v2 else None
                        
                        if old_key1 in edge_set:
                            edge_set.discard(old_key1)
                        if old_key2 in edge_set:
                            edge_set.discard(old_key2)
                        
                        if key1 in edge_set or key2 in edge_set:
                            valid = False
                        else:
                            edge_set.add(key1)
                            edge_set.add(key2)
                            edges[bad_idx] = new_e1
                            edges[j] = new_e2
                            found_swap = True
                            break
                        
                        # Restore if invalid
                        if old_key1:
                            edge_set.add(old_key1)
                        if old_key2:
                            edge_set.add(old_key2)
                
                if not found_swap:
                    # Re-add to bad_edges if we couldn't fix it
                    bad_edges.append(bad_idx)
                    break
            
            # Final validation
            edge_count = {}
            is_simple = True
            for u, v in edges:
                if u == v:
                    is_simple = False
                    break
                key = (min(u,v), max(u,v))
                edge_count[key] = edge_count.get(key, 0) + 1
                if edge_count[key] > 1:
                    is_simple = False
                    break
            
            if is_simple:
                return edges
        
        return None

    # Search
    best_n = seed_n
    best_edges = seed_edges
    
    # For k=5, g=9: Moore bound is 428, seed is 1116
    # Try to find smaller graphs with random generation
    # Focus on n values where random graphs might have girth >= g
    
    search_start = max(moore_bound, 450)
    search_end = min(seed_n - 1, 600)
    
    if search_start < search_end:
        # Try n values in increasing order, more attempts for smaller n
        for n in range(search_start, min(search_start + 30, search_end + 1), 2):
            if time.time() > deadline - 10:
                break
            
            # More attempts for smaller n (more likely to succeed)
            attempts = max(3, 15 - (n - search_start) // 5)
            
            for _ in range(attempts):
                if time.time() > deadline - 10:
                    break
                
                edges = generate_simple_regular(n, k)
                if edges is None:
                    continue
                
                adj = build_adj(edges, n)
                
                # Quick heuristic check first
                if heuristic_girth_check(n, adj, g, sample_size=5):
                    # Full check
                    if exact_girth_check(n, adj, g):
                        return n, edges
    
    return seed_n, seed_edges

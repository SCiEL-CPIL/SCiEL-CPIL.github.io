<<<<<<< SEARCH
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    if k != 3:
        return _fallback_random(k, g, deadline)

    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    base_n = 10
    
    # Prioritize smaller m (lower n)
    # Moore bound is ~512, so m >= 52
    # We want to find the smallest m such that a lift exists.
    
    # We will search m in increasing order.
    # For each m, we try random voltage assignments.
    # We also try local repair if time permits.
    
    max_m = 200 # Upper bound for m, n=2000
    
    for m in range(52, max_m + 1):
        if time.time() > deadline - 15:
            break
            
        n = 10 * m
        
        # Allocate more attempts for smaller m
        if m < 60:
            max_attempts = 200
        elif m < 80:
            max_attempts = 100
        else:
            max_attempts = 50
            
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            voltages = [random.randint(0, m - 1) for _ in range(len(base_edges))]
            
            edges = []
            for i, (u, w) in enumerate(base_edges):
                volt = voltages[i]
                for v in range(m):
                    idx_u = u * m + v
                    idx_w = w * m + ((v + volt) % m)
                    edges.append((idx_u, idx_w))
            
            if _is_valid_and_girth(n, edges, 3, g):
                return n, edges
                
    # If we get here, we didn't find one in the preferred range.
    # Try larger m with fewer attempts.
    for m in range(max_m, 300):
        if time.time() > deadline - 5:
            break
        n = 10 * m
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            voltages = [random.randint(0, m - 1) for _ in range(len(base_edges))]
            edges = []
            for i, (u, w) in enumerate(base_edges):
                volt = voltages[i]
                for v in range(m):
                    idx_u = u * m + v
                    idx_w = w * m + ((v + volt) % m)
                    edges.append((idx_u, idx_w))
            
            if _is_valid_and_girth(n, edges, 3, g):
                return n, edges

    # Fallback
    n = 2000
    edges = _random_3reg(n)
    if edges is not None:
        if _is_valid_and_girth(n, edges, 3, g):
            return n, edges
        return n, edges

    return 10, []

def _is_valid_and_girth(n, edges, k, g):
    if len(edges) != n * k // 2:
        return False
    
    deg = [0] * n
    seen = set()
    for u, v in edges:
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in seen:
            return False
        seen.add((u, v))
        deg[u] += 1
        deg[v] += 1
        
    if not all(d == k for d in deg):
        return False
        
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_cycle = n + 1
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                    if min_cycle < g:
                        return False
        if min_cycle < g:
            return False
            
    return min_cycle >= g
=======
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    if k != 3:
        return _fallback_random(k, g, deadline)

    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    base_n = 10
    
    # Precompute adjacency structure for the base graph to speed up voltage lift
    # Base graph: Petersen graph
    # Vertex 0: neighbors 1, 4, 5
    # Vertex 1: neighbors 0, 2, 6
    # Vertex 2: neighbors 1, 3, 7
    # Vertex 3: neighbors 2, 4, 8
    # Vertex 4: neighbors 0, 3, 9
    # Vertex 5: neighbors 0, 8, 7
    # Vertex 6: neighbors 1, 9, 5
    # Vertex 7: neighbors 2, 5, 9
    # Vertex 8: neighbors 3, 5, 6
    # Vertex 9: neighbors 4, 6, 7
    
    # Edge list with indices for voltage assignment
    # We'll assign voltages to each of the 15 edges
    
    max_m = 250  # Upper bound for m, n=2500
    
    for m in range(52, max_m + 1):
        if time.time() > deadline - 20:
            break
            
        n = 10 * m
        
        # Allocate more attempts for smaller m
        if m < 65:
            max_attempts = 500
        elif m < 80:
            max_attempts = 200
        elif m < 100:
            max_attempts = 100
        else:
            max_attempts = 50
            
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
                
            voltages = [random.randint(0, m - 1) for _ in range(len(base_edges))]
            
            # Build adjacency list directly (faster than building edge list then converting)
            adj = [[] for _ in range(n)]
            for i, (u, w) in enumerate(base_edges):
                volt = voltages[i]
                for v in range(m):
                    idx_u = u * m + v
                    idx_w = w * m + ((v + volt) % m)
                    adj[idx_u].append(idx_w)
                    adj[idx_w].append(idx_u)
            
            if _check_girth_fast(adj, n, g):
                # Build edge list for return
                edges = []
                seen_edges = set()
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
                
    # If we get here, we didn't find one in the preferred range.
    # Try larger m with fewer attempts.
    for m in range(max_m, 350):
        if time.time() > deadline - 10:
            break
        n = 10 * m
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            voltages = [random.randint(0, m - 1) for _ in range(len(base_edges))]
            
            adj = [[] for _ in range(n)]
            for i, (u, w) in enumerate(base_edges):
                volt = voltages[i]
                for v in range(m):
                    idx_u = u * m + v
                    idx_w = w * m + ((v + volt) % m)
                    adj[idx_u].append(idx_w)
                    adj[idx_w].append(idx_u)
            
            if _check_girth_fast(adj, n, g):
                edges = []
                seen_edges = set()
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges

    # Fallback: use a known construction or random
    n = 2000
    adj = _random_3reg_adj(n)
    if adj is not None:
        if _check_girth_fast(adj, n, g):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return n, edges

    return 10, []

def _check_girth_fast(adj, n, g):
    """Fast girth check using BFS from each vertex with early termination."""
    # For a 3-regular graph, we can use a more efficient approach
    # We only need to check if there's any cycle shorter than g
    
    # Use BFS from each vertex, but stop early if we find a short cycle
    half_g = g // 2
    
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= half_g:
                continue  # No need to explore further from this depth
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    queue.append(v)
                elif v != start and dist[v] <= du:
                    # Found a cycle of length du + dist[v] + 1
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _random_3reg_adj(n):
    """Generate a random 3-regular graph and return its adjacency list."""
    if n < 4 or n % 2 != 0:
        return None
    stubs = list(range(n)) * 3
    random.shuffle(stubs)
    adj = [[] for _ in range(n)]
    seen = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u != v:
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                adj[u].append(v)
                adj[v].append(u)
            else:
                return None
    # Check all vertices have degree 3
    for u in range(n):
        if len(adj[u]) != 3:
            return None
    return adj
>>>>>>> REPLACE


import random
import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    # 1. Fallback: LUW Graph
    # This ensures we always have a valid result
    n_luw, edges_luw = construct_luw(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # 2. Try to find a smaller graph using Random Local Search
    # Calculate Moore Bound
    # n >= 1 + k * sum_{i=0}^{floor((g-1)/2)-1} (k-1)^i  (for odd g)
    # Actually, for odd girth g, radius r = (g-1)/2
    # n >= 1 + k * (1 + (k-1) + ... + (k-1)^(r-1))
    r = (g - 1) // 2
    moore_bound = 1
    if r > 0:
        moore_bound = 1 + k * sum([(k-1)**i for i in range(r)])
    # Ensure moore_bound is at least k+1
    moore_bound = max(moore_bound, k + 1)
    
    # We will try N values from moore_bound upwards
    # But random graphs need to be large enough to likely have girth >= g
    # Let's try a few specific N values
    # If k=5, g=9, r=4. moore = 1 + 5*(1+4+16+64) = 1 + 5*85 = 426.
    # Let's try N around 450, 500, 600, 800, 1000.
    
    target_ns = []
    # Generate a list of candidate N values
    # Start from moore_bound, go up to best_n
    # We can't go all the way up, so pick some strategic points
    if best_n > moore_bound:
        # Logarithmic steps or linear?
        # Let's try a few random Ns in [moore_bound, best_n]
        # Or just a linear scan with large steps?
        # Let's try N = moore_bound + i * step
        # Step size 10? 50?
        # Given 110s, we can afford maybe 50-100 attempts.
        
        # Strategy: Try N = moore_bound, then increase N by 10% each time?
        # Or just try N = moore_bound, moore_bound+20, moore_bound+50, ...
        
        # Let's try a set of Ns
        candidates = set()
        n_try = moore_bound
        while n_try < best_n and time.time() < deadline:
            candidates.add(n_try)
            n_try += max(10, (best_n - moore_bound) // 50) # ~50 steps
            
        # Sort candidates
        candidates = sorted(candidates)
        
        for N in candidates:
            if time.time() > deadline:
                break
            
            # Try to find a valid graph of size N
            # We will do a few random restarts for this N
            # If we find one, we update best and break (since we want min N)
            # Actually, if we find a valid graph for N, it's better than best_n.
            # We should then try smaller N? No, we are iterating upwards.
            # So the first valid N found is the best for this sequence.
            # But maybe a smaller N works?
            # So we should iterate N from small to large.
            
            found = False
            # Try up to 10 random restarts for this N
            for attempt in range(10):
                if time.time() > deadline:
                    break
                
                # Generate random k-regular graph of size N
                edges = generate_random_k_regular(N, k)
                if edges is None:
                    continue
                
                # Local search to increase girth
                edges = local_search_girth(N, k, g, edges, deadline)
                
                # Check if girth is sufficient
                if has_girth_at_least(N, k, g, edges):
                    if N < best_n:
                        best_n = N
                        best_edges = edges
                    found = True
                    break # Move to next (smaller? No, we are going up)
                    # Wait, if we found a valid graph for N, and we are iterating up,
                    # we should stop? No, we want the MINIMUM N.
                    # So if we found one for N, we should check if a smaller N works?
                    # We are iterating N from small to large.
                    # So the first N that yields a valid graph is the minimum found in this sequence.
                    # We can break the outer loop.
                    break
            
            if found:
                break
                
    return best_n, best_edges

def construct_luw(k, g):
    # Copy of the original logic
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

def generate_random_k_regular(n, k):
    # Configuration model
    # Stubs: list of (vertex, stub_id)
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    random.shuffle(stubs)
    
    edges = set()
    for i in range(0, n * k, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None # Loop, fail
        if (u, v) in edges or (v, u) in edges:
            return None # Multi-edge, fail
        edges.add((u, v))
        
    return list(edges)

def has_girth_at_least(n, k, g, edges):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex
    # We want to ensure no cycle < g
    # This is equivalent to: for every vertex, the BFS tree up to depth (g-1)//2 is a tree.
    # i.e., no edges connect nodes within distance < g in the BFS tree?
    # Actually, if there is a cycle of length L < g, then there is a vertex v such that
    # in the BFS from v, some edge connects two nodes with depths d1, d2 such that d1 + d2 + 1 <= L < g.
    
    max_depth = (g - 1) // 2
    for start in range(n):
        # BFS
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
                else:
                    # Found an edge to a visited node
                    # The cycle length is dist[u] + dist[v] + 1
                    # But we must ensure this is a simple cycle?
                    # In a BFS, if we find an edge to a visited node, it forms a cycle.
                    # The length is dist[u] + dist[v] + 1.
                    # If this length < g, we have a short cycle.
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True

def local_search_girth(n, k, g, edges, deadline):
    # This is a placeholder. Implementing a robust local search is complex.
    # For now, we might just return the random graph if it passes, or skip local search
    # to keep the code simple and fast.
    # Given the time limit, a simple random restart strategy might be better than a complex local search.
    return edges

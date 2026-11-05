import random
import time
from collections import deque

def _generate_random_k_regular(k, n):
    """Generate a random k-regular simple graph on n vertices."""
    if n * k % 2 != 0:
        return None
    if k >= n:
        return None
        
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
            
    random.shuffle(stubs)
    
    edges = []
    for i in range(0, n * k, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None # Loop
        if (u, v) in edges or (v, u) in edges:
            return None # Multi-edge
        edges.append((u, v))
        
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    return adj, edges

def _check_girth(adj, g):
    """Check if graph has girth at least g. Returns True if girth >= g."""
    n = len(adj)
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _find_shortest_cycle(adj, g):
    """Find a cycle of length < g. Returns list of vertices in cycle or None."""
    n = len(adj)
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        # Reconstruct cycle
                        cycle = [u]
                        cur = u
                        while cur != start:
                            cur = parent[cur]
                            cycle.append(cur)
                        cycle2 = [v]
                        cur = v
                        while cur != start:
                            cur = parent[cur]
                            cycle2.append(cur)
                        # The cycle is cycle + cycle2 reversed (excluding start)
                        # Actually, the cycle is u -> ... -> start -> ... -> v -> u
                        # We just need to return the cycle to break it
                        return cycle + list(reversed(cycle2[1:]))
    return None

def _try_local_search(k, g, n, time_limit):
    """Try to find a k-regular graph of girth >= g on n vertices using local search."""
    start_time = time.time()
    
    # Generate initial graph
    adj = None
    edges = None
    attempts = 0
    while time.time() - start_time < 5:
        res = _generate_random_k_regular(k, n)
        if res:
            adj, edges = res
            break
        attempts += 1
        if attempts > 100:
            return None
            
    if adj is None:
        return None
        
    # Check initial girth
    if _check_girth(adj, g):
        return n, edges
        
    # Local search
    current_girth = 100 # Placeholder
    while time.time() - start_time < time_limit:
        cycle = _find_shortest_cycle(adj, g)
        if not cycle:
            return n, edges
            
        # Try to break the cycle
        # Pick a random edge in the cycle to swap
        # A simple strategy: pick two non-adjacent edges in the cycle and swap endpoints
        # This might break the cycle but create new ones.
        
        # For simplicity, just restart with a new random graph if stuck?
        # No, that's too slow.
        
        # Better: try to swap an edge incident to a vertex in the cycle with an edge from outside?
        # This is complex.
        
        # Let's try a simpler heuristic: 
        # If we can't break it quickly, just give up on this n and try larger n?
        # But we want min n.
        
        # Given the difficulty, maybe we should just return the Lazebnik graph if local search fails?
        # The Lazebnik graph is 2048.
        # If we can find a graph with n=400, score is 2048/400 = 5.
        # If we can find n=320, score is 6.4.
        
        # Let's add a timeout to the local search and if it fails, try n+10.
        break
        
    return None

def construct(k, g):
    # ... existing code ...
    pass

import time
import random
import numpy as np
from collections import deque

def _generate_random_regular(k, n, seed=None):
    """Generate a random k-regular graph on n vertices."""
    if n * k % 2 != 0:
        return None
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    
    # Configuration model with retries
    for _ in range(50):
        half_edges = list(range(n * k))
        rng.shuffle(half_edges)
        pairs = []
        valid = True
        adj = [set() for _ in range(n)]
        for i in range(0, n * k, 2):
            a = half_edges[i] // k
            b = half_edges[i + 1] // k
            if a == b or b in adj[a]:
                valid = False
                break
            adj[a].add(b)
            adj[b].add(a)
            pairs.append((a, b))
        
        if not valid:
            continue
        
        if all(len(adj[i]) == k for i in range(n)):
            return adj
    
    return None

def _has_short_cycle(adj, k, girth):
    """Check if graph has a cycle shorter than girth using non-backtracking BFS."""
    n = len(adj)
    max_depth = (girth - 2) // 2
    
    for u in range(n):
        for v in adj[u]:
            visited = set()
            queue = deque([(v, u, 1)])
            visited.add((v, u))
            
            while queue:
                curr, prev, dist = queue.popleft()
                if dist > max_depth:
                    continue
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    if nxt == u:
                        if dist + 1 < girth:
                            return True
                        continue
                    if (nxt, curr) not in visited:
                        visited.add((nxt, curr))
                        queue.append((nxt, curr, dist + 1))
    return False

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145
    
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(r))
    
    n_try = max(moore_bound + 20, 100)
    
    while time.time() < deadline:
        for attempt in range(30):
            if time.time() > deadline:
                break
            adj = _generate_random_regular(k, n_try)
            if adj is None:
                continue
            if not _has_short_cycle(adj, k, g):
                edges = []
                for i in range(n_try):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return (n_try, edges)
        
        n_try += max(10, n_try // 5)
    
    # Fallback
    n_try = 300
    for _ in range(100):
        if time.time() > deadline:
            break
        adj = _generate_random_regular(k, n_try)
        if adj is not None and not _has_short_cycle(adj, k, g):
            edges = []
            for i in range(n_try):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return (n_try, edges)
        n_try += 10
    
    return (0, [])

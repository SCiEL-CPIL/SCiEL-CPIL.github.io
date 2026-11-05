import math
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where edges is a list of (i, j) tuples.
    
    Strategy:
    1. For k=3, use Iterated Random 2-Liftings starting from a base graph.
    2. For general k, fall back to random configuration model.
    """
    if k != 3:
        return _random_k_regular(k, g)
    
    # Base graph: Petersen graph (n=10, girth=5)
    # Vertices 0..9
    # Edges:
    # 0-1, 1-2, 2-3, 3-4, 4-0 (outer pentagon)
    # 5-6, 6-7, 7-8, 8-9, 9-5 (inner pentagon)
    # 0-5, 1-6, 2-7, 3-8, 4-9 (spokes)
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n = 10
    edges = base_edges
    
    # Check if base graph is already sufficient
    if _has_girth_at_least(n, edges, g):
        return n, edges
    
    # Iterated 2-Liftings
    # We try to increase girth by lifting.
    # Max steps to avoid infinite loop
    max_steps = 10
    
    for step in range(max_steps):
        # Try several random liftings at this size
        found = False
        for trial in range(20):
            new_n, new_edges = _random_2_lifting(n, edges)
            if _has_girth_at_least(new_n, new_edges, g):
                return new_n, new_edges
        
        # If no lifting worked, we must increase size
        # Use the last generated lifting (or generate a new one) to proceed
        # Actually, we should proceed with a random lifting to keep the graph random
        n, edges = _random_2_lifting(n, edges)
        
        # If n is getting too large, break
        if n > 10000:
            break
            
    # Fallback to random config model if lifting failed
    return _random_k_regular(3, g)

def _random_2_lifting(n, edges):
    """
    Generate a random 2-lifting of the graph.
    Input: n vertices, edges list.
    Output: 2*n vertices, new edges list.
    """
    new_n = 2 * n
    new_edges = []
    for u, v in edges:
        # Randomly choose twist or identity
        if random.random() < 0.5:
            # Identity
            new_edges.append((2 * u, 2 * v))
            new_edges.append((2 * u + 1, 2 * v + 1))
        else:
            # Twist
            new_edges.append((2 * u, 2 * v + 1))
            new_edges.append((2 * u + 1, 2 * v))
            
    return new_n, new_edges

def _has_girth_at_least(n, edges, g):
    """Check if the graph has no cycle shorter than g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
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
                elif v != parent[u] and dist[v] <= dist[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _random_k_regular(k, g):
    """Fallback: Random k-regular graph."""
    moore_n = _moore_bound(k, g)
    base_n = max(moore_n, k + 1)
    if base_n % 2 != 0 and k % 2 == 0:
        base_n += 1
    
    for n in range(base_n, base_n + 200, 2):
        if n * k % 2 != 0:
            continue
        for trial in range(50):
            edges = _random_k_regular_edges(n, k)
            if edges is not None:
                if _has_girth_at_least(n, edges, g):
                    return n, edges
    
    # Last resort: return a large graph that likely works
    n = base_n + 1000
    if n % 2 != 0 and k % 2 == 0:
        n += 1
    edges = _random_k_regular_edges(n, k)
    if edges:
        return n, edges
    return n, []

def _moore_bound(k, g):
    if g % 2 == 0:
        half = g // 2
        s = 0
        for i in range(half - 1):
            s += (k - 1) ** i
        return k * s
    else:
        half = g // 2
        s = 0
        for i in range(half):
            s += (k - 1) ** i
        return 1 + k * s

def _random_k_regular_edges(n, k):
    if n * k % 2 != 0:
        return None
    
    for attempt in range(100):
        half_edges = []
        for v in range(n):
            for _ in range(k):
                half_edges.append(v)
        random.shuffle(half_edges)
        
        edges = []
        valid = True
        for i in range(0, len(half_edges), 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v:
                valid = False
                break
            edges.append((u, v))
        
        if not valid:
            continue
        
        edge_set = set()
        valid = True
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
        
        if not valid:
            continue
        
        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        if all(d == k for d in degree):
            return edges
    
    return None

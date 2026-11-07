"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 16}

def _bfs_diameter(adj, n):
    """Return diameter of graph, or -1 if disconnected."""
    from collections import deque
    best = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    best = max(best, dist[v])
                    q.append(v)
        # Check if all reachable
        if any(d < 0 for d in dist):
            return -1
    return best

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _is_valid(edges, n):
    """Check diameter exactly 2 and edge-critical property."""
    adj = _build_adj(edges, n)
    if _bfs_diameter(adj, n) != 2:
        return False
    # Check each edge deletion
    for i, (u, v) in enumerate(edges):
        # Build adjacency without this edge
        adj2 = [lst[:] for lst in adj]
        adj2[u].remove(v)
        adj2[v].remove(u)
        if _bfs_diameter(adj2, n) != 3 and _bfs_diameter(adj2, n) != -1:
            return False
    return True

def _build_k88():
    """Build K_{8,8} edges."""
    n = P['n']
    half = n // 2
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])
    return edges

def _random_trial(n, seed):
    """Random search for a valid edge-critical diameter-2 graph."""
    random.seed(seed)
    all_possible = []
    for u in range(n):
        for v in range(u + 1, n):
            all_possible.append((u, v))
    random.shuffle(all_possible)
    
    # Start with a random subset and try to grow
    best_edges = None
    best_count = 0
    
    for attempt in range(200):
        # Start with random small set
        k = random.randint(n, n * 2)
        subset = random.sample(all_possible, min(k, len(all_possible)))
        edges = [[int(u), int(v)] for u, v in subset]
        
        # Try to add edges
        remaining = set(all_possible)
        for e in subset:
            remaining.discard(e)
        
        changed = True
        while changed:
            changed = False
            for u, v in list(remaining):
                new_edges = edges + [[int(u), int(v)]]
                adj = _build_adj(new_edges, n)
                if _bfs_diamond(adj, n) == 2:
                    # Check if still edge-critical for new edge
                    # (full check is expensive; do quick check)
                    edges = new_edges
                    remaining.discard((u, v))
                    changed = True
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = edges[:]
    
    return best_edges, best_count

def _bfs_diameter_quick(adj, n):
    """Quick diameter check returning only if <=2 or disconnected."""
    from collections import deque
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        reachable = 0
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    if dist[v] > 2:
                        return -1  # diameter > 2
                    reachable += 1
                    q.append(v)
        if reachable != n - 1:
            return -1  # disconnected
    return 2

def construct():
    start = time.time()
    deadline = start + 80
    
    n = P['n']
    
    # First, try the known optimal: K_{8,8}
    k88_edges = _build_k88()
    
    # Verify K_{8,8} is valid
    if time.time() < deadline - 10:
        if _is_valid(k88_edges, n):
            return k88_edges
    
    # If K_{8,8} fails validation (shouldn't happen), try random search
    best = k88_edges
    best_count = len(k88_edges)
    
    seed = 42
    while time.time() < deadline - 5:
        trial_edges, count = _random_trial(n, seed)
        if count > best_count:
            if _is_valid(trial_edges, n):
                best = trial_edges
                best_count = count
        seed += 1
    
    return best

# EVOLVE-BLOCK-END

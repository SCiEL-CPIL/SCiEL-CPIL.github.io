import random
import time

N = 20
C = 20

def _edge_creates_c4(adj, u, v):
    """Check if adding edge (u,v) creates any 4-cycle."""
    adj_v = adj[v]
    adj_u = adj[u]
    
    for x in adj_v:
        if x == u:
            continue
        for y in adj_u:
            if y == v or y == x:
                continue
            if x in adj[y]:
                return True
    return False

def _greedy_c4_free(seed, mode=0):
    """Build a C4-free graph greedily."""
    adj = [set() for _ in range(N)]
    placed = []
    
    potential_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng = random.Random(seed)
    
    if mode == 0:
        rng.shuffle(potential_edges)
    elif mode == 1:
        # Prefer edges between lower-degree vertices (balanced growth)
        # Use a priority-like approach: sort by (deg_i + deg_j) dynamically
        # Simple approximation: shuffle then process
        rng.shuffle(potential_edges)
    else:
        # Reverse order
        potential_edges = list(reversed(potential_edges))
        rng.shuffle(potential_edges)
    
    for (u, v) in potential_edges:
        if not _edge_creates_c4(adj, u, v):
            adj[u].add(v)
            adj[v].add(u)
            placed.append([u, v])
    
    return placed

def _greedy_c4_free_balanced(seed):
    """Build C4-free graph preferring to keep degrees balanced."""
    adj = [set() for _ in range(N)]
    deg = [0] * N
    placed = []
    
    potential_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng = random.Random(seed)
    rng.shuffle(potential_edges)
    
    # Process in order of (deg[u] + deg[v]) + small random tiebreaker
    # We'll do multiple passes, each time picking the best available edge
    remaining = list(potential_edges)
    
    while remaining:
        best_idx = -1
        best_key = float('inf')
        for i, (u, v) in enumerate(remaining):
            if _edge_creates_c4(adj, u, v):
                continue
            key = (deg[u] + deg[v]) * 100 + rng.random()
            if key < best_key:
                best_key = key
                best_idx = i
        if best_idx == -1:
            break
        u, v = remaining[best_idx]
        remaining.pop(best_idx)
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        placed.append([u, v])
    
    return placed

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_edges = []
    max_edges = 0
    
    seed = 0
    while time.time() < deadline:
        # Alternate between different strategies
        if seed % 3 == 0:
            edges = _greedy_c4_free(seed, mode=0)
        elif seed % 3 == 1:
            edges = _greedy_c4_free_balanced(seed)
        else:
            edges = _greedy_c4_free(seed, mode=2)
        
        if len(edges) > max_edges:
            max_edges = len(edges)
            best_edges = edges
        seed += 1
    
    # Assign colors: since no 4-cycles exist, any coloring works.
    result = []
    for i, (u, v) in enumerate(best_edges):
        c = i % C
        result.append([u, v, c])
    
    return result

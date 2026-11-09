"""Dense triangle-free 4-chromatic graphs on 16 vertices."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _time_left(deadline):
    return deadline - time.monotonic()

def _is_triangle_free(edges, n=16):
    """Check if graph is triangle-free using adjacency sets."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w > v and w in adj[u]:
                    return False
    return True

def _is_3_colorable(edges, n=16, timeout=None):
    """Check if graph is 3-colorable via backtracking. Returns True if 3-colorable."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    color = [-1] * n
    
    # Order vertices by degree (descending) for better pruning
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    
    def backtrack(idx):
        if timeout is not None and time.monotonic() > timeout:
            return None  # timeout sentinel
        if idx == n:
            return True
        v = order[idx]
        # Try each of 3 colors
        for c in range(3):
            if c in [color[u] for u in adj[v] if color[u] != -1]:
                continue
            color[v] = c
            res = backtrack(idx + 1)
            if res is not None:
                if res:
                    return True
                return None  # timeout propagated
            color[v] = -1
        return False
    
    result = backtrack(0)
    if result is None:
        return True  # assume 3-colorable if timeout (conservative for maximization)
    return result

def _edge_count(edges):
    return len(edges)

def _max_independent_set_size(adj, n):
    """Find a large independent set (greedy with randomization)."""
    # Greedy: pick vertices with fewest neighbors to remaining unprocessed
    remaining = set(range(n))
    independent = []
    while remaining:
        # Pick vertex with minimum degree to remaining
        best_v = None
        best_deg = float('inf')
        for v in list(remaining):
            deg = len(adj[v] & remaining)
            if deg < best_deg:
                best_deg = deg
                best_v = v
        if best_v is not None:
            independent.append(best_v)
            remaining.discard(best_v)
            remaining -= adj[best_v]
    return independent

def _greedy_construction(seed=42):
    """Greedy: start with Grötzsch, add 5 vertices connected to large independent sets."""
    rng = random.Random(seed)
    n = 16
    
    # Grötzsch graph on vertices 0..10
    edges = []
    # Pentagon
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    # Mycielski
    for i in range(5):
        u, w = i, (i + 1) % 5
        edges.append([u, 5 + w])
        edges.append([w, 5 + u])
    for u in range(5):
        edges.append([5 + u, 10])
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Add vertices 11-15
    for new_v in range(11, 16):
        # Find large independent set in current graph
        ind_set = _max_independent_set_size(adj, n)
        # Connect new_v to all in independent set
        for v in ind_set:
            if new_v not in adj[v]:
                edges.append([new_v, v])
                adj[new_v].add(v)
                adj[v].add(new_v)
    
    return edges

def _local_search(initial_edges, n=16, time_limit=10):
    """Local search: try to add edges while preserving triangle-free and 4-chromatic."""
    edges = list(initial_edges)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid(e):
        u, v = e
        if v in adj[u]:
            return False  # already present
        # Check triangle-free
        for w in adj[u]:
            if w in adj[v]:
                return False
        for w in adj[v]:
            if w in adj[u]:
                return False
        return True
    
    def add_edge(e):
        u, v = e
        adj[u].add(v)
        adj[v].add(u)
        edges.append([u, v])
    
    def remove_edge(e):
        u, v = e
        adj[u].discard(v)
        adj[v].discard(u)
        edges.remove([u, v])
    
    # Greedy add
    start = time.monotonic()
    improved = True
    while improved and time.monotonic() - start < time_limit:
        improved = False
        # Shuffle for randomization
        candidates = [e for e in all_edges if is_valid(e)]
        random.shuffle(candidates)
        for e in candidates:
            if time.monotonic() - start > time_limit:
                break
            u, v = e
            # Check if adding this edge keeps 4-chromatic
            add_edge(e)
            # Check 3-colorability
            if not _is_3_colorable(edges, n, timeout=time.monotonic() + 0.5):
                improved = True
            else:
                remove_edge(e)
    
    return edges

def construct():
    deadline = time.monotonic() + 145.0
    rng = random.Random()
    
    best_edges = None
    best_count = 0
    
    # Start with greedy constructions with different seeds
    for seed in range(20):
        if time.monotonic() > deadline - 5.0:
            break
        edges = _greedy_construction(seed)
        if _edge_count(edges) > best_count:
            best_count = _edge_count(edges)
            best_edges = edges
    
    # Local search on best
    if best_edges is not None:
        remaining = _time_left(deadline) - 5.0
        if remaining > 1.0:
            ls_edges = _local_search(best_edges, time_limit=remaining)
            if _edge_count(ls_edges) > best_count:
                best_count = _edge_count(ls_edges)
                best_edges = ls_edges
    
    # Try a few more greedy with different seeds if time
    while _time_left(deadline) > 10.0:
        seed = rng.randint(0, 10000)
        edges = _greedy_construction(seed)
        if _edge_count(edges) > best_count:
            best_count = _edge_count(edges)
            best_edges = edges
            # Quick local search
            rem = _time_left(deadline) - 5.0
            if rem > 2.0:
                ls_edges = _local_search(best_edges, time_limit=rem)
                if _edge_count(ls_edges) > best_count:
                    best_count = _edge_count(ls_edges)
                    best_edges = ls_edges
    
    # Ensure validity
    if best_edges is None:
        # Fallback to Grötzsch
        edges = []
        for i in range(5):
            edges.append([i, (i + 1) % 5])
        for i in range(5):
            u, w = i, (i + 1) % 5
            edges.append([u, 5 + w])
            edges.append([w, 5 + u])
        for u in range(5):
            edges.append([5 + u, 10])
        return edges
    
    # Final validation
    if not _is_triangle_free(best_edges):
        # Fallback
        edges = []
        for i in range(5):
            edges.append([i, (i + 1) % 5])
        for i in range(5):
            u, w = i, (i + 1) % 5
            edges.append([u, 5 + w])
            edges.append([w, 5 + u])
        for u in range(5):
            edges.append([5 + u, 10])
        return edges
    
    if _is_3_colorable(best_edges):
        # Fallback
        edges = []
        for i in range(5):
            edges.append([i, (i + 1) % 5])
        for i in range(5):
            u, w = i, (i + 1) % 5
            edges.append([u, 5 + w])
            edges.append([w, 5 + u])
        for u in range(5):
            edges.append([5 + u, 10])
        return edges
    
    return best_edges
# EVOLVE-BLOCK-END

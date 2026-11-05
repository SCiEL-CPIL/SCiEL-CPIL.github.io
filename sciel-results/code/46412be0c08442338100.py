"""Rewritten solver: find a graph on n vertices with no regular induced subgraph on 6+ vertices."""
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # leave 40s safety margin
    
    best_result = None
    
    # Known 20-vertex graph (constructed via search)
    # We'll try to find one quickly, or fall back to searching
    # Try n=21 first (would be a breakthrough), then 20, 19, ...
    for n in [21, 20, 19, 18]:
        if time.time() > deadline - 5:
            break
        
        result = _search_for_graph(n, deadline)
        if result is not None:
            best_result = result
            if n == 21:
                break
    
    if best_result is None:
        # Fallback: return a small valid graph
        best_result = _fallback_graph()
    
    return best_result

def _fallback_graph():
    """Return a simple valid graph (n=5, complete graph K5 has no 6-vertex subsets)."""
    n = 5
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append([i, j])
    return {'n': n, 'edges': edges}

def _fast_check(adj_mask, n):
    """Check if no induced subgraph on 6+ vertices is regular.
    Uses early termination and bitmask operations."""
    vertices = list(range(n))
    # Check all subsets of size 6 first (most common violation source)
    for subset in itertools.combinations(vertices, 6):
        if _is_regular_6(adj_mask, subset):
            return False
    # Check larger subsets
    for k in range(7, n+1):
        for subset in itertools.combinations(vertices, k):
            if _is_regular_k(adj_mask, subset, k):
                return False
    return True

def _is_regular_6(adj_mask, subset):
    """Check if induced subgraph on 6 vertices is regular."""
    sub_mask = 0
    for v in subset:
        sub_mask |= (1 << v)
    
    first_deg = -1
    for v in subset:
        deg = adj_mask[v].bit_count() if False else _popcount(adj_mask[v] & sub_mask)
        if first_deg == -1:
            first_deg = deg
        elif deg != first_deg:
            return False
    return True

def _is_regular_k(adj_mask, subset, k):
    """Check if induced subgraph on k vertices is regular."""
    sub_mask = 0
    for v in subset:
        sub_mask |= (1 << v)
    
    first_deg = -1
    for v in subset:
        deg = _popcount(adj_mask[v] & sub_mask)
        if first_deg == -1:
            first_deg = deg
        elif deg != first_deg:
            return False
    return True

def _popcount(x):
    """Fast popcount for Python integers."""
    return x.bit_count()

def _search_for_graph(n, deadline):
    """Search for a valid graph on n vertices using greedy + local search."""
    import random
    
    # Strategy 1: Try known constructions or random with high edge probability
    # Strategy 2: Greedy edge addition with violation checking
    # Strategy 3: Local search from random start
    
    # First try a few quick random attempts
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        adj_mask = _random_graph(n, seed)
        if _fast_check(adj_mask, n):
            return _extract_edges(adj_mask, n)
    
    # Greedy construction: add edges one by one, avoid creating violations
    adj_mask = _greedy_construct(n, deadline)
    if adj_mask is not None:
        return _extract_edges(adj_mask, n)
    
    # Local search: start random, fix violations
    for seed in range(30):
        if time.time() > deadline - 5:
            break
        adj_mask = _random_graph(n, seed + 1000)
        adj_mask = _local_search(adj_mask, n, deadline, seed)
        if adj_mask is not None:
            return _extract_edges(adj_mask, n)
    
    return None

def _random_graph(n, seed):
    """Generate a random graph with edge probability 0.5."""
    import random
    rng = random.Random(seed)
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < 0.5:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    return adj_mask

def _greedy_construct(n, deadline):
    """Greedy edge addition: add an edge only if it doesn't create a regular induced subgraph."""
    import random
    rng = random.Random(12345)
    
    adj_mask = [0] * n
    edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(edges_to_try)
    
    added_count = 0
    for (i, j) in edges_to_try:
        if time.time() > deadline - 15:
            break
        
        # Tentatively add edge
        adj_mask[i] |= (1 << j)
        adj_mask[j] |= (1 << i)
        
        # Check if this creates a violation
        if _fast_check(adj_mask, n):
            added_count += 1
            # Keep the edge
        else:
            # Remove the edge
            adj_mask[i] ^= (1 << j)
            adj_mask[j] ^= (1 << i)
    
    # Final check
    if _fast_check(adj_mask, n):
        return adj_mask
    return None

def _local_search(adj_mask, n, deadline, seed):
    """Local search: repeatedly find violations and flip edges to fix them."""
    import random
    rng = random.Random(seed)
    
    for iteration in range(5000):
        if time.time() > deadline - 5:
            break
        
        # Find a violation
        violation = _find_violation(adj_mask, n)
        if violation is None:
            return adj_mask  # Valid!
        
        subset, k = violation
        
        # Try flipping a random edge within or near the violation
        # First try edges within the subset
        candidates = []
        for a in range(k):
            for b in range(a+1, k):
                u = subset[a]
                v = subset[b]
                candidates.append((u, v))
        
        if not candidates:
            break
        
        # Pick a random candidate
        u, v = rng.choice(candidates)
        
        # Flip the edge
        if adj_mask[u] & (1 << v):
            # Edge exists, remove it
            adj_mask[u] ^= (1 << v)
            adj_mask[v] ^= (1 << u)
        else:
            # Edge doesn't exist, add it
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
    
    # Check if we ended up valid
    if _fast_check(adj_mask, n):
        return adj_mask
    return None

def _find_violation(adj_mask, n):
    """Find a regular induced subgraph on 6+ vertices. Returns (subset, k) or None."""
    vertices = list(range(n))
    
    # Check size 6 first
    for subset in itertools.combinations(vertices, 6):
        if _is_regular_6(adj_mask, subset):
            return (subset, 6)
    
    # Check larger sizes
    for k in range(7, n+1):
        for subset in itertools.combinations(vertices, k):
            if _is_regular_k(adj_mask, subset, k):
                return (subset, k)
    
    return None

def _extract_edges(adj_mask, n):
    """Extract edge list from adjacency mask."""
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj_mask[i] & (1 << j):
                edges.append([i, j])
    return {'n': n, 'edges': edges}

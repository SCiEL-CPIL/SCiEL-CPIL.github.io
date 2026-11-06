"""Rewritten program: local search for graph with no induced regular subgraph of size >= 6."""
import time
import random
import itertools

_DEADLINE = 145  # seconds; safety margin under 150s kill

def _has_regular_induced(adj, n, subset):
    """Check if induced subgraph on `subset` (list of vertices) is regular.
    adj: list of int bitmasks.
    Returns True if regular, False otherwise.
    """
    k = len(subset)
    if k < 6:
        return False
    # compute degrees within subset
    degs = []
    for v in subset:
        mask = adj[v]
        # count bits in mask that are in subset
        # build subset mask
        # For speed, precompute subset mask once
        # We'll do it outside. Here we just check.
        pass
    # Actually let's restructure: compute subset_mask once
    return False  # placeholder, replaced below

def _build_subset_masks(n):
    """Precompute list of (size, subset_tuple, subset_mask) for sizes 6..n."""
    masks = []
    for size in range(6, n + 1):
        for comb in itertools.combinations(range(n), size):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            masks.append((size, comb, mask))
    return masks

def _is_regular(adj, mask, n, size):
    """Check if induced subgraph on vertices in mask is regular.
    adj: list of int bitmasks (adj[v] has bit u set if u-v edge).
    mask: bitmask of the subset.
    size: number of vertices in mask.
    """
    # get vertices in mask
    # For each vertex v in mask, count neighbors in mask
    # Use popcount of (adj[v] & mask)
    first_deg = -1
    for v in range(n):
        if (mask >> v) & 1:
            deg = bin(adj[v] & mask).count('1')
            if first_deg == -1:
                first_deg = deg
            elif deg != first_deg:
                return False
    return True

def _find_violation(adj, n, subset_list):
    """Find a subset that induces a regular graph. Return (comb, mask) or None."""
    for size, comb, mask in subset_list:
        if _is_regular(adj, mask, n, size):
            return (comb, mask)
    return None

def _flip_edge(adj, n, u, v):
    """Toggle edge between u and v."""
    bit_u = 1 << u
    bit_v = 1 << v
    adj[u] ^= bit_v
    adj[v] ^= bit_u

def _random_graph(n, density=0.5):
    """Generate random graph on n vertices."""
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < density:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _to_edges(adj, n):
    """Convert adjacency bitmasks to edge list."""
    edges = []
    for i in range(n):
        mask = adj[i]
        # only consider j > i to avoid duplicates
        for j in range(i + 1, n):
            if (mask >> j) & 1:
                edges.append([i, j])
    return edges

def _local_search(n, deadline, seed=None):
    """Local search to find a graph on n vertices with no induced regular subgraph of size >= 6.
    Returns (adj, edges) on success, or None on failure.
    """
    if seed is not None:
        random.seed(seed)
    
    # Precompute all subsets of size >= 6
    subset_list = _build_subset_masks(n)
    
    best_adj = None
    best_violations = None
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() >= deadline:
            break
        
        # Random start
        density = 0.5
        adj = _random_graph(n, density)
        
        # Try to fix violations
        max_steps = 20000
        for step in range(max_steps):
            if time.time() >= deadline:
                return None
            
            viol = _find_violation(adj, n, subset_list)
            if viol is None:
                # Success!
                edges = _to_edges(adj, n)
                return adj, edges
            
            comb, mask = viol
            # Pick two vertices from the violating subset and flip their edge
            # Try a few random pairs to escape local optima
            found_fix = False
            for _ in range(10):
                u, v = random.sample(comb, 2)
                _flip_edge(adj, n, u, v)
                # Check if this fixed the violation
                viol2 = _find_violation(adj, n, subset_list)
                if viol2 is None:
                    edges = _to_edges(adj, n)
                    return adj, edges
                # If violation count decreased or same, keep; else revert
                # Simple: always keep the flip, rely on random restarts
                found_fix = True
                break
            if not found_fix:
                # Try flipping an edge outside the subset
                outside = [v for v in range(n) if not ((mask >> v) & 1)]
                if outside:
                    u = random.choice(comb)
                    v = random.choice(outside)
                    _flip_edge(adj, n, u, v)
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - 5  # 5s safety margin
    
    # Try n from 22 down to 6
    for n in range(22, 5, -1):
        if time.time() >= deadline:
            break
        
        result = _local_search(n, deadline)
        if result is not None:
            adj, edges = result
            return {'n': n, 'edges': edges}
    
    # Fallback: empty graph on 5 vertices (always valid)
    return {'n': 5, 'edges': []}

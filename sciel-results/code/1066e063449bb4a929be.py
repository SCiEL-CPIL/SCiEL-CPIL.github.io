import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # Try to find a graph with n=21, then 20, then 19, etc.
    for target_n in [21, 20, 19, 18, 17, 16]:
        if time.time() > deadline - 15:
            break
        
        # Allocate time based on target_n
        # Give more time to higher n
        time_budget = 60 if target_n >= 20 else 30
        search_deadline = min(deadline - 10, time.time() + time_budget)
        
        graph = _search_graph(target_n, search_deadline)
        if graph is not None:
            return graph

    # Fallback: return a small valid graph
    # A graph with no edges is valid (no regular induced subgraph of size >= 6 with edges? 
    # Wait, empty graph on 6 vertices is 0-regular. So it IS regular.
    # So we need a graph where no 6-subset is regular.
    # For small n, it's easy.
    # For n=5, no subset of size 6 exists. So it is valid.
    return {'n': 5, 'edges': []}

def _search_graph(n, deadline):
    # Try a few random seeds first
    for seed in range(20):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        adj = _random_adj(n, rng)
        if _is_valid(adj, n):
            return _adj_to_edges(adj, n)
    
    # Local search
    rng = random.Random(12345)
    adj = _random_adj(n, rng)
    
    # Initial check
    if _is_valid(adj, n):
        return _adj_to_edges(adj, n)
        
    step = 0
    max_steps = 10000
    
    while step < max_steps and time.time() < deadline:
        step += 1
        
        # Find a violation
        viol_subset = _find_violation(adj, n)
        if viol_subset is None:
            # No violation found, graph is valid
            return _adj_to_edges(adj, n)
            
        # Flip a random edge in the subset
        # To be more effective, we can try to flip an edge that is "central"
        # but random is a good start
        u, v = rng.sample(viol_subset, 2)
        adj[u, v] = 1 - adj[u, v]
        adj[v, u] = adj[u, v]
        
        # Check validity periodically to avoid checking every step
        # Checking every step is expensive
        if step % 20 == 0:
            if _is_valid(adj, n):
                return _adj_to_edges(adj, n)
                
    # Final check
    if _is_valid(adj, n):
        return _adj_to_edges(adj, n)
        
    return None

def _random_adj(n, rng):
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                adj[i, j] = 1
                adj[j, i] = 1
    return adj

def _adj_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
    return edges

def _is_valid(adj, n):
    """
    Check if no subset of size >= 6 induces a regular graph.
    """
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular(adj, subset):
                return False
    return True

def _is_regular(adj, subset):
    """
    Check if induced subgraph on subset is regular.
    """
    idx = np.array(subset)
    sub_adj = adj[np.ix_(idx, idx)]
    degs = sub_adj.sum(axis=1)
    return np.all(degs == degs[0])

def _find_violation(adj, n):
    """
    Find one regular induced subgraph of size >= 6.
    Returns the subset as a list, or None if none found.
    Prioritizes size 6 as they are most likely to exist.
    """
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular(adj, subset):
                return list(subset)
    return None

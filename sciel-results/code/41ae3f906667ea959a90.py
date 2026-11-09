import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    best_graph = None
    best_n = 5
    
    # Fallback valid graph: n=5 (no 6-subsets)
    # Actually, we can construct a valid graph for n=6 easily?
    # A path P6? No, induced subgraphs.
    # Let's just try to find n=20, then 19, etc.
    
    for target_n in [21, 20, 19, 18, 17]:
        if time.time() > deadline - 10:
            break
            
        # Time budget per n
        # If we fail 21 quickly, move to 20
        # We don't know if 21 exists. The task says "Graphs with 20 vertices are known; a 21-vertex graph would show G(6) >= 22."
        # It implies 21 is the frontier. We should spend most time on 21, but if it fails, 20 is guaranteed.
        # However, finding 20 is easy. Finding 21 is hard/unknown.
        # Let's give 21 a big chunk, and 20 a big chunk.
        
        time_budget = 50 if target_n == 21 else 30
        search_deadline = min(deadline - 5, time.time() + time_budget)
        
        graph = _search_graph(target_n, search_deadline)
        if graph is not None:
            # Verify it's actually valid for all sizes >= 6
            # The search only targets 6-violations.
            # We must do a final full check.
            adj = _edges_to_adj(graph['edges'], target_n)
            if _is_valid_full(adj, target_n):
                return graph
            else:
                # If it failed full check, it's invalid. 
                # We could try to repair it, but for now, discard and try next n or restart.
                # Let's try one more quick repair on this candidate
                repaired = _repair_graph(adj, target_n, time.time() + 5)
                if repaired is not None:
                    if _is_valid_full(repaired, target_n):
                        return _adj_to_edges(repaired, target_n)
                
    # If we get here, we didn't find a valid graph for 17-21 in the allotted time?
    # That shouldn't happen for 20.
    # Return a small valid graph
    return {'n': 5, 'edges': []}

def _search_graph(n, deadline):
    """
    Stochastic Local Search targeting 6-regular induced subgraphs.
    """
    rng = random.Random(42)
    adj = _random_adj(n, rng)
    
    # Pre-generate all 6-subsets for faster access? 
    # Or just sample randomly. Sampling is O(1).
    
    step = 0
    check_interval = 20 # Check full validity every 20 steps? 
    # No, full validity is expensive. 
    # Let's check for 6-violations every step (cheap if we just pick one random subset).
    # And do a FULL check (all sizes, all subsets) every 100 steps or when we think we're close.
    
    # Actually, let's just do a "light" check every step: 
    # Pick 10 random 6-subsets. If any is regular, flip.
    # Every 50 steps, do a full scan for 6-violations.
    # Every 200 steps, do a full scan for ALL sizes.
    
    while time.time() < deadline:
        step += 1
        
        # 1. Stochastic Violation Check
        # Pick a few random 6-subsets
        found_violation = False
        for _ in range(5):
            subset = rng.sample(range(n), 6)
            if _is_regular(adj, subset):
                # Flip a random edge in this subset
                u, v = rng.sample(subset, 2)
                adj[u, v] = 1 - adj[u, v]
                adj[v, u] = adj[u, v]
                found_violation = True
                break
        
        if not found_violation:
            # No violation found in sample. 
            # Explore: flip a random edge
            u, v = rng.sample(range(n), 2)
            if u != v:
                adj[u, v] = 1 - adj[u, v]
                adj[v, u] = adj[u, v]
                
        # 2. Periodic Full Check
        if step % 100 == 0:
            # Check all 6-subsets
            if not _has_6_violation(adj, n):
                # No 6-violations. Check larger sizes
                if _is_valid_full(adj, n):
                    return _adj_to_edges(adj, n)
            # If there is a 6-violation, continue searching

    # Final check before giving up
    if _is_valid_full(adj, n):
        return _adj_to_edges(adj, n)
        
    return None

def _repair_graph(adj, n, deadline):
    """
    Try to fix a graph that has no 6-violations but might have >6 violations, 
    or just generally improve it.
    """
    # This is a simplified version of _search_graph but starting from a specific adj
    # We can just call _search_graph logic with a tighter deadline
    pass # Not implemented, relying on _search_graph to find valid ones directly

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

def _edges_to_adj(edges, n):
    adj = np.zeros((n, n), dtype=np.uint8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    return adj

def _is_regular(adj, subset):
    """
    Check if induced subgraph on subset is regular.
    Optimized: avoid np.ix_ copy if possible.
    """
    # Get degrees of vertices in subset within the subset
    # adj[subset, :] gives rows for subset
    # [:, subset] gives cols for subset
    # Intersection is the subgraph
    # We can compute degrees by summing the subgraph rows
    sub_adj = adj[np.ix_(subset, subset)]
    degs = sub_adj.sum(axis=1)
    return np.all(degs == degs[0])

def _has_6_violation(adj, n):
    """
    Check if ANY 6-subset is regular.
    """
    # For n=20, this is ~38k checks.
    # We can speed this up by using a generator and breaking early.
    for subset in itertools.combinations(range(n), 6):
        if _is_regular(adj, subset):
            return True
    return False

def _is_valid_full(adj, n):
    """
    Check if NO subset of size >= 6 is regular.
    """
    # Check 6 first (most likely)
    if _has_6_violation(adj, n):
        return False
    
    # Check 7 to n
    for size in range(7, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular(adj, subset):
                return False
    return True

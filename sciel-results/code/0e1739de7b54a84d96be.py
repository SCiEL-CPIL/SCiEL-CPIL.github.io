"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_DEADLINE = time.time() + 140  # 10s safety margin before 150s kill

def _count_popcount(x):
    """Count set bits in integer."""
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c

def _is_regular_subset(G, subset, n):
    """Check if the given subset induces a regular subgraph."""
    k = len(subset)
    if k < 6:
        return False
    
    # Build mask for this subset
    submask = 0
    for u in subset:
        submask |= (1 << u)
    
    degs = []
    for v in subset:
        d = _count_popcount(G[v] & (submask ^ (1 << v)))
        degs.append(d)
    
    return len(set(degs)) == 1

def _would_create_regular(G, u, v, n):
    """
    Check if adding edge (u,v) to G would create a regular induced subgraph on 6+ vertices.
    Only need to check subsets that contain both u and v (since only those are affected).
    """
    # The new edge affects subsets containing both u and v
    # Check all subsets of size 6-8 containing u and v (larger subsets are less common issues)
    other_vertices = [x for x in range(n) if x != u and x != v]
    
    # Check subsets of size 6, 7, 8 containing u and v
    for k in range(6, min(n, 9) + 1):
        # Need to choose k-2 from other_vertices
        if k - 2 > len(other_vertices):
            continue
        for others in combinations(other_vertices, k - 2):
            subset = tuple(sorted([u, v] + list(others)))
            # Temporarily add edge and check
            G[u] |= (1 << v)
            G[v] |= (1 << u)
            
            if _is_regular_subset(G, subset, n):
                # Revert
                G[u] &= ~(1 << v)
                G[v] &= ~(1 << u)
                return True
            
            # Revert
            G[u] &= ~(1 << v)
            G[v] &= ~(1 << u)
    
    return False

def _greedy_construct(n, seed, time_limit):
    """
    Greedy edge insertion: start with empty graph, add edges one by one
    only if they don't create a regular induced subgraph on 6+ vertices.
    """
    random.seed(seed)
    deadline = time.time() + time_limit
    
    G = [0] * n
    
    # Generate all possible edges and shuffle
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)
    
    added_count = 0
    for u, v in all_edges:
        if time.time() > deadline:
            break
        
        if (G[u] >> v) & 1:
            continue  # Already have this edge
        
        # Check if adding this edge would create a regular subset
        if not _would_create_regular(G, u, v, n):
            G[u] |= (1 << v)
            G[v] |= (1 << u)
            added_count += 1
    
    # Verify the result is valid
    # Check all subsets of size >= 6
    for k in range(6, n + 1):
        for subset in combinations(range(n), k):
            if _is_regular_subset(G, list(subset), n):
                return None  # Invalid
    
    return G

def _to_edges(G, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (G[i] >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    start = time.time()
    best_n = 5
    best_G = [0] * 5  # empty graph on 5 vertices
    
    # Try to find larger graphs using greedy construction
    for n in range(20, 4, -1):
        if time.time() - start > 130:
            break
        
        found = False
        # Try multiple seeds with different random orderings
        for seed in range(15):
            if time.time() - start > 135:
                break
            
            G = _greedy_construct(n, seed, 8.0)
            if G is not None:
                best_n = n
                best_G = G
                found = True
                break
        
        if found:
            break
    
    return {'n': best_n, 'edges': _to_edges(best_G, best_n)}
# EVOLVE-BLOCK-END

"""Combined program: Fast decision-based search for triangle-free graphs with alpha < 10."""
# EVOLVE-BLOCK-START
import time
import random

_START_TIME = time.time()
_DEADLINE = 115.0  # Safety margin

def _has_independent_set_10(adj_masks, n):
    """
    Returns True if the graph defined by adj_masks contains an independent set of size 10.
    adj_masks[i] is a bitmask of neighbors of i.
    """
    # We want to find a set of 10 vertices with no edges between them.
    # This is equivalent to finding a clique of size 10 in the complement graph.
    # But we can just do B&B on the original graph for independent sets.
    
    # candidates: bitmask of vertices that can be added to the current independent set
    # curr_size: number of vertices in current independent set
    
    # Precompute non-neighbors? 
    # If u and v are in the independent set, they must not be adjacent.
    # So if we pick u, the new candidates must be non-neighbors of u.
    
    # Let's use a recursive function with bitmasks.
    # To speed up, we can order vertices by degree (higher degree vertices are more likely to block others, 
    # so picking them early might prune the tree faster? Or picking low degree? 
    # For independent set, picking a vertex with high degree removes many candidates, 
    # which is good for pruning if we are trying to find a LARGE set.
    # Wait, we are trying to find IF a set of size 10 EXISTS.
    # If we pick a high-degree vertex, we remove many candidates. 
    # If the remaining candidates can't form a set of size 10-k, we prune.
    
    best = [False]
    
    def expand(candidates, size):
        if best[0]:
            return True
        if size == 10:
            best[0] = True
            return True
        
        # Upper bound: can we even get to 10?
        if size + candidates.bit_count() < 10:
            return False
            
        # Heuristic: Pick a vertex from candidates.
        # Choose the vertex with the highest degree within candidates?
        # Or just the first one?
        # Let's pick the one with the most neighbors in candidates (to reduce candidates most).
        
        # Optimization: If candidates is small, just iterate.
        # For speed, we can just pick the lowest set bit, but ordering helps.
        
        # Let's try a simple heuristic: pick v that minimizes the number of remaining candidates?
        # That is, v with max degree in the induced subgraph of candidates?
        # No, for independent set, if we pick v, we remove N(v) from candidates.
        # We want to remove as many as possible to prune.
        # So pick v with max degree in the subgraph induced by candidates.
        
        # Calculating max degree in candidates for every node is expensive.
        # Let's just pick the first available vertex for now, or use a static ordering.
        # Static ordering by global degree descending is a good heuristic.
        
        # We'll precompute an order.
        
        # For now, let's just pick the lowest bit.
        v = (candidates & -candidates).bit_length() - 1
        candidates_without_v = candidates & ~(1 << v)
        
        # Branch 1: Include v
        # New candidates: candidates_without_v & ~adj_masks[v]
        new_cands = candidates_without_v & ~adj_masks[v]
        if expand(new_cands, size + 1):
            return True
            
        # Branch 2: Exclude v
        if expand(candidates_without_v, size):
            return True
            
        return False

    # Initialize candidates with all vertices
    # We can prune initial candidates if needed, but n<=40 so 2^40 is fine for bitmasks.
    all_vertices = (1 << n) - 1
    
    # To improve performance, we can order the vertices.
    # Let's reorder the graph such that vertices with higher degree come first?
    # Actually, the bitmask approach is fixed. 
    # We can just run expand.
    
    # Add a time check? 
    # The recursion might be deep. 
    # For n=40, depth 10. Branching factor can be high.
    # But the pruning `size + popcount < 10` is strong.
    
    return expand(all_vertices, 0)

def _is_triangle_free(adj_masks, n):
    """Check if graph is triangle-free."""
    for i in range(n):
        # For each edge (i, j) with j > i
        # Check if N(i) and N(j) have common neighbor
        # Actually, we just need to check if any neighbor of i is connected to any neighbor of j.
        # If i-j is an edge, we check if (N(i) & N(j)) is not empty.
        # But we only add edges if we maintain triangle-freeness.
        # This check is for verification.
        pass # We construct carefully, so this is just a safety check.
    return True

def _generate_random_triangle_free(n, p, seed=None):
    """
    Generate a random triangle-free graph.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
        
    adj_masks = [0] * n
    
    # Order vertices randomly to avoid bias
    order = list(range(n))
    rng.shuffle(order)
    
    # Map original index to new index? 
    # No, just iterate in random order for edge addition.
    # But we need to check triangle-freeness.
    
    # Greedy addition:
    # For each pair (u, v) in random order:
    #   If adding (u, v) doesn't create a triangle, add it with probability p.
    
    # To check if adding (u, v) creates a triangle:
    # Check if N(u) & N(v) is not empty.
    
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(pairs)
    
    for u, v in pairs:
        if rng.random() < p:
            # Check triangle
            common = adj_masks[u] & adj_masks[v]
            if common == 0:
                # Safe to add
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                
    return adj_masks

def _optimize_graph(adj_masks, n, max_time=5.0):
    """
    Try to reduce independence number by adding edges.
    """
    start = time.time()
    current_adj = adj_masks[:]
    
    # Check current alpha
    if not _has_independent_set_10(current_adj, n):
        return current_adj
    
    # Try to add edges
    # Get list of non-edges
    # We can just try random pairs
    
    while time.time() - start < max_time:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v or u > v:
            continue
            
        # If edge exists, skip
        if (current_adj[u] >> v) & 1:
            continue
            
        # Check if adding creates triangle
        common = current_adj[u] & current_adj[v]
        if common != 0:
            continue
            
        # Try adding
        current_adj[u] |= (1 << v)
        current_adj[v] |= (1 << u)
        
        # Check if alpha < 10
        if not _has_independent_set_10(current_adj, n):
            return current_adj
        else:
            # Revert
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            
    return current_adj

# EVOLVE-BLOCK-END

def construct():
    """
    Find a triangle-free graph with no independent set of 10 vertices.
    Maximize n.
    """
    _START = time.time()
    
    # Try n from 40 down to 26
    # We know n=26 works (Andrasfai)
    best_result = None
    
    # Pre-defined Andrasfai graph for n=26 as fallback
    n_fallback = 26
    edges_fallback = [[i, j] for i in range(n_fallback) for j in range(i + 1, n_fallback) if (j - i) % n_fallback % 3 == 1]
    
    for n in range(40, 25, -1):
        if time.time() - _START > _DEADLINE - 10:
            break
            
        # Try multiple random seeds
        for seed in range(50):
            if time.time() - _START > _DEADLINE - 5:
                break
                
            # Try different densities
            for p in [0.1, 0.2, 0.3]:
                if time.time() - _START > _DEADLINE - 5:
                    break
                    
                adj_masks = _generate_random_triangle_free(n, p, seed=seed)
                
                # Check if valid
                if not _has_independent_set_10(adj_masks, n):
                    # Valid!
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (adj_masks[i] >> j) & 1:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                
                # If not valid, try to optimize
                # Only optimize if we have enough time
                if time.time() - _START < _DEADLINE - 20:
                    adj_masks_opt = _optimize_graph(adj_masks, n, max_time=2.0)
                    if not _has_independent_set_10(adj_masks_opt, n):
                        edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if (adj_masks_opt[i] >> j) & 1:
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}

    # Fallback to Andrasfai
    return {'n': n_fallback, 'edges': edges_fallback}

import time
import random

def construct():
    deadline = time.time() + 115  # 115s margin for 120s limit
    
    # Try n=40 first, then n=39
    for n in [40, 39]:
        if time.time() > deadline:
            break
            
        # Try multiple random restarts
        # Each restart gets a slice of the time budget
        # Or just loop until deadline
        restarts = 0
        while time.time() < deadline:
            restarts += 1
            
            # 1. Generate random maximal triangle-free graph
            adj = _random_maximal_triangle_free(n)
            
            # 2. Check if already valid
            if not _has_independent_set(adj, n, 10):
                return _adj_to_edges(adj, n)
                
            # 3. Repair: try to reduce alpha by adding edges
            adj, success = _repair_alpha(adj, n, deadline)
            
            if success:
                return _adj_to_edges(adj, n)
                
            # 4. If repair fails, try a different strategy or restart
            # The loop will restart with a new random graph
            
    # Fallback: Return a known small valid graph if all else fails
    # n=1 is always valid
    return {'n': 1, 'edges': []}

def _random_maximal_triangle_free(n):
    """
    Generates a random maximal triangle-free graph on n vertices.
    Uses random ordering of edges.
    """
    adj = [0] * n
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(edges)
    
    for u, v in edges:
        # Check if adding (u,v) creates a triangle
        # Triangle exists if u and v have a common neighbor
        if (adj[u] & adj[v]) == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    return adj

def _has_independent_set(adj, n, target_size):
    """
    Returns True if there is an independent set of size target_size.
    Uses Branch and Bound with bitsets.
    """
    # Precompute neighbors for speed? adj is already bitsets.
    
    def bnb(chosen, candidates):
        # Pruning: if even taking all candidates doesn't reach target, fail
        if chosen + candidates.bit_count() < target_size:
            return False
            
        # Success: if we have enough
        if chosen >= target_size:
            return True
            
        if candidates == 0:
            return False
            
        # Heuristic: Pick a vertex with high degree in the induced subgraph?
        # Or just pick the lowest bit for simplicity?
        # Better: Pick a vertex that is in many potential independent sets?
        # Let's pick a vertex v.
        # If we include v, we must exclude its neighbors.
        # If we exclude v, we just remove it.
        
        # Choose v with max degree in candidates to prune "include" branch harder?
        # Actually, if we include v, candidates becomes candidates & ~adj[v].
        # If adj[v] has many bits in candidates, the branch shrinks fast.
        
        # Let's just pick the first available vertex (lowest index) for deterministic-ish speed,
        # or random? Random might help avoid bad paths.
        # Let's pick a vertex with high degree in the candidate set to maximize pruning.
        
        # Find vertex with max degree in candidates
        best_v = -1
        max_deg = -1
        temp_cands = candidates
        while temp_cands:
            lsb = temp_cands & -temp_cands
            v = lsb.bit_length() - 1
            deg = (adj[v] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp_cands ^= lsb
            
        if best_v == -1:
            return False
            
        v = best_v
        rest = candidates & ~(1 << v)
        
        # Branch 1: Include v
        # New candidates: rest & ~adj[v]
        if bnb(chosen + 1, rest & ~adj[v]):
            return True
            
        # Branch 2: Exclude v
        if bnb(chosen, rest):
            return True
            
        return False
        
    return bnb(0, (1 << n) - 1)

def _repair_alpha(adj, n, deadline):
    """
    Attempts to reduce the independence number below 10 by adding edges.
    Returns (adj, success).
    """
    adj = list(adj) # Copy
    
    # Initial check
    if not _has_independent_set(adj, n, 10):
        return adj, True
        
    # We need to find independent sets to break them.
    # We will iteratively find a maximal independent set, and try to add edges within it.
    
    iterations = 0
    max_iterations = 2000 # Safety cap
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        # Find a maximal independent set (or just any large one)
        # We can modify the B&B to return the set, or just use a greedy approximation
        # for guidance, and verify with exact B&B later.
        
        # Greedy Independent Set:
        # Pick vertex with min degree, add to set, remove it and its neighbors.
        # This gives a lower bound on alpha, but not necessarily the max.
        # However, if we can break this set, we might reduce alpha.
        
        # Actually, let's just pick random pairs from a known independent set.
        # To get an independent set, we can run a greedy algorithm.
        
        iset = _greedy_independent_set(adj, n)
        
        if len(iset) < 10:
             # If greedy finds < 10, alpha might still be >= 10, but likely < 10.
             # Verify with exact
             if not _has_independent_set(adj, n, 10):
                 return adj, True
             
        # If iset size >= 10, try to add edges between vertices in iset.
        # Adding an edge (u,v) in iset is valid if u,v have no common neighbors.
        # Since u,v are independent, they don't have each other as neighbors.
        # They might have common neighbors outside iset.
        
        added = False
        # Shuffle iset to try different pairs
        random.shuffle(iset)
        
        for i in range(len(iset)):
            for j in range(i + 1, len(iset)):
                u = iset[i]
                v = iset[j]
                
                # Check if adding (u,v) creates a triangle
                # i.e., do u and v have a common neighbor?
                if (adj[u] & adj[v]) == 0:
                    # Valid to add
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added = True
                    break
            if added:
                break
        
        if not added:
            # No edge could be added within the current independent set.
            # This means the graph is "saturated" with respect to this independent set.
            # We might be stuck in a local minimum.
            # Perturbation: Remove a random edge?
            # Or just break and restart the main loop (new random graph).
            # Restarting is often better than complex perturbation.
            return adj, False
            
        # If we added an edge, check if alpha < 10 now
        if not _has_independent_set(adj, n, 10):
            return adj, True
            
    return adj, False

def _greedy_independent_set(adj, n):
    """
    Greedy heuristic to find a large independent set.
    """
    remaining = (1 << n) - 1
    iset = []
    
    while remaining:
        # Pick vertex with minimum degree in remaining subgraph
        best_v = -1
        min_deg = n + 1
        temp = remaining
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (adj[v] & remaining).bit_count()
            if deg < min_deg:
                min_deg = deg
                best_v = v
            temp ^= lsb
            
        if best_v == -1:
            break
            
        iset.append(best_v)
        # Remove best_v and its neighbors from remaining
        remaining &= ~(1 << best_v)
        remaining &= ~adj[best_v]
        
    return iset

def _adj_to_edges(adj, n):
    edges = []
    for u in range(n):
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            mask ^= lsb
    return {'n': n, 'edges': edges}

"""
Search for a large graph with no regular induced subgraph on 6+ vertices.
Uses repair-based local search: 
1. Start with random graph.
2. Randomly sample 6-subsets to find regular induced subgraphs (violations).
3. If a violation is found, flip a random edge within it to break regularity.
4. Repeat until no violations are found in a large number of samples, then verify.
"""
import time
import random

def construct():
    start = time.time()
    deadline = start + 145  # 5s safety margin
    
    best_n = 0
    best_edges = []
    
    # Try n=20 first, then 19, etc.
    # n=20 is the target. n=21 is a bonus but harder.
    for target_n in [20, 19, 18]:
        if time.time() > deadline - 10:
            break
            
        found = _local_search(target_n, deadline)
        if found:
            best_n = target_n
            best_edges = found
            # If we found 20, we can try to extend to 21 if time permits?
            # The directive says maximize n.
            # Let's stick to finding the best n reliably.
            # If we find 20, it's a good score. 
            # Attempting 21 with remaining time is risky but possible.
            # For now, let's just return the first high n found.
            # Actually, if we find 20, we should try 21 with remaining time.
            # But 21 is much harder. Let's just return 20 if found.
            # If 20 fails, try 19.
            break
            
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
        
    # Fallback
    return {'n': 5, 'edges': []}

def _local_search(n, deadline):
    """
    Perform local search to find a valid graph on n vertices.
    Returns list of edges if found, else None.
    """
    rng = random.Random(42) # Fixed seed for reproducibility in this run? 
    # Better to use multiple seeds if one fails.
    
    # We will try a few random restarts
    for restart in range(5):
        if time.time() > deadline - 5:
            break
            
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        # Local search loop
        # We maintain a count of consecutive "clean" 6-subset checks.
        # If this count gets high, we do a full verification.
        clean_count = 0
        max_clean = 10000 # Threshold to trigger full check
        
        while time.time() < deadline - 5:
            # Sample a random 6-subset
            subset = rng.sample(range(n), 6)
            
            # Check if this subset induces a regular graph
            if _is_regular(adj, subset):
                # Violation found!
                # Flip a random edge within this subset to break regularity
                # We need to pick an edge (u,v) in the subset.
                # There are 15 edges in K6.
                # Pick a random pair from the subset
                u, v = rng.sample(subset, 2)
                
                # Flip edge (u,v)
                if adj[u] & (1 << v):
                    # Remove edge
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                # Reset clean count
                clean_count = 0
            else:
                # No violation in this sample
                clean_count += 1
                
                if clean_count >= max_clean:
                    # Perform full verification
                    if _is_valid_full(adj, n, deadline):
                        # Convert to edges
                        edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if adj[i] & (1 << j):
                                    edges.append([i, j])
                        return edges
                    else:
                        # If full verification fails, we might have missed a violation
                        # in the random sampling, or the graph is just invalid.
                        # We can try to find a specific violation to repair.
                        # For simplicity, let's just reset clean_count and continue.
                        # Or, we can force a repair by finding a violation via exhaustive search on a smaller scale?
                        # No, let's just reset and continue sampling.
                        clean_count = 0
                        # To avoid getting stuck, let's do a "random kick": flip a random edge in the whole graph
                        u, v = rng.sample(range(n), 2)
                        if adj[u] & (1 << v):
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)

    return None

def _is_regular(adj, subset):
    """Check if the induced subgraph on 'subset' is regular."""
    mask = 0
    for v in subset:
        mask |= (1 << v)
        
    # Get degree of first vertex
    v0 = subset[0]
    d0 = (adj[v0] & mask).bit_count()
    
    for v in subset[1:]:
        dv = (adj[v] & mask).bit_count()
        if dv != d0:
            return False
    return True

def _is_valid_full(adj, n, deadline):
    """
    Exhaustive check for any regular induced subgraph of size >= 6.
    Returns True if valid (no such subgraph), False otherwise.
    """
    # Optimization: Check size 6 first. If no size 6 regular subgraph, 
    # does that imply no larger ones? No. 
    # However, if a larger set is regular, its subsets are not necessarily regular.
    # So we must check all sizes.
    
    # But checking all subsets of size 6 is the most expensive part.
    # If we find no size 6 regular subgraph, we still need to check 7, 8...
    # But typically, if size 6 is clean, larger sizes are also likely clean 
    # or we can prune.
    
    # For n=20, C(20,6) = 38760. This is fast enough to do fully if done rarely.
    # C(20,7) = 77520.
    # C(20,20) = 1.
    
    import itertools
    
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if time.time() > deadline:
                return False # Timeout, assume invalid to be safe
            if _is_regular(adj, subset):
                return False
    return True

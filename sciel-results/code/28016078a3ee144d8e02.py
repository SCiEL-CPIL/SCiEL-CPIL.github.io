import time
import random
import numpy as np

# EVOLVE-BLOCK-START

DEADLINE_SECONDS = 145.0
START_TIME = None

def _count_violations(adj, n):
    """
    Count the number of subsets of size >= 6 that induce a regular subgraph.
    Uses vectorized NumPy operations for speed.
    adj is a (n, n) binary matrix (symmetric, 0 diagonal).
    """
    total_violations = 0
    # Precompute degrees
    deg = adj.sum(axis=1)
    
    # We need to check subsets of size k=6..n
    # For n up to 22, this is feasible with vectorization if we're clever.
    # However, iterating over all combinations in Python is still slow.
    # Let's use a different approach:
    # A subset S is regular if all vertices in S have the same degree within S.
    # For a random graph, the number of regular induced subgraphs of size 6 is expected to be low.
    # We can sample or use a faster check.
    
    # Actually, for n=20, C(20,6) = 38760. This is small enough to check individually if the check is fast.
    # Let's optimize the individual check.
    
    import itertools
    
    for k in range(6, n + 1):
        for combo in itertools.combinations(range(n), k):
            # Extract subgraph
            # adj[combo][:, combo] is the induced subgraph
            # We need to check if all row sums are equal
            sub_adj = adj[np.ix_(combo, combo)]
            sub_deg = sub_adj.sum(axis=1)
            if np.all(sub_deg == sub_deg[0]):
                total_violations += 1
                # We can return early if we just need to know if > 0, but for optimization we might want the count.
                # For the search, we can stop at 1 to save time if we just want to know if it's valid.
                # Let's return early if we just need a validity check.
                # But for a fitness function, count is better.
                # Given time constraints, let's return 1 if any violation found, else 0?
                # No, let's try to be a bit more informative but fast.
                # If we find one, we can break out of the inner loop?
                # For now, let's just count.
    return total_violations

def _is_valid(adj, n):
    """Check if the graph has no regular induced subgraph of size >= 6."""
    return _count_violations(adj, n) == 0

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    deadline = START_TIME + DEADLINE_SECONDS
    rng = np.random.default_rng(42)
    
    best_n = 0
    best_adj = None
    
    # We want to maximize n.
    # Try n from 22 down to 6.
    for n in range(22, 5, -1):
        if time.time() > deadline - 10.0:
            break
            
        # Try to find a valid graph of size n.
        # Strategy: Random restarts with local search or just random sampling?
        # Random sampling is unlikely to find a valid graph for n=20 or 21.
        # Local search is better.
        
        # Let's try a few random initializations and then local search.
        max_restarts = 50 if n <= 20 else 20
        
        found = False
        for r in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Initialize random graph
            # p = 0.5 is a good start.
            p = 0.5
            # Generate random symmetric matrix
            adj = (rng.random((n, n)) < p).astype(np.int8)
            # Make it symmetric and zero diagonal
            adj = np.maximum(adj, adj.T)
            np.fill_diagonal(adj, 0)
            
            # Local search:
            # While there are violations, pick a violating subset and flip an edge.
            # Flipping an edge (u, v) changes the degree of u and v in any subset containing both.
            # It changes the degree of u in subsets containing u but not v, and vice versa.
            
            max_steps = 2000
            for step in range(max_steps):
                if time.time() > deadline - 2.0:
                    break
                    
                # Check validity
                # This is the bottleneck.
                # Let's optimize the violation check.
                # Instead of counting all, just find one.
                
                violation = None
                # Check size 6 first (most likely)
                import itertools
                for combo in itertools.combinations(range(n), 6):
                    sub_adj = adj[np.ix_(combo, combo)]
                    sub_deg = sub_adj.sum(axis=1)
                    if np.all(sub_deg == sub_deg[0]):
                        violation = combo
                        break
                if violation is None:
                    # Check larger sizes?
                    # If no size 6 violation, there might still be a size 7+ violation.
                    # But if we fix all size 6, size 7+ might disappear or be rarer.
                    # For now, let's check 7 and 8 too if time permits.
                    for k in range(7, n + 1):
                        for combo in itertools.combinations(range(n), k):
                            sub_adj = adj[np.ix_(combo, combo)]
                            sub_deg = sub_adj.sum(axis=1)
                            if np.all(sub_deg == sub_deg[0]):
                                violation = combo
                                break
                        if violation is not None:
                            break
                
                if violation is None:
                    # Valid!
                    found = True
                    best_n = n
                    best_adj = adj.copy()
                    break
                
                # Flip an edge to break the violation.
                # Pick two vertices from the violating subset.
                u, v = rng.choice(violation, size=2, replace=False)
                # Flip edge (u, v)
                adj[u, v] = 1 - adj[u, v]
                adj[v, u] = adj[u, v]
                
                # Also, consider flipping an edge between a vertex in the violation and one outside?
                # That might be more effective at breaking the regularity without creating new ones.
                # Let's try flipping (u, w) where w is outside the violation.
                # If we flip (u, v) inside, it changes degrees of u and v in the subgraph.
                # If we flip (u, w) outside, it changes degree of u in the subgraph (if w not in S, degree of u in S doesn't change? No, degree of u in S is count of neighbors in S. w is not in S, so degree of u in S doesn't change. Wait.)
                # Degree of u in S = sum(adj[u, s] for s in S).
                # If we flip (u, w) where w not in S, this does NOT change the degree of u in S.
                # So flipping an internal edge is the only way to change the degrees within S.
                # So we must flip an internal edge.
            
            if found:
                break
        
        if found:
            break
    
    if best_n == 0:
        # Fallback
        return {'n': 5, 'edges': []}
    
    # Convert adj to edge list
    edges = []
    for u in range(best_n):
        for v in range(u + 1, best_n):
            if best_adj[u, v] == 1:
                edges.append([u, v])
    
    return {'n': best_n, 'edges': edges}

# EVOLVE-BLOCK-END

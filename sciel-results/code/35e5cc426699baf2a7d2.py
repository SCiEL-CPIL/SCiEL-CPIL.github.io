import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _check_subgraph(adj, n, size, k):
    """Check if a specific subset of size k is a regular subgraph."""
    # Get the vertices
    subset = list(subset)  # This won't work, need to pass subset
    pass

def _is_regular_subgraph(adj, n, subset):
    """Check if the induced subgraph on 'subset' is regular."""
    k = len(subset)
    if k < 6:
        return False
    # Compute degrees within the subset
    idx_map = {v: i for i, v in enumerate(subset)}
    degrees = [0] * k
    for i, u in enumerate(subset):
        for j, v in enumerate(subset):
            if i != j and adj[u][v] == 1:
                degrees[i] += 1
    # Check if all degrees are equal
    d0 = degrees[0]
    for d in degrees[1:]:
        if d != d0:
            return False
    return True

def _find_violation(adj, n, min_size=6):
    """Find a violation (regular induced subgraph of size >= min_size).
    Returns (subset, size) or (None, 0)."""
    # Check from smallest to largest size
    for k in range(min_size, n + 1):
        # Sample some subsets to avoid full enumeration if too slow
        # For n <= 22, C(22,6) = 74613, which is manageable.
        # C(22,7) = 170544, C(22,8) = 319770, etc.
        # Total subsets of size >= 6 for n=22 is about 1.8 million.
        # This is doable in Python if optimized.
        
        # Use itertools.combinations
        for subset in combinations(range(n), k):
            if _is_regular_subgraph(adj, n, subset):
                return list(subset), k
    return None, 0

def _count_violations(adj, n, min_size=6):
    """Count the number of regular induced subgraphs of size >= min_size."""
    count = 0
    for k in range(min_size, n + 1):
        for subset in combinations(range(n), k):
            if _is_regular_subgraph(adj, n, subset):
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # Target n values to try, in descending order
    # Start with 21, then 20, 19...
    best_result = None
    
    for n_target in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline - 5.0:
            break
            
        # Try to find a valid graph on n_target vertices
        # Strategy: Start with a random graph and perform local search
        # to eliminate regular induced subgraphs of size >= 6.
        
        found = False
        max_attempts = 50  # Number of random restarts
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 2.0:
                break
                
            # Initialize a random graph
            # Use a moderate edge probability
            p = 0.5
            adj = [[0] * n_target for _ in range(n_target)]
            for u in range(n_target):
                for v in range(u + 1, n_target):
                    if random.random() < p:
                        adj[u][v] = 1
                        adj[v][u] = 1
            
            # Local search: repeatedly find a violation and flip an edge in it
            # to break regularity.
            for step in range(2000):
                if time.time() > deadline - 1.0:
                    break
                    
                # Find a violation
                subset, size = _find_violation(adj, n_target, 6)
                if subset is None:
                    # No violation found!
                    found = True
                    # Build edge list
                    edges = []
                    for u in range(n_target):
                        for v in range(u + 1, n_target):
                            if adj[u][v] == 1:
                                edges.append([u, v])
                    if best_result is None or n_target > best_result[0]:
                        best_result = (n_target, edges)
                    break
                else:
                    # Flip a random edge within the violating subset
                    # We need to pick an edge (u, v) in the subset such that
                    # flipping it breaks the regularity.
                    # Simple heuristic: flip a random edge in the subset.
                    # To be more effective, we can try flipping edges that
                    # are likely to break the regular degree condition.
                    
                    # Let's just pick a random pair from the subset.
                    # There are C(size, 2) edges.
                    # We pick a random u, v in subset.
                    while True:
                        u = random.choice(subset)
                        v = random.choice(subset)
                        if u != v:
                            break
                    
                    # Flip the edge
                    adj[u][v] ^= 1
                    adj[v][u] ^= 1
            
            if found:
                break
            
            # If not found, try another random restart
            
        if found:
            # We found a valid graph for this n_target.
            # Since we are trying larger n first, we can return immediately
            # or continue to see if we can find an even larger one?
            # The loop iterates n_target in descending order, so the first found
            # is the largest.
            break
        else:
            # No valid graph found for this n_target within time budget.
            # Continue to smaller n_target.
            pass
    
    if best_result is not None:
        n_found, edges_found = best_result
        return {'n': n_found, 'edges': edges_found}
    else:
        # Fallback: return a known valid small graph (e.g., n=5 empty graph)
        # Or try to construct one quickly.
        # For n < 6, any graph is valid.
        return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END

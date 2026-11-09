import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # We want to maximize n. 
    # Known: n=20 is possible. n=21 is the target.
    # We will try n=21 first with a limited budget, then n=20, etc.
    
    rng = random.Random(42)
    
    # Fallback: A small valid graph.
    # n=5 is trivially valid (no subset of size 6).
    # But we can easily find n=10 or n=15.
    # Let's generate a random n=10 graph as a robust fallback.
    def get_fallback(n=10):
        adj = [0] * n
        for _ in range(10000):
            for i in range(n):
                adj[i] = 0
                for j in range(i+1, n):
                    if rng.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            if is_valid(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
        # If we fail (unlikely), return n=5
        return {'n': 5, 'edges': []}
        
    best_result = get_fallback(10)
    
    # Try larger n
    # Start with 21, then 20, etc.
    for n in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        if time.time() > deadline - 5:
            break
            
        # Allocate time based on n. 
        # n=21 is harder, give it less time than n=20? 
        # Actually, n=20 is known to exist, so it should be found quickly.
        # n=21 might not exist or be rare.
        
        target_time = 10.0 if n == 21 else 20.0
        if time.time() + target_time > deadline:
            target_time = deadline - time.time() - 2
            
        result = local_search(n, target_time, rng)
        
        if result is not None:
            return result
        elif result == 'timeout' and best_result['n'] < n:
             # If we timed out but didn't find one, keep the best valid one found so far
             # But we didn't find a valid one for this n, so best_result remains unchanged.
             pass
             
    return best_result

def is_regular_induced(adj, subset):
    """Check if the induced subgraph on subset is regular."""
    s = len(subset)
    if s < 2:
        return True
    # Create mask
    mask = 0
    for v in subset:
        mask |= (1 << v)
    
    # Calculate degrees
    degs = []
    for v in subset:
        degs.append((adj[v] & mask).bit_count())
        
    # Check if all degrees are the same
    d0 = degs[0]
    for d in degs[1:]:
        if d != d0:
            return False
    return True

def find_violation(adj, n):
    """Find a subset of size >= 6 that induces a regular graph.
    Returns the subset if found, None if graph is valid."""
    # Check sizes from 6 up to n
    for k in range(6, n + 1):
        for subset in combinations(range(n), k):
            if is_regular_induced(adj, subset):
                return subset
    return None

def is_valid(adj, n):
    """Check if graph has no regular induced subgraph on 6+ vertices."""
    return find_violation(adj, n) is None

def local_search(n, time_limit, rng):
    """
    Local search to find a valid graph on n vertices.
    Returns dict if found, None if not found within time_limit.
    """
    start_time = time.time()
    
    # Initialize random graph
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Check if already valid
    if is_valid(adj, n):
        return to_result(adj, n)
    
    # Local search loop
    # We track the current number of violations? 
    # Simpler: Just find *a* violation, flip an edge, repeat.
    # To avoid getting stuck, we can use simulated annealing or just random restarts.
    # Given the small size, random restarts are effective.
    
    max_restarts = 20
    restarts = 0
    
    while time.time() < start_time + time_limit:
        if restarts >= max_restarts:
            # Re-initialize
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            restarts = 0
            if is_valid(adj, n):
                return to_result(adj, n)
            max_restarts += 10 # Allow more restarts if we keep failing
            
        viol = find_violation(adj, n)
        if viol is None:
            return to_result(adj, n)
            
        # Pick an edge to flip
        # Strategy: Flip an edge within the violating subset to break regularity.
        # Or flip a random edge.
        # Flipping an edge inside the subset is more likely to break that specific violation.
        # However, it might create new ones.
        
        # Let's pick a random pair from the subset
        u, v = rng.sample(viol, 2)
        
        # Flip edge (u, v)
        if adj[u] & (1 << v):
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # Occasionally, flip a random edge outside the subset to escape local minima
        if rng.random() < 0.1:
            a, b = rng.sample(range(n), 2)
            if adj[a] & (1 << b):
                adj[a] ^= (1 << b)
                adj[b] ^= (1 << a)
            else:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                
    return None

def to_result(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END

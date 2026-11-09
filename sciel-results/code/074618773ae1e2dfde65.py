import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit (evaluator kills at 150s, but we want to be safe)
    
    rng = random.Random(42)
    
    # Fallback: A small valid graph.
    def get_fallback(n=10):
        adj = [0] * n
        for _ in range(5000):
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
        return {'n': 5, 'edges': []}
        
    best_result = get_fallback(10)
    
    # Try larger n
    # We know n=20 exists. Try 21, then 20.
    for n in [21, 20, 19, 18, 17, 16]:
        if time.time() > deadline - 10:
            break
            
        # Allocate time
        # n=21 is hard, n=20 is medium.
        target_time = 15.0 if n == 21 else 25.0
        if time.time() + target_time > deadline:
            target_time = deadline - time.time() - 5
            
        result = local_search(n, target_time, rng)
        
        if result is not None:
            return result
             
    return best_result

def is_regular_induced(adj, subset, mask):
    """Check if the induced subgraph on subset is regular."""
    s = len(subset)
    if s < 2:
        return True
    
    # Calculate degrees
    d0 = (adj[subset[0]] & mask).bit_count()
    for v in subset[1:]:
        if (adj[v] & mask).bit_count() != d0:
            return False
    return True

def find_violation(adj, n):
    """Find a subset of size >= 6 that induces a regular graph.
    Returns the subset if found, None if graph is valid."""
    # Check sizes from 6 up to n
    # Smaller subsets are more likely to be regular in random graphs?
    # Actually, as k increases, the number of subsets increases.
    # But the probability of a random k-subset being regular might decrease?
    # For k=6, there are few regular graphs (0,1,2,3,4,5-regular).
    # For k=10, many more regular graphs exist.
    # So checking k=6 first is good for quick rejection of random graphs.
    
    for k in range(6, n + 1):
        for subset in combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if is_regular_induced(adj, subset, mask):
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
                
    if is_valid(adj, n):
        return to_result(adj, n)
    
    max_restarts = 50
    restarts = 0
    no_improve_counter = 0
    min_violations_found = float('inf')
    
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
            no_improve_counter = 0
            if is_valid(adj, n):
                return to_result(adj, n)
            max_restarts += 20
            
        viol = find_violation(adj, n)
        if viol is None:
            return to_result(adj, n)
            
        # Heuristic: Try to break the violation
        # Pick a random edge in the violation
        u, v = rng.sample(viol, 2)
        
        # Flip edge (u, v)
        if adj[u] & (1 << v):
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # Check if we improved? 
        # We don't count violations (too slow). 
        # Just keep going.
        # If we are stuck, do a random jump.
        no_improve_counter += 1
        if no_improve_counter > 100:
            # Jump: flip 3 random edges
            for _ in range(3):
                a, b = rng.sample(range(n), 2)
                if adj[a] & (1 << b):
                    adj[a] ^= (1 << b)
                    adj[b] ^= (1 << a)
                else:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
            no_improve_counter = 0
            restarts += 1 # Count jump as partial restart?

    return None

def to_result(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END

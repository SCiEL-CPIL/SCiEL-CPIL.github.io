import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def is_regular_subgraph(adj, subset):
        """Check if the induced subgraph on 'subset' is regular."""
        if len(subset) < 6:
            return False
        
        mask = 0
        for v in subset:
            mask |= (1 << v)
            
        degrees = []
        for v in subset:
            # Count neighbors of v within the subset
            deg = (adj[v] & mask).bit_count()
            degrees.append(deg)
            
        # Check if all degrees are the same
        first_deg = degrees[0]
        for d in degrees[1:]:
            if d != first_deg:
                return False
        return True

    def find_violation(adj, n):
        """Find any induced regular subgraph of size >= 6."""
        # Check sizes from 6 up to n
        # Smaller sizes are more likely to be violated, so check them first
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                if is_regular_subgraph(adj, subset):
                    return subset
        return None

    def flip_edge(adj, u, v):
        """Flip the edge between u and v."""
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    def try_solve(n, time_limit):
        """Attempt to find a valid graph on n vertices within time_limit."""
        start = time.time()
        max_restarts = 100
        
        for restart in range(max_restarts):
            if time.time() - start > time_limit:
                return None
            
            # Initialize random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search
            steps = 0
            max_steps = 10000
            while steps < max_steps:
                if time.time() - start > time_limit:
                    return None
                
                violation = find_violation(adj, n)
                if violation is None:
                    # Valid graph found
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                
                # Pick a random edge within the violating subset to flip
                subset = list(violation)
                # Randomly pick two vertices from the subset
                u, v = random.sample(subset, 2)
                flip_edge(adj, u, v)
                steps += 1
                
        return None

    # Strategy: Try n=21 first, then n=20
    # Allocate time: 40s for n=21, 70s for n=20
    
    best_result = None
    
    # Try n=21
    result_21 = try_solve(21, 40)
    if result_21:
        best_result = result_21
    else:
        # Try n=20
        result_20 = try_solve(20, 70)
        if result_20:
            best_result = result_20

    if best_result:
        return best_result
    else:
        # Fallback: Return a small valid graph (e.g., empty graph on 5 vertices)
        # This ensures we don't crash
        return {'n': 5, 'edges': []}

import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Precompute all subsets of size >= 6 for a given n
    def get_subsets(n):
        subsets = []
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                subsets.append(frozenset(subset))
        return subsets

    # Function to check if a subset of vertices induces a regular graph
    # adj is a list of bitmasks
    def is_regular_subset(vertices, adj):
        if len(vertices) < 6:
            return False
        degrees = []
        for u in vertices:
            deg = 0
            for v in vertices:
                if u != v:
                    if (adj[u] >> v) & 1:
                        deg += 1
            degrees.append(deg)
        if all(d == degrees[0] for d in degrees):
            return True
        return False

    # Function to find a violating subset or return None if valid
    def find_violation(n, adj, subsets):
        for subset in subsets:
            if is_regular_subset(list(subset), adj):
                return subset
        return None

    # Search for a valid graph
    # We try sizes from 20 down to 6
    for n in range(20, 5, -1):
        if time.time() - start_time > time_limit:
            break
            
        subsets = get_subsets(n)
        
        # Strategy: Random initialization + Local Search
        # We will try a few random graphs and then mutate.
        
        found = False
        best_graph = None
        
        for attempt in range(2000):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Check if valid
            violation = find_violation(n, adj, subsets)
            if violation is None:
                found = True
                best_graph = adj
                break
            
            # If not valid, try to repair
            # Use local search: flip edges in the violating subset or random edges
            current_adj = adj[:]
            improved = False
            
            # Try to fix by flipping edges within the violating subset
            # A regular graph on k vertices has all degrees equal to some d.
            # Flipping an edge changes degrees of two vertices by +/- 1.
            # We want to break the regularity.
            
            for _ in range(50):
                if time.time() - start_time > time_limit:
                    break
                
                # Pick a random edge to flip
                # Bias towards edges inside the current violation
                if random.random() < 0.7 and violation:
                    vlist = list(violation)
                    u = random.choice(vlist)
                    v = random.choice(vlist)
                    if u == v:
                        continue
                else:
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v:
                        continue
                
                # Flip edge
                if (current_adj[u] >> v) & 1:
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                
                # Check if still valid
                violation = find_violation(n, current_adj, subsets)
                if violation is None:
                    found = True
                    best_graph = current_adj
                    improved = True
                    break
            
            if found:
                break
        
        if found:
            # Convert adj to edges list
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (best_graph[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        
        # If n=20 failed, try n=19, etc.
        
    # Fallback: n=6
    # A non-regular graph on 6 vertices
    # e.g., K6 minus an edge
    n = 6
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (i == 0 and j == 1): # Missing edge (0,1)
                edges.append([i, j])
    return {'n': n, 'edges': edges}

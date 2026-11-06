import time
import random

def construct():
    # Target n is 39, as R(3,10) > 39 is known.
    # We attempt to find a triangle-free graph on n vertices with alpha < 10.
    
    deadline = time.time() + 110.0  # Leave 10s safety margin
    
    # Try n from 39 down to 30
    for n in range(39, 29, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this n
        max_restarts = 200
        for _ in range(max_restarts):
            if time.time() > deadline:
                break
            graph = greedy_triangle_free(n, random_seed=random.randint(0, 2**31))
            if graph is not None:
                # Verify alpha < 10
                if max_independent_set_size(graph, n) < 10:
                    return {'n': n, 'edges': graph}
                    
    # Fallback: Return a known small valid graph if search fails (should not happen for n=39)
    # Andrásfai graph And(9) on 26 vertices
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}

def greedy_triangle_free(n, random_seed=0):
    """
    Constructs a maximal triangle-free graph using greedy edge addition.
    Returns list of edges or None if timeout (handled externally).
    """
    rng = random.Random(random_seed)
    # Adjacency as list of sets for fast lookup
    adj = [set() for _ in range(n)]
    edges = []
    
    # Shuffle vertices to randomize the greedy choice
    vertices = list(range(n))
    rng.shuffle(vertices)
    
    # We iterate through pairs in a random order
    # To be efficient, we can just iterate i, j and check
    # But random order helps escape local minima? 
    # Actually, simple greedy on a fixed order is deterministic.
    # Randomizing the order of vertex pairs is better.
    
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(pairs)
    
    for u, v in pairs:
        # Check if adding (u, v) creates a triangle
        # Triangle exists if there is a common neighbor w
        # i.e., w in adj[u] AND w in adj[v]
        if not adj[u].intersection(adj[v]):
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
            
    return edges

def max_independent_set_size(edges, n):
    """
    Exact branch and bound for maximum independent set.
    Returns the size of the maximum independent set.
    """
    # Build adjacency matrix or sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Use a simple branch and bound with coloring bound (Tomita's algorithm simplified)
    # For n=39, this is very fast.
    
    best = 0
    # Vertices remaining to be considered
    # We use a recursive function
    
    import sys
    sys.setrecursionlimit(10000)

    def branch_and_bound(vertices, current_size):
        nonlocal best
        if not vertices:
            if current_size > best:
                best = current_size
            return
        
        # Bound: current_size + len(vertices) <= best
        if current_size + len(vertices) <= best:
            return
            
        # Pick a vertex v
        # Heuristic: pick vertex with highest degree in the subgraph induced by vertices
        # Or just the first one
        v = vertices[0]
        remaining = vertices[1:]
        
        # Branch 1: Include v
        # Remove v and its neighbors from remaining
        neighbors = adj[v]
        new_remaining = [u for u in remaining if u not in neighbors]
        branch_and_bound(new_remaining, current_size + 1)
        
        # Branch 2: Exclude v
        branch_and_bound(remaining, current_size)
        
    # Initial call
    # To speed up, we can order vertices by degree
    degs = [len(adj[i]) for i in range(n)]
    ordered_vertices = sorted(range(n), key=lambda x: degs[x], reverse=True)
    # Actually, for independent set, we want to branch on vertices that are "central" 
    # so that excluding them removes many options? 
    # Standard B&B: 
    # Let's use a bitset representation for speed if possible, but sets are okay for n=39.
    
    branch_and_bound(list(range(n)), 0)
    return best

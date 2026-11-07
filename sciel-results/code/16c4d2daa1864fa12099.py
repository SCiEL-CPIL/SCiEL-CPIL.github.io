import random
import time
from itertools import combinations

P = {'n': 12, 'minimum_degree': 3, 'girth': 4}

def get_cycles(graph, n):
    """
    Find all simple cycles in the graph.
    Returns a set of cycle lengths.
    """
    cycles = set()
    # Use DFS to find cycles
    # To avoid duplicates, we can enforce that the cycle starts at the smallest vertex
    # and follows increasing order? No, simple cycles don't have an order.
    # Standard approach: Find all cycles, normalize them (rotate so min vertex is first, then direction), store frozenset of edges or tuple of vertices.
    
    adj = {i: set() for i in range(n)}
    for u, v in graph:
        adj[u].add(v)
        adj[v].add(u)
        
    found_cycles = set()
    
    def dfs(start, current, path, visited):
        # If we return to start and path length >= 3 (edges)
        if current == start and len(path) > 1:
            # Normalize cycle
            # Rotate so min vertex is first
            min_idx = min(range(len(path)), key=lambda i: path[i])
            rotated = path[min_idx:] + path[:min_idx]
            # Check reverse
            rev = [rotated[0]] + rotated[1:][::-1]
            # Actually, just store the set of edges or a canonical tuple
            # Tuple of vertices in order
            t1 = tuple(rotated)
            t2 = tuple(rev)
            key = min(t1, t2)
            if key not in found_cycles:
                found_cycles.add(key)
                cycles.add(len(path)) # length is number of vertices = number of edges
            return
        
        for neighbor in adj[current]:
            if neighbor == start:
                if len(path) >= 2: # At least 3 vertices (triangle) but we need girth 4, so len >= 3 edges?
                    # Path is list of vertices. Start is first.
                    # If current connects to start, we close the loop.
                    # Length of cycle is len(path)
                    dfs(start, start, path, visited)
            elif neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(start, neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)
                
    # Start DFS from each vertex, but only allow visiting vertices > start to avoid duplicates?
    # No, that misses cycles that don't include the smallest vertex as the "start" in the DFS tree sense?
    # Better: Iterate all pairs?
    # For n=12, DFS from each vertex is fine.
    for start in range(n):
        # Only consider cycles where 'start' is the smallest vertex in the cycle
        # We can enforce this by only visiting neighbors > start?
        # No, the cycle can go through smaller vertices?
        # If we restrict the DFS to only visit vertices > start, we will only find cycles where all other vertices are > start.
        # This ensures each cycle is found exactly once (when starting at its min vertex).
        visited = {start}
        path = [start]
        # Modify dfs to only go to neighbors > start?
        # If we do that, we might miss cycles that "wrap around"?
        # No, a simple cycle has a unique minimum vertex. If we start there and only visit larger vertices, we will find the cycle if the edges allow it.
        # But the cycle might go from a large vertex to a small vertex (which is the start).
        # So the restriction "only visit > start" is valid for finding the cycle structure, provided we allow the final step to 'start'.
        
        def dfs_restricted(start, current, path, visited):
            if current == start and len(path) > 1:
                # Found a cycle
                # Check if it's a simple cycle (it is, by construction)
                # Add length
                cycles.add(len(path))
                return
            
            for neighbor in adj[current]:
                if neighbor == start:
                    if len(path) >= 2: # Cycle length >= 3
                         # Check girth? We filter later.
                         dfs_restricted(start, start, path, visited)
                elif neighbor > start and neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs_restricted(start, neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)
                    
        dfs_restricted(start, start, [start], {start})

    return cycles

def construct():
    start_time = time.time()
    deadline = 80.0 # 10s margin
    
    n = 12
    min_deg = 3
    
    best_score = 0
    best_edges = None
    best_val = 100
    
    # Seed: Two K3,3
    seed_edges = set()
    for i in range(6):
        for u,v in [(i,(i+1)%6),(i+6,(i+1)%6+6),(i,i+6)]:
            seed_edges.add(tuple(sorted((u,v))))
    seed_val = len(get_cycles(seed_edges, n))
    if seed_val < best_val:
        best_val = seed_val
        best_edges = [list(e) for e in sorted(seed_edges)]
        
    # Random Search
    # Try to find a graph with val = 1
    # Use a dense graph and remove edges?
    # Or build from scratch?
    
    # Strategy: Random bipartite graph with min degree 3.
    # Split vertices into A (0-5) and B (6-11).
    # Ensure min degree 3.
    
    for _ in range(5000):
        if time.time() > deadline:
            break
            
        # Generate random bipartite graph
        edges = set()
        # Randomly connect A and B
        # To ensure min degree 3, we can start with a random regular-ish graph
        # Simple approach: Random permutation?
        
        # Try to construct a 3-regular bipartite graph
        # Random matching combination
        # 3 random perfect matchings between A and B
        
        matchings = []
        for _ in range(3):
            perm = list(range(6))
            random.shuffle(perm)
            m = set()
            for i in range(6):
                m.add((i, perm[i] + 6))
            matchings.append(m)
            
        # Union of matchings
        for m in matchings:
            for u, v in m:
                edges.add(tuple(sorted((u, v))))
                
        # Check min degree
        deg = [0]*n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            
        if min(deg) < 3:
            continue
            
        # Check girth and cycle lengths
        # First check if triangle-free (bipartite is always triangle-free)
        # Check if it has a cycle (must, since min deg 3)
        
        lengths = get_cycles(edges, n)
        
        if not lengths:
            continue
            
        val = len(lengths)
        
        if val < best_val:
            best_val = val
            best_edges = [list(e) for e in sorted(edges)]
            if best_val == 1:
                return best_edges

    # If we didn't find 1, try local search on the best found?
    # Or just return the best found.
    
    return best_edges

# EVOLVE-BLOCK-END

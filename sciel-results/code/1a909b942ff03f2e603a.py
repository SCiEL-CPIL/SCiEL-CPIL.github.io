import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n = 20
    vertices = list(range(n))
    
    # Generate a 1-factorization of K20 using round-robin method
    # For even n, we can use the standard round-robin scheduling
    # Fix vertex 0, rotate others
    def round_robin_factorization(n):
        """Returns list of n-1 perfect matchings covering all edges of K_n (n even)."""
        if n % 2 != 0:
            raise ValueError("n must be even")
        
        matchings = []
        # Use vertices 0..n-1
        # Standard algorithm: fix vertex 0, rotate n-1 other vertices
        # Actually for even n, fix two vertices and rotate n-2? No.
        # For n even: label vertices 0..n-1. In round r (0..n-2), 
        # vertex 0 pairs with vertex r, and other pairs are formed.
        
        # Better approach: use the circle method
        # Place n-1 vertices on a circle, fix one vertex
        # Actually, let me use a cleaner construction
        
        # For n=20: vertices 0..19
        # In each round, create a perfect matching
        # Round r: pair (0, r), and for i=1..(n//2-1), pair (r-i mod (n-1), r+i mod (n-1))
        # where we work with vertices 1..n-1 for the circle part
        
        circle_vertices = list(range(1, n))  # vertices 1..19
        m = n - 1  # 19 vertices on circle
        
        for r in range(n - 1):  # n-1 = 19 rounds
            matching = []
            # Vertex 0 pairs with circle_vertices[r]
            matching.append((0, circle_vertices[r]))
            
            # Pair up remaining circle vertices
            # The remaining vertices are circle_vertices[0..m-1] except index r
            # Pair (r-i mod m, r+i mod m) for i=1..m//2
            for i in range(1, m // 2 + 1):
                u_idx = (r - i) % m
                v_idx = (r + i) % m
                u = circle_vertices[u_idx]
                v = circle_vertices[v_idx]
                matching.append((u, v))
            
            matchings.append(matching)
        
        return matchings
    
    # Verify the factorization covers all edges exactly once
    def verify_factorization(matchings, n):
        seen = set()
        for matching in matchings:
            for (u, v) in matching:
                edge = (min(u,v), max(u,v))
                if edge in seen:
                    return False
                seen.add(edge)
        return len(seen) == n * (n - 1) // 2
    
    # Check if a given edge coloring satisfies the rainbow 4-cycle property
    def check_rainbow_4cycles(edge_colors, n):
        """edge_colors: dict mapping (u,v) with u<v to color. Returns True if valid."""
        # Build adjacency with colors
        adj = {v: {} for v in range(n)}
        for (u, v), c in edge_colors.items():
            adj[u][v] = c
            adj[v][u] = c
        
        # Check all 4-cycles: for each pair of non-adjacent vertices, 
        # check all common neighbors... actually, enumerate all 4-cycles
        # A 4-cycle is a-b-c-d-a where a,b,c,d distinct
        # To avoid duplicates, fix a < b < c < d and check cycle a-b-c-d-a
        # But that's not all 4-cycles. Better: for each set of 4 vertices, 
        # check all 3 possible 4-cycle orderings
        
        for a, b, c, d in combinations(range(n), 4):
            # Three possible 4-cycles on {a,b,c,d}:
            # 1. a-b-c-d-a: needs edges (a,b),(b,c),(c,d),(d,a)
            # 2. a-b-d-c-a: needs edges (a,b),(b,d),(d,c),(c,a)
            # 3. a-c-b-d-a: needs edges (a,c),(c,b),(b,d),(d,a)
            
            cycles = [
                [(a,b), (b,c), (c,d), (d,a)],
                [(a,b), (b,d), (d,c), (c,a)],
                [(a,c), (c,b), (b,d), (d,a)]
            ]
            
            for cycle in cycles:
                # Check all 4 edges exist
                colors = []
                valid = True
                for (u, v) in cycle:
                    key = (min(u,v), max(u,v))
                    if key not in edge_colors:
                        valid = False
                        break
                    colors.append(edge_colors[key])
                
                if valid and len(set(colors)) < 4:
                    return False
        
        return True
    
    # Generate factorization and assign colors
    matchings = round_robin_factorization(n)
    
    if not verify_factorization(matchings, n):
        # Fallback: just return a star
        return [[0, v, 0] for v in range(1, n)]
    
    # Assign color i to matching i
    edge_colors = {}
    for i, matching in enumerate(matchings):
        for (u, v) in matching:
            key = (min(u,v), max(u,v))
            edge_colors[key] = i
    
    # Check if this factorization is already valid
    if check_rainbow_4cycles(edge_colors, n):
        # Great! Return all edges
        result = []
        for (u, v), c in edge_colors.items():
            result.append([u, v, c])
        return result
    
    # If not valid, we need to remove edges to fix violations
    # Strategy: iteratively find violating 4-cycles and remove the most critical edge
    
    def find_violating_4cycle(edge_colors, n):
        """Find one violating 4-cycle, return its edges or None."""
        adj = {v: {} for v in range(n)}
        for (u, v), c in edge_colors.items():
            adj[u][v] = c
            adj[v][u] = c
        
        for a, b, c, d in combinations(range(n), 4):
            cycles = [
                [(a,b), (b,c), (c,d), (d,a)],
                [(a,b), (b,d), (d,c), (c,a)],
                [(a,c), (c,b), (b,d), (d,a)]
            ]
            
            for cycle in cycles:
                colors = []
                valid = True
                for (u, v) in cycle:
                    key = (min(u,v), max(u,v))
                    if key not in edge_colors:
                        valid = False
                        break
                    colors.append(edge_colors[key])
                
                if valid and len(set(colors)) < 4:
                    return cycle
        
        return None
    
    # Iteratively remove edges to fix violations
    # To maximize edges, remove the edge that appears in the most violations
    # But that's expensive. Let's just remove one edge per violation iteratively.
    
    max_iterations = 5000
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        
        cycle = find_violating_4cycle(edge_colors, n)
        if cycle is None:
            break
        
        # Remove one edge from this cycle
        # Heuristic: remove the edge whose removal breaks the most 4-cycles
        # For simplicity, just remove the first edge
        # Better: count how many times each edge in this cycle appears in violations
        # For now, just remove a random edge from the cycle
        edge_to_remove = cycle[0]
        key = (min(edge_to_remove[0], edge_to_remove[1]), max(edge_to_remove[0], edge_to_remove[1]))
        del edge_colors[key]
    
    # Build result
    result = []
    for (u, v), c in edge_colors.items():
        result.append([u, v, c])
    
    return result

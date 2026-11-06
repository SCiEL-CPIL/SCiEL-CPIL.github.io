import time
from itertools import combinations

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Build projective plane of order 5: PG(2,5)
    # 31 points, 31 lines, each line has 6 points, each point on 6 lines
    # Two points determine exactly one line => no C4 (any two vertices share at most 1 common neighbor)
    # Independence number = 6 (no line has 2 collinear points in an independent set... 
    # actually independent set = set of points with no two on same line = cap set)
    # In PG(2,q), max cap size for q=5 is 6.
    
    q = 5
    # Points of PG(2,q) represented as (x,y) affine + point at infinity
    # Total points: q^2 + q + 1 = 31
    
    # Generate all points
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y))
    # Points at infinity: one for each slope m in {0,1,...,q-1} plus vertical
    for m in range(q):
        points.append((m, 'inf'))
    points.append(('vert', 'inf'))
    
    n_pts = len(points)  # 31
    point_idx = {p: i for i, p in enumerate(points)}
    
    # Generate all lines
    # Lines: y = mx + b for m in 0..q-1, b in 0..q-1 (q^2 lines)
    # Plus vertical lines x = c for c in 0..q-1 (q lines)
    # Plus line at infinity (1 line)
    # Total: q^2 + q + 1 = 31 lines
    
    lines = []
    for m in range(q):
        for b in range(q):
            line_pts = []
            for x in range(q):
                y = (m * x + b) % q
                line_pts.append((x, y))
            line_pts.append((m, 'inf'))
            lines.append(line_pts)
    for c in range(q):
        line_pts = []
        for y in range(q):
            line_pts.append((c, y))
        line_pts.append(('vert', 'inf'))
        lines.append(line_pts)
    # Line at infinity
    line_inf = [(m, 'inf') for m in range(q)] + [('vert', 'inf')]
    lines.append(line_inf)
    
    # Build adjacency: two points are adjacent if they are on the same line
    # In PG(2,q), any two distinct points are on exactly one line, so ALL pairs are adjacent!
    # Wait, that means the full graph is complete K_31, which is C4-free but alpha=1.
    # That's not useful. We need the complement: the "non-collinear" relation isn't standard.
    
    # Actually, for C4-free: we need no two vertices with 2 common neighbors.
    # In the line graph or point graph of PG(2,q):
    # Point graph: vertices = points, edge if collinear on a line. But all pairs are collinear.
    # 
    # Better approach: Use the incidence structure differently.
    # Let's use a different C4-free graph: the polarity graph of PG(2,q) or
    # simply use a known C4-free graph construction.
    
    # Alternative: Use the complement of the line graph, or use a strongly regular graph.
    # 
    # Actually, let me reconsider. A C4-free graph with good properties:
    # - The Erdos-Renyi graph / random C4-free graphs
    # - Or use a projective plane's "non-adjacency" in a specific way
    
    # Let me use a simpler approach: start with 10 disjoint triangles (n=30, alpha=10)
    # and try to add vertices by connecting them to existing vertices while
    # maintaining C4-freeness and keeping alpha < 11.
    
    # Start with 10 disjoint triangles
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    
    n = 30
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Precompute common neighbor counts for C4 checking
    def has_c4(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = 0
                for w in range(n):
                    if adj[u][w] and adj[v][w]:
                        common += 1
                        if common > 1:
                            return True
        return False
    
    def check_alpha_less_than_11(adj, n):
        """Check that there's no independent set of size 11."""
        # Use greedy + backtracking for small n
        # For n up to ~43, use a branch and bound
        
        # Quick check: if n < 11, alpha < 11 trivially
        if n < 11:
            return True
        
        # Use a simple branch and bound
        # Sort vertices by degree (high degree first to reduce branching)
        degrees = [sum(adj[v]) for v in range(n)]
        order = sorted(range(n), key=lambda v: -degrees[v])
        
        best = [0]
        
        def backtrack(start, current_size, excluded):
            if time.time() > deadline - 10:
                return
            if current_size >= 11:
                best[0] = 11
                return
            remaining = n - start
            if current_size + remaining <= best[0]:
                return
            if start == n:
                best[0] = max(best[0], current_size)
                return
            # Try including vertex order[start]
            v = order[start]
            if not any(excluded[v]):
                # Can we include v? Check it's not adjacent to any in current independent set
                # Actually we need to track the current set
                pass
            # This is getting complex. Let me use a simpler approach.
        
        # Simpler: use maximum independent set via branch and bound
        # For our purposes, we just need to verify alpha < 11
        # Since we're adding vertices carefully, let's use a heuristic check
        
        # Greedy lower bound on alpha
        alpha_lb = 0
        remaining = set(range(n))
        while remaining:
            # Pick vertex with minimum degree in remaining subgraph
            min_deg = float('inf')
            min_v = None
            for v in remaining:
                deg_in_remaining = sum(1 for u in remaining if not adj[v][u])
                if deg_in_remaining < min_deg:
                    min_deg = deg_in_remaining
                    min_v = v
            alpha_lb += 1
            # Remove min_v and all its non-neighbors from remaining
            remaining.discard(min_v)
            to_remove = set()
            for u in remaining:
                if not adj[min_v][u]:
                    to_remove.add(u)
            remaining -= to_remove
        
        if alpha_lb >= 11:
            return False
        
        # For a more precise check with limited time
        # Try a simple backtracking to find independent set of size 11
        if n <= 43:
            # Use a quick backtracking with pruning
            def find_indep_set_11(vertices, size):
                if size >= 11:
                    return True
                if not vertices:
                    return False
                if len(vertices) < 11 - size:
                    return False
                # Pick first vertex
                v = vertices[0]
                rest = vertices[1:]
                # Option 1: include v, then can only include non-neighbors of v from rest
                non_neighbors = [u for u in rest if not adj[v][u]]
                if find_indep_set_11(non_neighbors, size + 1):
                    return True
                # Option 2: exclude v
                if find_indep_set_11(rest, size):
                    return True
                return False
            
            # This could be exponential, so limit it
            # Only do this if n is small enough
            if n <= 35:
                if find_indep_set_11(list(range(n)), 0):
                    return False
        
        return True
    
    # Try to add vertices one by one
    for new_v in range(n, 43):
        if time.time() > deadline - 20:
            break
        
        found = False
        # Try different connection patterns for the new vertex
        # The new vertex can connect to a subset of existing vertices
        # Constraint: for any two existing vertices u,v that both connect to new_v,
        # they must not have a common neighbor other than new_v (and each other if adjacent)
        # More precisely: if new_v connects to u and v, then u and v must have at most 
        # 1 common neighbor in the existing graph (since new_v would be a second common neighbor)
        
        # Strategy: try to connect new_v to as few vertices as possible while
        # keeping alpha < 11
        
        # First try: connect new_v to no one (isolated vertex)
        # This increases alpha by 1, so if alpha was 10, now it's 11 -> bad
        # So we need to connect it to at least enough vertices to reduce alpha
        
        # Try connecting to one vertex from each triangle (to keep alpha bounded)
        # If we connect new_v to one vertex in each of the 10 triangles, 
        # the max independent set can still pick 1 from each triangle, but 
        # if new_v is in the independent set, it can't pick its neighbors.
        
        # Let's try a greedy approach: for each possible set of neighbors,
        # check C4-freeness and alpha constraint
        
        # Simple heuristic: connect new_v to vertices to minimize alpha impact
        # while maintaining C4-free
        
        # Try: connect new_v to exactly one vertex per triangle
        # There are 3^10 = 59049 possibilities, too many to enumerate fully
        # Use greedy: for each triangle, pick the vertex that causes least C4 issues
        
        # Actually, let's try a simpler pattern:
        # Connect new_v to vertex 0 of each triangle: {0, 3, 6, 9, ...}
        # Check C4: new_v is adjacent to 0,3,6,... 
        # For any pair like (0,3): they're in different triangles, no common neighbors in base graph
        # So adding new_v gives them exactly 1 common neighbor (new_v) -> OK
        # For pair (0,1): both in triangle 0. 0 is neighbor of new_v, 1 is not.
        # Common neighbors of 0 and 1: vertex 2 (in triangle 0). That's 1. OK.
        # So this should be C4-free!
        
        # Let's verify: new_v connects to {0, 3, 6, ..., 27}
        # Any two of these vertices are in different triangles, so they have no common neighbors
        # in the base graph. Adding new_v gives them exactly 1 common neighbor. C4-free.
        
        # Now check alpha: In the original graph, alpha = 10 (one from each triangle).
        # With new_v connected to 0,3,6,...,27:
        # If new_v is in the independent set, we can't pick 0,3,6,...,27.
        # From triangle 0, we can pick 1 or 2 (not 0)
        # From triangle 1, we can pick 4 or 5 (not 3)
        # etc.
        # So max independent set containing new_v: 1 + 1*10 = 11? No wait,
        # from each triangle we can still pick 1 vertex (just not the one connected to new_v)
        # So alpha = 1 + 10 = 11. That's too high!
        
        # We need to connect new_v to at least 2 vertices in some triangles
        # to reduce the max independent set
        
        # If new_v connects to 2 vertices in triangle t, then if new_v is in IS,
        # we can only pick 1 vertex from triangle t (the one not connected to new_v)
        # Wait, we can still pick 1 vertex from each triangle regardless.
        # The issue is: if new_v is in IS, we exclude its neighbors.
        # From triangle t, if new_v connects to 2 vertices, we can pick the 3rd.
        # If new_v connects to 1 vertex, we can pick either of the other 2.
        # Either way, we can pick 1 from each triangle.
        # So alpha containing new_v is always 1 + 10 = 11.
        
        # Therefore, an isolated or lightly connected vertex will always give alpha >= 11.
        # We need a different approach: modify the base graph or use a different structure.
        
        # Alternative: use a base graph with alpha < 10, then we have room to add vertices.
        # Or: don't add isolated vertices, but merge/restructure.
        
        # Let me try a different base: 9 disjoint triangles (n=27, alpha=9)
        # Then add 6 vertices, each connected to 2 vertices in different triangles
        # to keep alpha < 11.
        
        # Actually, let me just return the 30-vertex solution as a safe baseline
        # and try to improve within the time budget.
        
        # For now, return the 30-vertex graph
        break
    
    # Validate the 30-vertex graph
    assert not has_c4(adj, n), "C4 detected!"
    
    return {'n': n, 'edges': edges}

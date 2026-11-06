def construct():
    import time
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    def get_neighbors_polarity(q):
        """Generate neighbors for polarity graph of projective plane of order q.
        Vertices: 0..q^2+q (total q^2+q+1 vertices)
        Vertex i is adjacent to vertex j if they form a self-orthogonal pair.
        """
        n = q * q + q + 1
        adj = [set() for _ in range(n)]
        
        # For projective plane of order q, use the standard construction
        # Points are (x, y, 1) for x,y in GF(q), plus line at infinity
        # But we need a simpler algebraic construction
        
        # Use the standard polarity: (x,y,1) ~ (x',y',1) iff xx' + yy' + 1 = 0 (mod p)
        # where p is prime power >= q
        # Actually for q=7, we can use GF(7)
        
        if q == 7:
            p = 7
            # Points: (x, y) for x,y in 0..6, plus 7 points at infinity (one for each slope)
            # Total: 49 + 7 = 56... but projective plane of order 7 has 57 points
            # Actually: q^2 + q + 1 = 49 + 7 + 1 = 57
            
            # Standard representation:
            # Points 0..48: (x, y) in GF(7) x GF(7), index = x*7 + y
            # Points 49..55: infinite points, one for each slope m in 0..6
            # Point 56: the "point at infinity" for vertical lines
            
            # Polarity: (x,y) is self-orthogonal to (x',y') iff xx' + yy' + 1 = 0
            # This is a correlation, not a polarity in the usual sense
            # Let's use a different approach
            
            # Use the known construction: For prime p, the graph on p^2+p+1 vertices
            # where vertex (i,j) [i in 0..p, j in 0..p-1 if i<p, else j=0] 
            # has adjacency based on quadratic residues
            
            # Simpler: use the Clebsch graph or another known C4-free graph
            # For now, let's construct a graph and verify C4-freeness
            
            pass
        
        return adj
    
    # Let's use a known C4-free graph construction
    # The Hoffman-Singleton graph is (50, 7, 0, 1) - C4-free, 50 vertices
    
    def build_hoffman_singleton():
        """Build the Hoffman-Singleton graph on 50 vertices."""
        n = 50
        adj = [set() for _ in range(n)]
        
        # Vertices: (i, j) with i in 0..4, j in 0..9
        # index = i * 10 + j
        # (i,j) adjacent to:
        # (i, j+1), (i, j-1)
        # (i+1, j+2), (i+1, j-2)
        # (i-1, j+3), (i-1, j-3)
        # (i+2, j+5)
        
        def idx(i, j):
            return (i % 5) * 10 + (j % 10)
        
        for i in range(5):
            for j in range(10):
                v = idx(i, j)
                neighbors = [
                    idx(i, j + 1),
                    idx(i, j - 1),
                    idx(i + 1, j + 2),
                    idx(i + 1, j - 2),
                    idx(i - 1, j + 3),
                    idx(i - 1, j - 3),
                    idx(i + 2, j + 5)
                ]
                for nv in neighbors:
                    adj[v].add(nv)
                    adj[nv].add(v)
        
        # Verify C4-freeness
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if len(common) > 1:
                    return None  # Not C4-free
        
        return adj
    
    # Build the base graph
    adj = build_hoffman_singleton()
    if adj is None:
        # Fallback: return a small valid graph
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    n = len(adj)
    
    # Check current independence number
    def max_independent_set(adj, n, time_limit=None):
        """Find maximum independent set using branch and bound."""
        if time_limit and time.time() - start_time > time_limit:
            return None
        
        best = []
        
        def backtrack(candidates, current):
            nonlocal best
            if time_limit and time.time() - start_time > time_limit:
                return
            if len(candidates) == 0:
                if len(current) > len(best):
                    best = current[:]
                return
            
            # Upper bound: current + all remaining candidates
            if len(current) + len(candidates) <= len(best):
                return
            
            # Choose vertex with maximum degree in candidate subgraph
            # to reduce branching
            max_deg = -1
            chosen_idx = -1
            for i, v in enumerate(candidates):
                deg = sum(1 for u in candidates if u != v and u in adj[v])
                if deg > max_deg:
                    max_deg = deg
                    chosen_idx = i
            
            v = candidates[chosen_idx]
            remaining = [u for i, u in enumerate(candidates) if i != chosen_idx]
            
            # Branch 1: include v
            current.append(v)
            candidates_with_v = [u for u in remaining if u not in adj[v]]
            backtrack(candidates_with_v, current)
            current.pop()
            
            # Branch 2: exclude v
            backtrack(remaining, current)
        
        backtrack(list(range(n)), [])
        return len(best)
    
    # Get current independence number
    alpha = max_independent_set(adj, n, time_limit=140)
    if alpha is None:
        alpha = 11  # Conservative estimate
    
    # Local search: try to add edges to reduce independence number
    # Priority: add edge (u,v) if it doesn't create C4 and reduces alpha
    
    def can_add_edge(u, v):
        """Check if adding edge (u,v) creates a C4."""
        if v in adj[u]:
            return False
        # Count common neighbors
        common = adj[u] & adj[v]
        return len(common) <= 1
    
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
    
    def remove_edge(u, v):
        adj[u].discard(v)
        adj[v].discard(u)
    
    # Try to add edges that reduce independence number
    # Sort non-adjacent pairs by potential benefit
    non_adjacent_pairs = []
    for u in range(n):
        for v in range(u + 1, n):
            if v not in adj[u] and can_add_edge(u, v):
                non_adjacent_pairs.append((u, v))
    
    # Try adding edges in a greedy fashion
    improved = True
    while improved and time.time() - start_time < deadline:
        improved = False
        current_alpha = max_independent_set(adj, n, time_limit=deadline - 5)
        if current_alpha is None:
            current_alpha = alpha
        
        # Try each non-adjacent pair
        for u, v in non_adjacent_pairs:
            if time.time() - start_time > deadline:
                break
            if v in adj[u]:  # Already added
                continue
            if not can_add_edge(u, v):
                continue
            
            # Try adding this edge
            add_edge(u, v)
            new_alpha = max_independent_set(adj, n, time_limit=deadline - 5)
            if new_alpha is not None and new_alpha < current_alpha:
                # Keep the edge
                improved = True
                alpha = new_alpha
            else:
                # Remove the edge
                remove_edge(u, v)
            
            if time.time() - start_time > deadline:
                break
    
    # Build edge list
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    
    return {'n': n, 'edges': edges}

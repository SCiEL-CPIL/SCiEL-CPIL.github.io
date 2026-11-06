"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph with no K5 and no independent set of size 5.
    Strategy: Start with a known (5,5)-free graph (Paley on 37 vertices) and
    attempt to add vertices via a deterministic local search heuristic.
    We try to find a 37-vertex base and then add vertices one by one,
    checking constraints. If we can't add to the Paley graph, we try other
    constructions or combinations.
    
    However, the best known is 42 (Exoo). The Paley graph on 37 is a good seed.
    We will attempt to extend it. If extension fails or is too slow, we fall back
    to the Paley graph or a known 42-vertex construction if we can encode it.
    
    Since encoding the full 42-vertex graph is complex and error-prone without the
    exact edge list, we will focus on a robust algorithm that tries to grow the graph.
    
    Approach:
    1. Start with Paley(37).
    2. Try to add a new vertex. For each potential new vertex, we can define its
       adjacency to existing vertices. We want to maximize the number of edges
       while keeping max clique < 5 and max independent set < 5.
       Actually, a simpler heuristic: try to add a vertex that is connected to
       a specific subset of vertices.
       
    Let's try a different, more reliable approach:
    Use the known fact that the Paley graph on 37 is (5,5)-free.
    Can we combine two Paley graphs? No, disjoint union creates large independent sets.
    
    Let's try to implement a simple vertex addition algorithm.
    Given a graph G, can we add a vertex v such that G+{v} is (5,5)-free?
    v must not create a K5. So v's neighbors in G must not contain a K4.
    v must not create an independent set of size 5. So v's non-neighbors in G
    must not contain an independent set of size 4.
    
    We can iterate over possible neighborhoods for v. But there are 2^37 possibilities.
    Instead, we can use a greedy approach:
    Try to add v connected to all vertices? No, that creates a K5 if G has a K4.
    Try to add v connected to no vertices? No, that creates an independent set of size 5 if G has an independent set of size 4.
    
    Let's check if Paley(37) has any K4 or independent set of size 4.
    If it does, we might be able to add a vertex carefully.
    
    Actually, a known result is that the Paley graph on 37 is "maximal" in some sense?
    Or maybe we can use a different base.
    
    Let's try to construct the 42-vertex graph directly if possible, or use a
    heuristic that is fast enough.
    
    Alternative: Use the Schläfli graph? It has 27 vertices.
    Can we add vertices to the Schläfli graph?
    
    Let's stick to the Paley graph but try to "blow up" vertices or add vertices.
    
    Here is a strategy:
    1. Generate Paley(37).
    2. Check if we can add a 38th vertex.
       For a new vertex v, let N(v) be its neighborhood.
       Condition 1: N(v) contains no K4.
       Condition 2: V \ (N(v) ∪ {v}) contains no independent set of size 4.
       
    We can try a few simple patterns for N(v):
    - N(v) = all vertices: Fails if G has K4.
    - N(v) = empty: Fails if G has independent set of size 4.
    - N(v) = a random subset?
    
    Let's write a helper to check if a set of vertices contains a K4 or independent set of size 4.
    For 37 vertices, checking all subsets of size 4 is C(37,4) ~ 66k, which is fast.
    
    We can try to find a valid N(v) by starting with a random subset and adjusting,
    or by trying specific symmetric subsets.
    
    However, 300 seconds is a lot. We can afford to search for a 42-vertex graph if we
    have a good heuristic.
    
    Let's try this:
    Start with Paley(37).
    Try to add vertices one by one.
    For each new vertex, we try to find a neighborhood S such that:
      1. S contains no K4.
      2. V\S contains no independent set of size 4.
    
    We can iterate through all subsets S of V? 2^37 is too large.
    But we can use a greedy repair algorithm.
    Start with S = V (connect to all). Check for K4 in S. If found, remove vertices from S
    until no K4 remains. Then check for independent set of size 4 in V\S. If found,
    add vertices back to S (i.e., remove from V\S) until no independent set of size 4 remains.
    Iterate this process.
    
    Let's implement this heuristic.
    """
    
    import itertools
    import random
    
    def has_k4(vertices, adj_matrix):
        """Check if the induced subgraph on 'vertices' contains a K4."""
        verts = list(vertices)
        for combo in itertools.combinations(verts, 4):
            if all(adj_matrix[combo[i]][combo[j]] for i in range(4) for j in range(i+1, 4)):
                return True
        return False
    
    def has_ind4(vertices, adj_matrix):
        """Check if the induced subgraph on 'vertices' contains an independent set of size 4."""
        verts = list(vertices)
        for combo in itertools.combinations(verts, 4):
            if not any(adj_matrix[combo[i]][combo[j]] for i in range(4) for j in range(i+1, 4)):
                return True
        return False
    
    def is_valid_graph(n, edges):
        """Check if the graph has no K5 and no independent set of size 5."""
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # Check K5
        for combo in itertools.combinations(range(n), 5):
            if all(adj[combo[i]][combo[j]] for i in range(5) for j in range(i+1, 5)):
                return False
        
        # Check Independent Set of size 5
        for combo in itertools.combinations(range(n), 5):
            if not any(adj[combo[i]][combo[j]] for i in range(5) for j in range(i+1, 5)):
                return False
        return True

    # Base: Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    n = q
    
    # Build adjacency matrix
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Function to add a vertex with neighborhood S
    def try_add_vertex(adj, n, edges):
        """
        Try to add a new vertex to the graph.
        Returns (new_adj, new_n, new_edges) or None if failed.
        """
        # We need to find a subset S of {0..n-1} such that:
        # 1. S contains no K4
        # 2. V\S contains no independent set of size 4
        
        # Heuristic: Start with S = V, remove vertices to break K4s.
        # Then check V\S for independent sets, add vertices to S to break them.
        
        S = set(range(n))
        
        # Step 1: Remove vertices from S to eliminate K4s in S
        changed = True
        while changed:
            changed = False
            # Find a K4 in S
            verts = list(S)
            for combo in itertools.combinations(verts, 4):
                if all(adj[combo[i]][combo[j]] for i in range(4) for j in range(i+1, 4)):
                    # Found a K4, remove one vertex from it
                    # Which one? Maybe the one with highest degree in S?
                    # For now, just remove the first one.
                    S.remove(combo[0])
                    changed = True
                    break
            if changed:
                continue
                
        # Step 2: Check V\S for independent sets of size 4
        # If found, we need to add vertices to S (i.e., connect new vertex to them)
        # to break the independent set.
        # Wait, if V\S has an independent set of size 4, adding a vertex v connected to
        # one of them doesn't break the independence of the other 4?
        # No, the independent set is in the original graph. The new vertex v is NOT in V\S.
        # The condition is: The set of non-neighbors of v (which is V\S) must not contain
        # an independent set of size 4 in the original graph G.
        # So if V\S contains an independent set of size 4 in G, then adding v with
        # non-neighborhood V\S will create an independent set of size 5 (the 4 from G plus v).
        # So we MUST ensure V\S has no independent set of size 4 in G.
        
        # So if V\S has an independent set of size 4, we must move at least one vertex from
        # V\S to S.
        
        changed = True
        while changed:
            changed = False
            non_neighbors = set(range(n)) - S
            verts = list(non_neighbors)
            for combo in itertools.combinations(verts, 4):
                if not any(adj[combo[i]][combo[j]] for i in range(4) for j in range(i+1, 4)):
                    # Found an independent set of size 4 in non_neighbors
                    # Move one vertex to S
                    S.add(combo[0])
                    changed = True
                    break
            if changed:
                continue
        
        # Now S is a candidate.
        # We must re-verify that S has no K4, because adding vertices to S might have
        # created a K4? No, S only grows in Step 2. If S had no K4 after Step 1,
        # and we add vertices, we might create a K4.
        # So we need to iterate Steps 1 and 2 until stable.
        
        stable = False
        while not stable:
            stable = True
            # Check S for K4
            verts = list(S)
            for combo in itertools.combinations(verts, 4):
                if all(adj[combo[i]][combo[j]] for i in range(4) for j in range(i+1, 4)):
                    S.remove(combo[0])
                    stable = False
                    break
            
            if not stable:
                continue
                
            # Check V\S for Ind4
            non_neighbors = set(range(n)) - S
            verts = list(non_neighbors)
            for combo in itertools.combinations(verts, 4):
                if not any(adj[combo[i]][combo[j]] for i in range(4) for j in range(i+1, 4)):
                    S.add(combo[0])
                    stable = False
                    break
                    
        if S is None or len(S) == 0:
            return None
            
        # Construct new graph
        new_n = n + 1
        new_adj = [[False]*new_n for _ in range(new_n)]
        for i in range(n):
            for j in range(n):
                new_adj[i][j] = adj[i][j]
            new_adj[i][n] = (i in S)
            new_adj[n][i] = (i in S)
            
        new_edges = list(edges)
        for i in S:
            new_edges.append((i, n))
            
        # Verify the new graph is (5,5)-free
        # This is expensive, but n is small (38, 39, ...)
        if is_valid_graph(new_n, new_edges):
            return new_adj, new_n, new_edges
        else:
            return None

    # Try to add vertices
    # We might need to try multiple times or different heuristics if it fails.
    # Let's try to add up to 5 vertices.
    for _ in range(5):
        result = try_add_vertex(adj, n, edges)
        if result:
            adj, n, edges = result
        else:
            break
            
    return n, edges

# EVOLVE-BLOCK-END

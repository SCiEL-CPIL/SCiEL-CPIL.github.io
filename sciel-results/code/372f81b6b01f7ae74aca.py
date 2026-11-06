import random

def construct():
    n = 16
    # Use a fixed seed for reproducibility in the search
    random.seed(42)
    
    # 1. Base Graph: Grötzsch Graph (Mycielski of C5)
    # Vertices 0-4: C5 cycle
    # Vertices 5-9: Copies of 0-4
    # Vertex 10: Root
    edges = []
    # C5
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    # Mycielski step
    for i in range(5):
        u = i
        v = (i + 1) % 5
        edges.append([u, 5 + v])
        edges.append([v, 5 + u])
    for i in range(5):
        edges.append([5 + i, 10])
        
    # Current vertex count: 11
    # Remaining vertices: 11, 12, 13, 14, 15
    
    # We will try to add these vertices one by one.
    # For each new vertex, we can connect it to any subset of existing vertices
    # as long as no triangle is formed and the graph remains 4-chromatic (not 3-colorable).
    
    # To ensure "not 3-colorable", we must maintain the property.
    # Since the base is 4-chromatic, adding vertices can potentially allow a 3-coloring 
    # if the new vertices don't "force" a conflict with the existing structure 
    # or if they allow recoloring. However, usually, adding vertices with high degree 
    # to a non-3-colorable graph keeps it non-3-colorable, especially if they connect to
    # vertices that are forced to different colors in any 3-coloring of the base.
    
    # Wait, if we just add vertices, the graph might become 3-colorable?
    # No, if G is not 3-colorable, G+V (adding vertices and edges) is not necessarily not 3-colorable.
    # Example: If G requires 4 colors, but we add a vertex connected to nothing, it's still 4-chromatic.
    # If we add a vertex connected to a triangle? No triangles allowed.
    # Actually, any supergraph of a non-3-colorable graph is non-3-colorable?
    # Yes. If G is not 3-colorable, there is no mapping f: V(G) -> {1,2,3} such that all edges are valid.
    # If G' is a supergraph (V(G) subset V(G'), E(G) subset E(G')), any valid 3-coloring of G' restricted to V(G)
    # would be a valid 3-coloring of G. Since none exists, G' is not 3-colorable.
    # So we just need to ensure Triangle-Free.
    
    # Goal: Maximize edges while keeping it triangle-free.
    # Strategy: Greedy addition of vertices with maximum possible edges to existing vertices
    # subject to triangle-freedom.
    
    # Helper to check if adding edge (u, v) creates a triangle
    def creates_triangle(u, v, edge_set, adj):
        # Check common neighbors of u and v
        common = set(adj[u]) & set(adj[v])
        if common:
            return True
        return False

    # Initialize adjacency
    adj = {i: set() for i in range(n)}
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(frozenset((u, v)))
        
    current_edges = len(edge_set)
    
    # Vertices to add: 11, 12, 13, 14, 15
    new_vertices = [11, 12, 13, 14, 15]
    
    for v in new_vertices:
        # Determine which existing vertices (and previously added new vertices) can connect to v
        # without creating a triangle.
        # A vertex u can connect to v if N(u) and N(v) are disjoint (excluding each other).
        # Initially N(v) is empty.
        # As we add edges to v, N(v) grows.
        
        # We want to maximize degree of v.
        # This is a maximum independent set problem on the "forbidden" graph?
        # Or simply: We can connect v to any subset S of existing vertices such that
        # no two vertices in S are adjacent (because if u1, u2 in S and u1-u2 is an edge,
        # then u1-u2-v is a triangle).
        # Wait, is that the only constraint?
        # Yes, for a single new vertex v, adding edges (v, u) for u in S creates a triangle
        # if and only if there exist u1, u2 in S such that (u1, u2) is an edge.
        # So S must be an independent set in the subgraph induced by existing vertices.
        
        # So, to maximize edges for v, we should connect v to a Maximum Independent Set (MIS) of the current graph.
        
        # Finding MIS is NP-hard, but n is small (max 15). We can use a simple backtracking or approximation.
        # Given time limit 120s, we can afford a decent heuristic or exact search for small n.
        
        # Let's implement a simple backtracking for MIS on the current graph (vertices 0..v-1).
        # Actually, we can just pick a large independent set.
        # Let's use a greedy approach with random restarts to find a large independent set.
        
        def find_large_independent_set(nodes):
            # Greedy with randomization
            best_size = 0
            best_set = set()
            # Try multiple random orders
            for _ in range(50):
                random.shuffle(nodes)
                current_set = set()
                for u in nodes:
                    # Check if u is adjacent to any node in current_set
                    if not (adj[u] & current_set):
                        current_set.add(u)
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set
            return best_set
        
        # The nodes available to connect to v are all existing vertices (0 to v-1)
        available_nodes = list(range(v))
        # Note: We can only connect to vertices that are already in the graph.
        # Since we are adding v sequentially, all 0..v-1 are in the graph.
        
        iset = find_large_independent_set(available_nodes)
        
        # Connect v to all in iset
        for u in iset:
            edges.append([v, u])
            adj[v].add(u)
            adj[u].add(v)
            edge_set.add(frozenset((v, u)))
            
        current_edges = len(edge_set)
        
    return edges

# EVOLVE-BLOCK-START
# (The code above is the implementation)
# EVOLVE-BLOCK-END

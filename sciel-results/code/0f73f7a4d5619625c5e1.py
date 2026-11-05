import itertools

def construct():
    n = 16
    # Start with a known Q3-free graph: K_{3,13} + C_{13} on B
    # Vertices 0,1,2 in A; 3..15 in B
    edges = []
    edge_set = set()
    
    # Add all edges between A and B
    for a in range(3):
        for b in range(3, 16):
            edges.append([a, b])
            edge_set.add((a, b))
            edge_set.add((b, a))
            
    # Add cycle C_{13} on B
    b_vertices = list(range(3, 16))
    for i in range(len(b_vertices)):
        u = b_vertices[i]
        v = b_vertices[(i + 1) % len(b_vertices)]
        edges.append([u, v])
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    # Now try to add more edges within B (and possibly within A, but A has only 3 vertices)
    # First, add all missing edges within A (triangle on 0,1,2)
    for u in range(3):
        for v in range(u+1, 3):
            if (u, v) not in edge_set:
                edges.append([u, v])
                edge_set.add((u, v))
                edge_set.add((v, u))
    
    # Now try to add edges within B greedily
    # Candidate edges: all pairs within B not already present
    candidate_edges = []
    for i in range(len(b_vertices)):
        for j in range(i+1, len(b_vertices)):
            u = b_vertices[i]
            v = b_vertices[j]
            if (u, v) not in edge_set:
                candidate_edges.append((u, v))
    
    # For each candidate edge, check if adding it creates a Q3 subgraph
    # We need an efficient check. Let's precompute adjacency for the current graph.
    
    def is_q3_free(edge_set, n):
        """Check if graph with given edge_set on n vertices is Q3-free."""
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            if u < v:
                adj[u].add(v)
                adj[v].add(u)
        
        # For each subset of 8 vertices, check if it induces Q3
        # This is expensive but n=16 is small enough with optimization
        # Actually, let's use a smarter approach:
        # Q3 is bipartite with parts of size 4.
        # Check all ways to pick 4 vertices as one part and 4 as the other.
        
        vertices = list(range(n))
        
        # Precompute for each 4-subset, the common neighbors
        # Actually, let's just enumerate all 8-subsets and check
        # Number of 8-subsets of 16 is C(16,8) = 12870, manageable
        
        for subset in itertools.combinations(vertices, 8):
            s = set(subset)
            # Check if the induced subgraph on s is isomorphic to Q3
            # Q3 has 12 edges, 8 vertices, each vertex degree 3
            # First check degrees
            degs = []
            for v in subset:
                d = len([u for u in s if (v, u) in edge_set or (u, v) in edge_set])
                degs.append(d)
            if sorted(degs) != [3]*8:
                continue
            
            # Check if it's bipartite and has 12 edges
            # Try to 2-color
            # Build local adjacency
            local_adj = {v: [] for v in subset}
            for v in subset:
                for u in subset:
                    if v != u and ((v, u) in edge_set or (u, v) in edge_set):
                        local_adj[v].append(u)
            
            # Try BFS coloring
            color = {}
            is_bipartite = True
            for start in subset:
                if start in color:
                    continue
                color[start] = 0
                queue = [start]
                while queue:
                    node = queue.pop(0)
                    for neighbor in local_adj[node]:
                        if neighbor not in color:
                            color[neighbor] = 1 - color[node]
                            queue.append(neighbor)
                        elif color[neighbor] == color[node]:
                            is_bipartite = False
                            break
                if not is_bipartite:
                    break
            
            if not is_bipartite:
                continue
            
            # Count edges
            edge_count = 0
            for v in subset:
                for u in local_adj[v]:
                    if v < u:
                        edge_count += 1
            
            if edge_count == 12:
                # Check if each vertex has exactly 3 neighbors in the same part... 
                # Actually, in Q3, each vertex is connected to 3 vertices in the opposite part.
                # Since it's bipartite and 3-regular with 8 vertices and 12 edges, it's Q3.
                return False
        
        return True
    
    # Greedily add candidate edges
    added = 0
    for (u, v) in candidate_edges:
        test_set = edge_set | {(u, v), (v, u)}
        if is_q3_free(test_set, n):
            edge_set.add((u, v))
            edge_set.add((v, u))
            edges.append([u, v])
            added += 1
    
    return edges

        def _has_q3(adj_matrix):
            # 1. Identify edges in triangles
            # adj_matrix is NxN boolean
            # Edge (u,v) is in triangle if exists w s.t. adj[u][v] and adj[u][w] and adj[v][w]
            
            # Build list of edges to check
            edges_to_check = []
            for u in range(_N):
                for v in range(u+1, _N):
                    if adj_matrix[u][v]:
                        # Check if (u,v) is in a triangle
                        in_triangle = False
                        # Common neighbors of u and v
                        # Use bitmasks for speed?
                        # For now, simple loop
                        for w in range(_N):
                            if w != u and w != v and adj_matrix[u][w] and adj_matrix[v][w]:
                                in_triangle = True
                                break
                        if not in_triangle:
                            edges_to_check.append((u,v))
            
            # If no edges left, no Q3
            if not edges_to_check:
                return False
            
            # Build adjacency for G_noTri
            # We only care about edges in edges_to_check
            # But the backtracking needs to know if an edge exists in G_noTri.
            # Let's create a set of valid edges
            valid_edges = set(edges_to_check)
            
            # Backtracking on Q3
            # ...
        
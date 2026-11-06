<<<<<<< SEARCH
    def structured_initial():
        """Create a structured initial graph using a balanced bipartite-like construction."""
        adj = [0] * n
        
        # Use a construction based on projective plane or similar
        # Partition into two sets of 10, create bipartite graph with controlled degrees
        left = list(range(10))
        right = list(range(10, 20))
        
        # Create a bipartite graph where each left vertex connects to 4 right vertices
        # and each right vertex connects to 4 left vertices, with no two left vertices
        # sharing more than 2 right neighbors (and vice versa)
        
        # Use a 10x10 matrix with row/col sum 4 and pairwise intersection <= 2
        # This is equivalent to a 2-(10,4,2) design or similar
        
        # Simple approach: use cyclic shifts
        for i in range(10):
            for j in range(4):
                right_idx = (i * 4 + j) % 10
                u = left[i]
                v = right[right_idx]
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Add edges within left and right sets if possible
        for i in range(10):
            for j in range(i + 1, 10):
                u, v = left[i], left[j]
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        for i in range(10):
            for j in range(i + 1, 10):
                u, v = right[i], right[j]
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        return adj
=======
    def structured_initial():
        """Create a structured initial graph using a balanced bipartite-like construction."""
        adj = [0] * n
        
        # Use a construction based on projective plane or similar
        # Partition into two sets of 10, create bipartite graph with controlled degrees
        left = list(range(10))
        right = list(range(10, 20))
        
        # Create a bipartite graph where each left vertex connects to 4 right vertices
        # and each right vertex connects to 4 left vertices, with no two left vertices
        # sharing more than 2 right neighbors (and vice versa)
        
        # Use a 10x10 matrix with row/col sum 4 and pairwise intersection <= 2
        # This is equivalent to a 2-(10,4,2) design or similar
        
        # Simple approach: use cyclic shifts
        for i in range(10):
            for j in range(4):
                right_idx = (i * 4 + j) % 10
                u = left[i]
                v = right[right_idx]
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Add edges within left and right sets if possible
        for i in range(10):
            for j in range(i + 1, 10):
                u, v = left[i], left[j]
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        for i in range(10):
            for j in range(i + 1, 10):
                u, v = right[i], right[j]
                if is_k23_free(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        return adj
>>>>>>> REPLACE


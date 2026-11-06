import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Try n=40 first
    for n in [40, 39, 38]:
        if time.time() > deadline:
            break
            
        # Search for a valid graph on n vertices
        best_edges = None
        for attempt in range(200): # Adjust based on time
            if time.time() > deadline:
                break
                
            # 1. Generate random graph
            p = 0.3 # Adjust density
            adj = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i, j] = True
                        adj[j, i] = True
                        
            # 2. Remove triangles
            # While there are triangles, remove a random edge from one
            max_iter = 1000
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                # Find a triangle
                tri_found = False
                # Sample random edges to find triangles faster?
                # Or just scan. n=40 is small.
                # Check if any triangle exists
                # A triangle exists if there is i<j<k with adj[i,j] and adj[j,k] and adj[i,k]
                # We can check by iterating i, then j, then k?
                # Or use matrix multiplication?
                # A^2[i,k] > 0 and adj[i,k] is False? No.
                # Common neighbor of i and k is i,j,k triangle if adj[i,j] and adj[j,k].
                # If adj[i,k] is also True, it's a triangle.
                
                # Fast check:
                # For each edge (i,k), if they have a common neighbor j, and adj[i,k] is True, it's a triangle.
                # Actually, if adj[i,k] is True, and they have a common neighbor, it's a triangle.
                # We want to find ANY triangle.
                
                # Let's just pick a random edge (i,k) and check if it's in a triangle.
                # If yes, remove a random edge from the triangle.
                
                # Generate list of edges
                edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j]]
                if not edges:
                    break
                    
                random.shuffle(edges)
                
                for i, k in edges:
                    # Check if (i,k) is in a triangle
                    # Common neighbors of i and k
                    common = np.where(adj[i] & adj[k])[0]
                    # Exclude i and k
                    common = common[common != i]
                    common = common[common != k]
                    
                    if len(common) > 0:
                        # Triangle exists: i, k, and some j in common
                        # Pick a random j
                        j = common[random.randint(0, len(common)-1)]
                        # Remove a random edge from (i,j), (j,k), (i,k)
                        # Let's remove (i,k)
                        adj[i, k] = False
                        adj[k, i] = False
                        tri_found = True
                        break
                        
                if not tri_found:
                    # No triangle found in this edge sample?
                    # This might mean no triangles, or we missed them.
                    # For n=40, scanning all edges is fast.
                    # Let's do a full scan to be sure.
                    has_tri = False
                    for i in range(n):
                        for k in range(i+1, n):
                            if adj[i, k]:
                                common = np.where(adj[i] & adj[k])[0]
                                common = common[(common != i) & (common != k)]
                                if len(common) > 0:
                                    has_tri = True
                                    break
                        if has_tri:
                            break
                    if not has_tri:
                        break
                    else:
                        # We found a triangle but didn't remove it in the sample loop?
                        # Just remove the one we found
                        # (This logic is getting messy, let's simplify)
                        pass
                        
            # Simplified Triangle Removal:
            # Repeat until no triangles:
            # 1. Find all triangles.
            # 2. If none, break.
            # 3. Remove a random edge from a random triangle.
            
            # Let's restart the triangle removal logic cleanly.
            adj = adj.copy()
            for _ in range(500):
                if time.time() > deadline:
                    break
                # Find a triangle
                tri = None
                for i in range(n):
                    for k in range(i+1, n):
                        if adj[i, k]:
                            common = np.where(adj[i] & adj[k])[0]
                            common = common[(common != i) & (common != k)]
                            if len(common) > 0:
                                j = common[0]
                                tri = (i, j, k)
                                break
                    if tri:
                        break
                if not tri:
                    break
                # Remove random edge from tri
                i, j, k = tri
                choice = random.randint(0, 2)
                if choice == 0:
                    adj[i, j] = False; adj[j, i] = False
                elif choice == 1:
                    adj[j, k] = False; adj[k, j] = False
                else:
                    adj[i, k] = False; adj[k, i] = False
                    
            # 3. Check Independent Set
            if not _has_independent_set_size_10(adj, n):
                # Valid!
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i, j]:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
                
    # Fallback to n=39 or known graph
    # ...

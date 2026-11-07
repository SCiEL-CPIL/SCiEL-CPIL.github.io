"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    deadline = time.time() + 50  # Safety margin
    
    # Build base adjacency
    adj = [set() for _ in range(n)]
    existing_edges = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        existing_edges.add((min(u, v), max(u, v)))
    
    def bfs_diameter(adj):
        """Compute diameter via BFS from each node."""
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > max_dist:
                            max_dist = dist[v]
                        q.append(v)
            if -1 in dist:
                return float('inf')
        return max_dist
    
    def creates_triangle(u, v, adj):
        """Check if adding edge (u,v) creates a triangle."""
        # If u and v share any common neighbor, it's a triangle
        # We can iterate through the smaller set
        if len(adj[u]) < len(adj[v]):
            for w in adj[u]:
                if w in adj[v]:
                    return True
        else:
            for w in adj[v]:
                if w in adj[u]:
                    return True
        return False

    added_edges = []
    
    # Greedy loop
    while time.time() < deadline:
        current_diam = bfs_diameter(adj)
        if current_diam <= 4:
            break
        
        best_edge = None
        best_diam = current_diam
        
        # Generate candidate edges
        # We can just iterate all pairs not in existing_edges
        # To speed up, we can skip pairs that are too close (dist < 2) as they definitely create triangles in a path-like graph,
        # but with added edges, dist 2 might not create a triangle if the middle node is not connected to both? 
        # Actually, if (u, u+1) and (u+1, u+2) are edges, then (u, u+2) creates a triangle.
        # So generally, we should check explicitly.
        
        # Optimization: Only consider edges that connect "far" apart nodes?
        # No, sometimes a short edge helps. But in a path, short edges create triangles easily.
        # Let's just iterate all valid pairs.
        
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) in existing_edges:
                    continue
                
                # Check triangle condition
                if creates_triangle(u, v, adj):
                    continue
                
                # Try adding
                adj[u].add(v)
                adj[v].add(u)
                
                # Calculate new diameter
                new_diam = bfs_diameter(adj)
                
                # Remove edge
                adj[u].remove(v)
                adj[v].remove(u)
                
                if new_diam < best_diam:
                    best_diam = new_diam
                    best_edge = (u, v)
                
                # Early break if we found a perfect diameter (4)
                if best_diam <= 4:
                    # We can't break out of nested loops easily, so we flag
                    pass
            
            # Optional: break inner loop if best_diam is already 4? 
            # No, we need to find the *best* edge to minimize count? 
            # Actually, we just need *any* edge that reduces diameter. 
            # To minimize the *number* of edges, we should pick the one that reduces diameter the most.
            # If multiple edges reduce to 4, any is fine for this step, but might affect future steps.
            # Greedy "max reduction" is a good heuristic.
        
        if best_edge is None:
            # No valid edge found to reduce diameter?
            # This should not happen for N=24, target diam 4.
            # If it does, we break to avoid infinite loop.
            break
        
        # Add the best edge
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        existing_edges.add((min(u, v), max(u, v)))
        added_edges.append([int(u), int(v)])
        
        # Check if we are done
        if bfs_diameter(adj) <= 4:
            break

    # Verify final solution
    if bfs_diameter(adj) <= 4:
        return added_edges
    else:
        # Fallback: return empty or best effort? 
        # The task says "Minimize the number of added edges". 
        # If we fail, we return what we have? 
        # "Any broken rule makes the whole result invalid". 
        # So we must return a valid solution or nothing? 
        # If we return an invalid solution, score is 0.
        # If we return [] and it's invalid, score is 0.
        # We should try hard to find a valid one.
        # Given the small size, the greedy should work.
        return added_edges

# EVOLVE-BLOCK-END

"""Degree-ordered edge-replacement heuristic for minimizing max triangle degree sum."""
import random
import itertools

def construct():
    n = 16
    target_edges = 65
    
    # Start with K_{8,8} + 1 edge: vertices 0-7 in A, 8-15 in B
    # Add edge (0,1) within A to get 65 edges
    adj = [set() for _ in range(n)]
    
    # Complete bipartite K_{8,8}
    for u in range(8):
        for v in range(8, 16):
            adj[u].add(v)
            adj[v].add(u)
    
    # One extra edge within A
    adj[0].add(1)
    adj[1].add(0)
    
    def get_degrees(adj):
        return [len(adj[i]) for i in range(n)]
    
    def get_triangles(adj):
        """Find all triangles efficiently."""
        triangles = []
        for i in range(n):
            for j in adj[i]:
                if j <= i:
                    continue
                common = adj[i] & adj[j]
                for k in common:
                    if k > j:
                        triangles.append((i, j, k))
        return triangles
    
    def max_tri_sum(triangles, degrees):
        if not triangles:
            return 0
        return max(degrees[u] + degrees[v] + degrees[w] for u, v, w in triangles)
    
    degrees = get_degrees(adj)
    best_score = max_tri_sum(get_triangles(adj), degrees)
    
    # Main optimization loop
    for iteration in range(5000):
        triangles = get_triangles(adj)
        if not triangles:
            break
        
        degrees = get_degrees(adj)
        current_max = max_tri_sum(triangles, degrees)
        
        # Find worst triangles
        worst_tris = [(u, v, w) for u, v, w in triangles 
                     if degrees[u] + degrees[v] + degrees[w] == current_max]
        
        if not worst_tris:
            break
        
        improved = False
        
        # For each worst triangle, try to reduce its degree sum
        for tri in worst_tris:
            u, v, w = tri
            
            # Identify the highest-degree vertex in this triangle
            tri_vertices = [u, v, w]
            tri_vertices.sort(key=lambda x: -degrees[x])
            high_vertex = tri_vertices[0]
            
            # Strategy: remove an edge incident to the high-degree vertex
            # and add an edge that doesn't create worse triangles
            # and reduces the overall max
            
            # Try removing each edge incident to high_vertex in this triangle
            incident_edges = []
            for other in [v, w]:
                if other in adj[high_vertex]:
                    incident_edges.append((high_vertex, other))
            
            for (a, b) in incident_edges:
                # Try to find a good non-edge to add
                # Prioritize non-edges between low-degree vertices
                # that are not adjacent to high_vertex (to avoid creating new bad triangles)
                
                best_swap = None
                best_new_max = current_max
                
                # Generate candidate non-edges, sorted by sum of degrees (lower is better)
                candidates = []
                for c in range(n):
                    for d in range(c + 1, n):
                        if d not in adj[c]:
                            # Skip if this would create a triangle with high_vertex
                            # that has worse sum
                            # Check: if c and d are both neighbors of high_vertex,
                            # then (high_vertex, c, d) would be a new triangle
                            new_tri_with_high = False
                            if c in adj[high_vertex] and d in adj[high_vertex]:
                                new_tri_with_high = True
                            
                            deg_sum = degrees[c] + degrees[d]
                            candidates.append((deg_sum, c, d, new_tri_with_high))
                
                # Sort by degree sum (prefer lower)
                candidates.sort()
                
                # Try top candidates
                for deg_sum, c, d, new_tri_with_high in candidates[:100]:
                    # Skip if this creates a triangle with the high vertex that's worse
                    if new_tri_with_high:
                        # The new triangle (high_vertex, c, d) would have sum:
                        # (degrees[high_vertex] - 1) + (degrees[c] + 1) + (degrees[d] + 1)
                        # = degrees[high_vertex] + degrees[c] + degrees[d] + 1
                        new_sum = degrees[high_vertex] + degrees[c] + degrees[d] + 1
                        if new_sum >= current_max:
                            continue
                    
                    # Simulate the swap
                    new_degrees = degrees[:]
                    new_degrees[a] -= 1
                    new_degrees[b] -= 1
                    new_degrees[c] += 1
                    new_degrees[d] += 1
                    
                    # Check if any existing triangle (after swap) has sum > best_new_max
                    # Only need to check triangles that include a, b, c, or d
                    affected = set([a, b, c, d])
                    new_max = 0
                    
                    for tri2 in triangles:
                        t = set(tri2)
                        if t & affected:
                            s = new_degrees[tri2[0]] + new_degrees[tri2[1]] + new_degrees[tri2[2]]
                            if s > new_max:
                                new_max = s
                    
                    # Also check new triangles created by adding (c, d)
                    common_cd = adj[c] & adj[d]
                    for x in common_cd:
                        s = new_degrees[c] + new_degrees[d] + new_degrees[x]
                        if s > new_max:
                            new_max = s
                    
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_swap = (c, d)
                
                if best_swap and best_new_max < current_max:
                    c, d = best_swap
                    # Perform the swap
                    adj[a].remove(b)
                    adj[b].remove(a)
                    adj[c].add(d)
                    adj[d].add(c)
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # Local optimum: try random perturbations
            if iteration % 5 == 0:
                # Random edge swap
                edge_list = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edge_list.append((u, v))
                
                non_edges = []
                for c in range(n):
                    for d in range(c + 1, n):
                        if d not in adj[c]:
                            non_edges.append((c, d))
                
                if edge_list and non_edges:
                    a, b = random.choice(edge_list)
                    c, d = random.choice(non_edges)
                    
                    # Only accept if it doesn't increase max too much
                    new_degrees = degrees[:]
                    new_degrees[a] -= 1
                    new_degrees[b] -= 1
                    new_degrees[c] += 1
                    new_degrees[d] += 1
                    
                    # Quick check: max degree sum shouldn't exceed current_max + 2
                    max_possible = max(new_degrees) * 3
                    if max_possible <= current_max + 3:
                        adj[a].remove(b)
                        adj[b].remove(a)
                        adj[c].add(d)
                        adj[d].add(c)
    
    # Final output
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    
    return {'n': n, 'edges': edges}

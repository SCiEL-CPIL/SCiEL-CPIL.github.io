"""Rewritten program using degree-ordered edge-replacement heuristic."""
import random
import itertools

def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _compute_degrees(adj, n):
    return [len(adj[i]) for i in range(n)]

def _find_triangles(adj, n):
    """Find all triangles in the graph."""
    triangles = []
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[j]:
                if k > j and k in adj[i]:
                    triangles.append((i, j, k))
    return triangles

def _max_triangle_degree_sum(triangles, degrees):
    if not triangles:
        return 0
    return max(degrees[u] + degrees[v] + degrees[w] for u, v, w in triangles)

def construct():
    n = 16
    # Start with K_{8,8} plus one edge inside part A (vertices 0-7)
    part_a = list(range(8))
    part_b = list(range(8, 16))
    
    edges = []
    for u in part_a:
        for v in part_b:
            edges.append([u, v])
    edges.append([0, 1])
    
    adj = _build_adjacency(n, edges)
    degrees = _compute_degrees(adj, n)
    
    # We need exactly 65 edges. K_{8,8} has 64 edges, plus 1 = 65. Good.
    
    # Iterative improvement
    for iteration in range(2000):
        triangles = _find_triangles(adj, n)
        if not triangles:
            break
        
        max_sum = _max_triangle_degree_sum(triangles, degrees)
        
        # Find the worst triangle (highest degree sum)
        worst_triangles = [(u, v, w) for u, v, w in triangles 
                         if degrees[u] + degrees[v] + degrees[w] == max_sum]
        
        if not worst_triangles:
            break
        
        # Try to improve by swapping edges
        improved = False
        
        for tri in worst_triangles:
            u, v, w = tri
            tri_edges = [(u, v), (v, w), (u, w)]
            
            # For each edge in the triangle, try to remove it and add a non-edge
            # that doesn't create new bad triangles
            for e_idx, (a, b) in enumerate(tri_edges):
                # Try removing edge (a, b)
                # Check if (a, b) is actually in the graph
                if b not in adj[a]:
                    continue
                
                # Find a non-edge (c, d) to add
                # We want to add an edge that helps reduce degree sums
                # Strategy: add an edge between two vertices with lower degrees
                # that are not already connected
                
                # Generate candidate non-edges
                candidates = []
                for c in range(n):
                    for d in range(c + 1, n):
                        if d not in adj[c]:
                            # Check that adding (c, d) doesn't create new triangles
                            # with worse degree sums
                            new_triangles_created = []
                            for x in adj[c] & adj[d]:
                                new_triangles_created.append((c, d, x))
                            
                            # Check if any new triangle has higher degree sum
                            # after the swap
                            new_degrees = degrees[:]
                            new_degrees[a] -= 1
                            new_degrees[b] -= 1
                            new_degrees[c] += 1
                            new_degrees[d] += 1
                            
                            # Check all triangles after swap
                            # This is expensive, so let's be smarter
                            # Only check if the new triangles are worse
                            is_better = True
                            for nt in new_triangles_created:
                                x1, x2, x3 = nt
                                new_sum = new_degrees[x1] + new_degrees[x2] + new_degrees[x3]
                                if new_sum > max_sum:
                                    is_better = False
                                    break
                            
                            if is_better:
                                # Also need to ensure we don't create a triangle 
                                # with sum > max_sum among existing triangles
                                # that are affected by degree changes
                                # For now, accept if no new triangle is worse
                                candidates.append((c, d, new_degrees))
                
                # Pick a candidate that minimizes the new max triangle degree sum
                best_candidate = None
                best_new_max = max_sum + 1
                
                for c, d, new_degrees in candidates[:50]:  # Limit for speed
                    # Compute new max triangle degree sum approximately
                    # Just check the triangles involving c or d
                    new_max = 0
                    for tri2 in triangles:
                        x1, x2, x3 = tri2
                        s = new_degrees[x1] + new_degrees[x2] + new_degrees[x3]
                        if s > new_max:
                            new_max = s
                    # Also check new triangles
                    for x in adj[c] & adj[d]:
                        s = new_degrees[c] + new_degrees[d] + new_degrees[x]
                        if s > new_max:
                            new_max = s
                    
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_candidate = (c, d)
                
                if best_candidate and best_new_max < max_sum:
                    c, d = best_candidate
                    # Perform the swap
                    adj[a].remove(b)
                    adj[b].remove(a)
                    adj[c].add(d)
                    adj[d].add(c)
                    
                    # Update edge list
                    edges = [[min(u, v), max(u, v)] for u in range(n) for v in adj[u] if u < v]
                    
                    degrees = _compute_degrees(adj, n)
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # Try a random perturbation to escape local optimum
            if iteration % 10 == 0:
                # Randomly swap an edge
                edge_list = [[min(u, v), max(u, v)] for u in range(n) for v in adj[u] if u < v]
                if edge_list:
                    idx = random.randint(0, len(edge_list) - 1)
                    a, b = edge_list[idx]
                    # Find a random non-edge
                    non_edges = []
                    for c in range(n):
                        for d in range(c + 1, n):
                            if d not in adj[c]:
                                non_edges.append((c, d))
                    if non_edges:
                        c, d = random.choice(non_edges)
                        adj[a].remove(b)
                        adj[b].remove(a)
                        adj[c].add(d)
                        adj[d].add(c)
                        degrees = _compute_degrees(adj, n)
    
    # Final edge list
    edges = [[min(u, v), max(u, v)] for u in range(n) for v in adj[u] if u < v]
    
    return {'n': n, 'edges': edges}

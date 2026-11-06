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

    def calculate_max_tri_sum(adj, triangles, degrees):
        if not triangles:
            return 0
        max_sum = 0
        for u, v, w in triangles:
            s = degrees[u] + degrees[v] + degrees[w]
            if s > max_sum:
                max_sum = s
        return max_sum

    def check_swap(adj, degrees, triangles, remove_u, remove_v, add_u, add_v):
        """
        Check if swapping edge (remove_u, remove_v) with (add_u, add_v)
        results in a strictly lower max triangle degree sum.
        Returns (new_max_sum, is_better)
        """
        # 1. Update degrees
        new_degrees = degrees[:]
        new_degrees[remove_u] -= 1
        new_degrees[remove_v] -= 1
        new_degrees[add_u] += 1
        new_degrees[add_v] += 1
        
        # 2. Identify affected triangles
        # Triangles that are destroyed: those containing (remove_u, remove_v)
        # Triangles that are modified: those containing add_u or add_v (and not destroyed)
        # Triangles that are created: those containing (add_u, add_v)
        
        # We need to find the max sum in the new graph.
        # To be safe and simple for n=16, we can just recalculate all triangle sums
        # using the new degrees, but we must handle the topology change.
        
        # Topology change:
        # Remove edge (remove_u, remove_v)
        # Add edge (add_u, add_v)
        
        # Let's construct a list of valid triangles in the new graph.
        # This is expensive if done from scratch.
        # Optimization: 
        # The set of triangles changes only locally.
        # However, n=16 is small. Let's just rebuild the triangle list?
        # 5000 iterations * 100 candidates * (rebuild triangles) might be too slow.
        
        # Better: 
        # We know the old triangles.
        # 1. Mark triangles containing (remove_u, remove_v) as INVALID.
        # 2. For all other old triangles, their structure is unchanged, only degrees changed.
        #    We can just recalculate their sums with new_degrees.
        # 3. Find NEW triangles created by adding (add_u, add_v).
        #    A new triangle is (add_u, add_v, x) where x is a common neighbor of add_u and add_v 
        #    in the NEW graph.
        #    Note: x must be adjacent to add_u and add_v.
        #    Since we only added (add_u, add_v), the common neighbors are 
        #    (adj[add_u] & adj[add_v]) in the NEW graph.
        #    But adj sets in `adj` are old.
        #    New neighbors of add_u: adj[add_u] U {add_v} - {remove_v if remove_v==add_u? no}
        #    Actually, we are swapping. 
        #    Let's just update the adj sets temporarily or calculate on the fly.
        
        # Given the small size, a semi-brute force check of new triangles is fine.
        # But we can optimize by checking only candidates that might be high.
        
        # Let's try a simpler heuristic first: 
        # If the swap reduces the degree of the highest degree vertex in the worst triangle, 
        # and doesn't create a triangle with a vertex of very high degree, it's likely good.
        
        # For correctness and speed balance:
        # We will calculate the max sum for:
        # A) All old triangles that do NOT contain (remove_u, remove_v)
        # B) All new triangles formed by (add_u, add_v, x)
        
        # A) Old triangles (excluding destroyed ones)
        max_sum_A = 0
        for t in triangles:
            # Check if this triangle is destroyed
            # Triangle (u,v,w) is destroyed if it contains edge (remove_u, remove_v)
            # i.e., {remove_u, remove_v} is a subset of {t[0], t[1], t[2]}
            if (remove_u in t and remove_v in t):
                continue
            s = new_degrees[t[0]] + new_degrees[t[1]] + new_degrees[t[2]]
            if s > max_sum_A:
                max_sum_A = s
                
        # B) New triangles
        # Common neighbors of add_u and add_v in the new graph
        # New adj[add_u] = adj[add_u] U {add_v} (and maybe others if we removed an edge incident to add_u? 
        # No, we only removed (remove_u, remove_v). 
        # So adj[add_u] changes only if add_u == remove_u or add_u == remove_v.
        # Similarly for add_v.
        
        # Let's compute the common neighbors properly.
        # common = (adj[add_u] | {add_v}) & (adj[add_v] | {add_u})
        # But we must exclude edges that don't exist or were removed.
        # Actually, the edge (add_u, add_v) is now present.
        # The neighbors of add_u in the new graph are:
        # old neighbors, plus add_v.
        # UNLESS add_v was a neighbor and we removed it? No, we are adding it.
        # UNLESS add_u or add_v is remove_u or remove_v, in which case we might have removed an edge.
        
        # Let's just build the sets for the two vertices involved in the new edge.
        new_adj_u = set(adj[add_u])
        new_adj_v = set(adj[add_v])
        
        # Account for removal of (remove_u, remove_v)
        if remove_u == add_u and remove_v in new_adj_u:
            new_adj_u.remove(remove_v)
        elif remove_v == add_u and remove_u in new_adj_u:
            new_adj_u.remove(remove_u)
            
        if remove_u == add_v and remove_v in new_adj_v:
            new_adj_v.remove(remove_v)
        elif remove_v == add_v and remove_u in new_adj_v:
            new_adj_v.remove(remove_u)
            
        # Account for addition of (add_u, add_v)
        new_adj_u.add(add_v)
        new_adj_v.add(add_u)
        
        # Common neighbors x such that x != add_u and x != add_v
        common = new_adj_u & new_adj_v
        common.discard(add_u)
        common.discard(add_v)
        
        max_sum_B = 0
        for x in common:
            # Triangle (add_u, add_v, x)
            s = new_degrees[add_u] + new_degrees[add_v] + new_degrees[x]
            if s > max_sum_B:
                max_sum_B = s
                
        new_max = max(max_sum_A, max_sum_B)
        return new_max

    degrees = get_degrees(adj)
    triangles = get_triangles(adj)
    current_max = calculate_max_tri_sum(adj, triangles, degrees)
    
    # Main optimization loop
    no_improvement_count = 0
    max_iterations = 2000
    
    for iteration in range(max_iterations):
        if no_improvement_count > 50:
            # Random restart or perturbation
            # Try a random swap that doesn't increase max
            edge_list = [(u, v) for u in range(n) for v in adj[u] if u < v]
            non_edges = [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]
            
            if not edge_list or not non_edges:
                break
                
            # Try a few random swaps
            found_perturb = False
            for _ in range(10):
                ru, rv = random.choice(edge_list)
                au, av = random.choice(non_edges)
                # Avoid removing the edge we're adding (not possible)
                # Check if it's a valid swap (distinct edges)
                if (ru, rv) == (au, av) or (ru, rv) == (av, au):
                    continue
                
                new_max = check_swap(adj, degrees, triangles, ru, rv, au, av)
                if new_max <= current_max:
                    # Apply swap
                    adj[ru].remove(rv)
                    adj[rv].remove(ru)
                    adj[au].add(av)
                    adj[av].add(au)
                    
                    degrees = get_degrees(adj)
                    triangles = get_triangles(adj)
                    current_max = new_max
                    no_improvement_count = 0
                    found_perturb = True
                    break
            
            if not found_perturb:
                # If we can't even maintain the score, we might be stuck or done.
                # For this problem, we should always be able to maintain or improve.
                # If we can't, break.
                break
            continue

        # Find worst triangles
        worst_tris = []
        for t in triangles:
            s = degrees[t[0]] + degrees[t[1]] + degrees[t[2]]
            if s == current_max:
                worst_tris.append(t)
        
        if not worst_tris:
            break
            
        improved = False
        
        # Try to improve by swapping edges in worst triangles
        # Sort worst triangles by the degree of their highest vertex (desc) to target the biggest offenders
        worst_tris.sort(key=lambda t: -max(degrees[t[0]], degrees[t[1]], degrees[t[2]]))
        
        for tri in worst_tris:
            u, v, w = tri
            # Edges in this triangle: (u,v), (u,w), (v,w)
            edges_to_try = [(u, v), (u, w), (v, w)]
            
            # Sort edges by the degree sum of the vertices involved? 
            # We want to remove an edge connected to the high degree vertex.
            # Let's just try all 3.
            
            for (ru, rv) in edges_to_try:
                # Find candidate non-edges to add
                # Prioritize non-edges between low degree vertices
                # To avoid creating high sum triangles, we want x, y with low degrees
                # and not adjacent to high degree vertices if possible.
                
                candidates = []
                for x in range(n):
                    for y in range(x + 1, n):
                        if y not in adj[x]:
                            # Skip if this edge is the one we're removing (shouldn't happen as it's an edge)
                            if (x == ru and y == rv) or (x == rv and y == ru):
                                continue
                            # Heuristic: prefer low degree sum
                            deg_sum = degrees[x] + degrees[y]
                            candidates.append((deg_sum, x, y))
                
                # Sort by degree sum (lower is better)
                candidates.sort()
                
                # Try top candidates
                # Limit to top 50 for speed
                for deg_sum, au, av in candidates[:50]:
                    new_max = check_swap(adj, degrees, triangles, ru, rv, au, av)
                    
                    if new_max < current_max:
                        # Apply swap
                        adj[ru].remove(rv)
                        adj[rv].remove(ru)
                        adj[au].add(av)
                        adj[av].add(au)
                        
                        degrees = get_degrees(adj)
                        triangles = get_triangles(adj)
                        current_max = new_max
                        improved = True
                        no_improvement_count = 0
                        break
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            no_improvement_count += 1
            
    # Final output
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    
    return {'n': n, 'edges': edges}

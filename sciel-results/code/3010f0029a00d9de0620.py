import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 18
    
    # Helper: Check if graph is K4-free
    def is_k4_free(adj):
        # adj is a list of sets or bitmasks. Let's use bitmasks for speed.
        # adj[i] is a bitmask of neighbors of i.
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    # Check if any common neighbor k forms a K4 with i, j, k
                    # Neighbors of i and j
                    common = adj[i] & adj[j]
                    # We need to check if there is a k in common such that 
                    # i, j, k form a triangle? No, K4 requires 4 vertices.
                    # K4: i, j, k, l.
                    # If i, j are connected, and k is a common neighbor, 
                    # we need to check if k and l are connected?
                    # Actually, simpler: iterate over all 4-cliques? No, too slow.
                    # Check if there exists k such that i,j,k form a triangle, 
                    # and then if there exists l connected to i,j,k.
                    # Or just: for each pair (i,j), if connected, 
                    # let C = N(i) & N(j).
                    # For each k in C, if i,j,k is a triangle (always true if k in C and i-j connected? 
                    # No, i-j connected, k connected to i and j. Yes, i,j,k is a triangle).
                    # Now, does this triangle extend to a K4?
                    # We need a vertex l connected to i, j, and k.
                    # l must be in N(i) & N(j) & N(k).
                    # So if (N(i) & N(j) & N(k)) is non-empty, we have a K4.
                    for k in range(n):
                        if k == i or k == j: continue
                        if (adj[i] & (1 << k)) and (adj[j] & (1 << k)):
                            # i, j, k is a triangle
                            common_ijk = adj[i] & adj[j] & adj[k]
                            if common_ijk:
                                return False
        return True

    # Helper: Max triangle-free induced subgraph size
    # This is equivalent to Min Vertex Cover of the hypergraph of triangles.
    # Score = n - min_cover_size.
    def max_triangle_free_size(adj):
        # Find all triangles
        triangles = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    # Iterate through common neighbors
                    k_mask = common
                    while k_mask:
                        lsb = k_mask & -k_mask
                        k = lsb.bit_length() - 1
                        triangles.append((i, j, k))
                        k_mask ^= lsb
        
        if not triangles:
            return n
            
        # B&B for Hitting Set
        best_cover = n # Max possible
        
        # Sort triangles by size (all 3) but maybe by frequency? 
        # For B&B, order matters.
        
        def solve(triangles, covered_mask, current_size):
            nonlocal best_cover
            if current_size >= best_cover:
                return
            if not triangles:
                best_cover = current_size
                return
                
            # Pruning: Greedy lower bound
            # Find max set of disjoint triangles
            # This is expensive. Simple bound: 1 if any triangle left.
            # Better: Count disjoint triangles greedily.
            avail = []
            for t in triangles:
                # Check if t is not covered by current_cover? 
                # No, 'covered_mask' is the set of vertices we have picked.
                # A triangle is "hit" if any of its vertices is in covered_mask.
                # We only need to consider triangles NOT hit yet.
                if not (covered_mask & (1 << t[0])) and not (covered_mask & (1 << t[1])) and not (covered_mask & (1 << t[2])):
                    avail.append(t)
            
            if not avail:
                best_cover = min(best_cover, current_size)
                return
            
            # Greedy packing for lower bound
            # Try to pack as many disjoint triangles as possible from avail
            # This is NP-hard too, but we can do a quick greedy.
            # Actually, just using 1 is weak.
            # Let's use a simple heuristic: 
            # If we have k triangles, and they are all disjoint, we need k.
            # Let's just branch.
            
            # Pick a triangle to branch on
            # Heuristic: Pick the triangle with the most "constrained" vertices?
            # Or just the first one.
            t = avail[0]
            
            # Branch on each vertex in t
            for v in t:
                new_mask = covered_mask | (1 << v)
                # Recurse with triangles that are still not hit
                # We can filter triangles list for efficiency, but passing list is ok for n=18
                # Optimization: Pass index or filtered list
                solve([x for x in avail if not (x[0]==v or x[1]==v or x[2]==v)], new_mask, current_size + 1)
                
        solve(triangles, 0, 0)
        return n - best_cover

    # Initialize with a random K4-free graph
    # Start with K6,6,6 and randomize? Or random graph and remove K4s?
    # Random graph with p=0.5, then remove edges to make K4-free.
    
    def generate_random_k4_free():
        adj = [0] * n
        # Random edges
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Remove edges to ensure K4-free
        # Iterate through all K4s? Too many.
        # Instead, check K4s formed by new edges?
        # Simpler: Start with empty graph, add random edges if they don't form K4.
        adj = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        for u, v in edges:
            # Check if adding (u, v) creates a K4
            # A K4 containing (u, v) must have two other vertices x, y
            # such that u, v, x, y are all connected.
            # x, y must be in N(u) & N(v).
            # And x, y must be connected.
            common = adj[u] & adj[v]
            # Check if there is an edge in common
            # Iterate through pairs in common
            c_mask = common
            has_k4 = False
            # Efficient check: if there is any edge in the subgraph induced by common
            # We can just iterate through vertices in common
            verts = []
            while c_mask:
                lsb = c_mask & -c_mask
                idx = lsb.bit_length() - 1
                verts.append(idx)
                c_mask ^= lsb
            
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    if adj[verts[i]] & (1 << verts[j]):
                        has_k4 = True
                        break
                if has_k4: break
            
            if not has_k4:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    best_adj = generate_random_k4_free()
    best_score = max_triangle_free_size(best_adj)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate neighbor
        adj_copy = best_adj[:]
        u, v = random.randint(0, n-1), random.randint(0, n-1)
        if u == v: continue
        if u > v: u, v = v, u
        
        # Try to flip edge (u, v)
        if adj_copy[u] & (1 << v):
            # Remove edge
            adj_copy[u] ^= (1 << v)
            adj_copy[v] ^= (1 << u)
            # Removing edge cannot create K4, so it's always valid
            new_adj = adj_copy
        else:
            # Add edge
            # Check if it creates K4
            common = adj_copy[u] & adj_copy[v]
            c_mask = common
            has_k4 = False
            verts = []
            while c_mask:
                lsb = c_mask & -c_mask
                idx = lsb.bit_length() - 1
                verts.append(idx)
                c_mask ^= lsb
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    if adj_copy[verts[i]] & (1 << verts[j]):
                        has_k4 = True
                        break
                if has_k4: break
            
            if has_k4:
                continue # Invalid move
            
            adj_copy[u] |= (1 << v)
            adj_copy[v] |= (1 << u)
            new_adj = adj_copy
        
        new_score = max_triangle_free_size(new_adj)
        
        # We want to MINIMIZE new_score
        delta = new_score - best_score
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            best_adj = new_adj
            best_score = new_score
        
        T *= alpha
        
        # Occasional restart? No, SA should handle it.
        
    # Return edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    return edges

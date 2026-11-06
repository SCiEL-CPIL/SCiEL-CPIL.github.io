import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 18
    
    # Helper: Check if adding edge (u, v) creates a K4
    def creates_k4(adj, u, v):
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        # In a K4-free graph, common neighbors must form an independent set.
        # So if there is any edge between two common neighbors, it's a K4.
        c_mask = common
        # We need to check if the subgraph induced by 'common' has any edges.
        # Since n=18, common is small.
        verts = []
        while c_mask:
            lsb = c_mask & -c_mask
            idx = lsb.bit_length() - 1
            verts.append(idx)
            c_mask ^= lsb
        
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                if adj[verts[i]] & (1 << verts[j]):
                    return True
        return False

    # Helper: Get all triangles as a list of (u, v, w) or bitmasks
    def get_triangles(adj):
        triangles = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    k_mask = common
                    while k_mask:
                        lsb = k_mask & -k_mask
                        k = lsb.bit_length() - 1
                        triangles.append((i, j, k))
                        k_mask ^= lsb
        return triangles

    # Helper: Greedy Vertex Cover Size of the triangle hypergraph
    # Returns the size of the cover found by the greedy algorithm.
    # We want to MAXIMIZE this size (to minimize the triangle-free set size).
    def greedy_vertex_cover_size(adj):
        triangles = get_triangles(adj)
        if not triangles:
            return 0
            
        # Count frequency of each vertex in triangles
        # We will iteratively pick the vertex with highest frequency
        # and remove all triangles containing it.
        
        # To speed up, we can use a set of active triangles.
        # But for n=18, the number of triangles is at most 816.
        # Simple list filtering is probably fine.
        
        active_triangles = triangles
        cover_size = 0
        
        while active_triangles:
            # Count occurrences
            counts = [0] * n
            for t in active_triangles:
                counts[t[0]] += 1
                counts[t[1]] += 1
                counts[t[2]] += 1
            
            # Find max
            max_count = -1
            best_v = -1
            for v in range(n):
                if counts[v] > max_count:
                    max_count = counts[v]
                    best_v = v
            
            if best_v == -1:
                break
                
            cover_size += 1
            
            # Filter triangles
            new_active = []
            for t in active_triangles:
                if best_v not in t:
                    new_active.append(t)
            active_triangles = new_active
            
        return cover_size

    # Helper: Exact Max Triangle-Free Size (Min Vertex Cover)
    # Uses B&B.
    def exact_max_triangle_free_size(adj):
        triangles = get_triangles(adj)
        if not triangles:
            return n
            
        best_cover = n
        
        # Precompute triangle bitmasks for faster intersection?
        # t_mask = (1<<i)|(1<<j)|(1<<k)
        t_masks = [ (1<<t[0]) | (1<<t[1]) | (1<<t[2]) for t in triangles ]
        
        def solve(remaining_indices, covered_mask, current_size):
            nonlocal best_cover
            if current_size >= best_cover:
                return
            if not remaining_indices:
                best_cover = current_size
                return
                
            # Lower bound: we need at least 1 more if remaining is not empty
            # Better bound: Max number of disjoint triangles in remaining?
            # For now, just branch.
            
            # Pick a triangle to branch on.
            # Heuristic: Pick one with vertices having high degree in remaining hypergraph?
            # Just pick the first one.
            idx = remaining_indices[0]
            t = triangles[idx]
            t_mask = t_masks[idx]
            
            # Branch on each vertex in t
            for v in t:
                new_mask = covered_mask | (1 << v)
                # Filter remaining triangles that are not hit by v
                new_remaining = []
                for r_idx in remaining_indices:
                    if r_idx == idx: continue
                    # Check if v is in triangle r_idx
                    if not (t_masks[r_idx] & (1 << v)):
                        new_remaining.append(r_idx)
                        
                solve(new_remaining, new_mask, current_size + 1)
                
        # Initial call
        # We can sort triangles to help B&B?
        # Sort by "density" or just random?
        # Let's just use 0..len-1
        solve(list(range(len(triangles))), 0, 0)
        return n - best_cover

    # Initialize with a random K4-free graph
    def generate_random_k4_free():
        adj = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if not creates_k4(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    best_adj = generate_random_k4_free()
    best_greedy_score = n - greedy_vertex_cover_size(best_adj)
    best_exact_score = exact_max_triangle_free_size(best_adj)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.9995
    
    step = 0
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
            new_adj = adj_copy
        else:
            # Add edge
            if creates_k4(adj_copy, u, v):
                continue
            adj_copy[u] |= (1 << v)
            adj_copy[v] |= (1 << u)
            new_adj = adj_copy
        
        # Evaluate with greedy score
        new_greedy_cover = greedy_vertex_cover_size(new_adj)
        new_greedy_score = n - new_greedy_cover
        
        # We want to MINIMIZE the score (maximize cover)
        # So we accept if new_greedy_score < best_greedy_score (improvement)
        # Or with probability exp(-delta/T) where delta = new_score - best_score
        # Note: lower score is better.
        # delta = new_greedy_score - best_greedy_score
        # If delta < 0, it's an improvement.
        
        delta = new_greedy_score - best_greedy_score
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            best_adj = new_adj
            best_greedy_score = new_greedy_score
            
            # Check exact score periodically or if greedy score is very good
            # If greedy score is lower than current best exact score, it might be better
            if best_greedy_score < best_exact_score:
                exact_score = exact_max_triangle_free_size(best_adj)
                if exact_score < best_exact_score:
                    best_exact_score = exact_score
                # If exact score is much higher, update best_greedy_score to reflect reality?
                # No, keep greedy for search direction, but track best exact for return.
                # Actually, if greedy says 10 but exact is 12, we might be misled.
                # But greedy is a lower bound on the cover size? No.
                # Greedy cover size <= Optimal cover size.
                # So Greedy Score (n - greedy_cover) >= Exact Score (n - opt_cover).
                # We are minimizing the score.
                # So Greedy Score is an UPPER BOUND on the true score.
                # If Greedy Score < Best Exact Score, the true score might be even lower.
                # So it's worth checking.
                
            step += 1
            if step % 1000 == 0:
                # Occasionally re-evaluate best exact to ensure we don't miss a global optimum
                # if the greedy metric diverged
                pass

        T *= alpha
        
    # Final exact check on best_adj
    final_score = exact_max_triangle_free_size(best_adj)
    
    # Return edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    return edges

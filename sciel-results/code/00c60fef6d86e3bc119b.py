import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 18
    vertices = list(range(n))
    
    # Helper to check if graph is K4-free using bitmask adjacency
    # adj is a list of n integers, where adj[u] has bit v set if (u,v) is an edge
    def is_k4_free(adj):
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] & (1 << b)):
                    continue
                for c in range(b + 1, n):
                    if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)):
                        continue
                    # a-b-c is a triangle
                    # Check if there is a d > c connected to a, b, c
                    common = adj[a] & adj[b] & adj[c]
                    # We need d > c
                    # Mask for bits > c
                    mask_d = ((1 << n) - 1) ^ ((1 << (c + 1)) - 1)
                    if common & mask_d:
                        return False
        return True

    # Helper to calculate score: largest triangle-free induced subgraph
    # adj is a list of n integers
    def calc_score(adj):
        # Branch and Bound for max triangle-free set
        best_size = 0
        
        # Precompute neighbors
        adj_list = [0] * n
        for u in range(n):
            adj_list[u] = adj[u]
            
        def has_triangle(mask):
            # Check if induced subgraph on vertices in mask has a triangle
            # Iterate through vertices in mask
            # Optimization: if popcount < 3, return False
            if mask.bit_count() < 3:
                return False
            
            # Iterate through edges in the induced subgraph
            # For each edge (u, v) in mask, check if N(u) & N(v) & mask has any vertex
            # We can iterate u in mask
            m = mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                m ^= lsb
                
                # Neighbors of u within mask
                nbrs_u = adj_list[u] & mask
                # We only need to consider v > u to avoid double counting?
                # Actually, for triangle detection, any edge (u,v) with u!=v works.
                # But we can restrict v > u for efficiency?
                # Let's just iterate v in nbrs_u
                # If we find any common neighbor, it's a triangle.
                # Wait, if (u,v) is an edge, and w is common neighbor, (u,v,w) is triangle.
                # So we just need to check if (adj[u] & adj[v] & mask) != 0 for any edge (u,v) in mask.
                
                # To avoid checking (u,v) and (v,u), let's just check all v in nbrs_u.
                # It's redundant but simple.
                # Better: only check v > u?
                # Let's stick to checking all v in nbrs_u.
                # Actually, if u is in mask, and v is in nbrs_u, then v is in mask.
                # So we check adj[u] & adj[v] & mask.
                # If non-zero, triangle found.
                
                # Optimization:
                # Iterate v in nbrs_u
                # We can break early if found.
                
                # Let's iterate v in nbrs_u
                # But we need to be careful not to count the same triangle multiple times?
                # Doesn't matter for boolean check.
                
                # However, iterating all v in nbrs_u for each u is O(E).
                # Is there a faster way?
                # For small n=18, O(E) is fine.
                
                # Let's try to optimize by only checking v > u?
                # If we only check v > u, we cover all edges exactly once.
                # nbrs_u_gt_u = nbrs_u & (((1 << n) - 1) ^ ((1 << (u + 1)) - 1))
                # Then iterate v in nbrs_u_gt_u.
                
                nbrs_u_gt_u = nbrs_u & (((1 << n) - 1) ^ ((1 << (u + 1)) - 1))
                m2 = nbrs_u_gt_u
                while m2:
                    lsb2 = m2 & -m2
                    v = lsb2.bit_length() - 1
                    m2 ^= lsb2
                    
                    # Check common neighbor
                    if adj_list[u] & adj_list[v] & mask:
                        return True
                        
            return False
            
        def search(mask):
            nonlocal best_size
            if mask.bit_count() <= best_size:
                return
            
            if not has_triangle(mask):
                best_size = mask.bit_count()
                return
            
            # Find a triangle to branch on
            # We need to find one specific triangle (u,v,w) in mask
            # We can reuse the logic from has_triangle to find one
            
            # Iterate u in mask
            m = mask
            found_tri = None
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                m ^= lsb
                
                nbrs_u_gt_u = (adj_list[u] & mask) & (((1 << n) - 1) ^ ((1 << (u + 1)) - 1))
                m2 = nbrs_u_gt_u
                while m2:
                    lsb2 = m2 & -m2
                    v = lsb2.bit_length() - 1
                    m2 ^= lsb2
                    
                    common = adj_list[u] & adj_list[v] & mask
                    if common:
                        # Find a vertex w in common
                        # w can be any, let's pick the lowest
                        lsb3 = common & -common
                        w = lsb3.bit_length() - 1
                        found_tri = (u, v, w)
                        break
                if found_tri:
                    break
            if found_tri:
                u, v, w = found_tri
                # Branch: remove u, v, or w
                # Remove u
                search(mask & ~(1 << u))
                # Remove v
                search(mask & ~(1 << v))
                # Remove w
                search(mask & ~(1 << w))
            else:
                # Should not happen if has_triangle is True
                best_size = max(best_size, mask.bit_count())
                
        # Start with all vertices
        full_mask = (1 << n) - 1
        search(full_mask)
        return best_size

    # Convert adjacency list (bitmask) to edge list
    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    # Initialize with Turan Graph T(18,3) - K6,6,6
    # Parts: 0-5, 6-11, 12-17
    initial_adj = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    initial_adj[a] |= (1 << b)
                    initial_adj[b] |= (1 << a)
                    
    best_adj = initial_adj[:]
    best_score = calc_score(best_adj)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    
    all_possible_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_possible_edges.append((u, v))
            
    current_adj = best_adj[:]
    current_score = best_score
    
    # Precompute for K4 check on edge addition?
    # When adding (u,v), we need to check if there exist w,x such that
    # (u,w), (u,x), (v,w), (v,x), (w,x) are all present.
    # This is equivalent to checking if there is a K4 containing (u,v).
    # We can precompute for each edge (u,v) the set of pairs (w,x) that would form a K4?
    # Or just check on the fly.
    # On the fly:
    # Common neighbors of u and v: C = N(u) & N(v)
    # We need to find w, x in C such that (w,x) is an edge.
    # So we need to check if the induced subgraph on C contains an edge.
    # This is fast: iterate w in C, check if N(w) & C has any bit set.
    
    while time.time() < deadline:
        # Pick a random edge to toggle
        u, v = random.choice(all_possible_edges)
        
        # Check if edge is currently present
        present = bool(current_adj[u] & (1 << v))
        
        if present:
            # Remove edge
            new_adj = current_adj[:]
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            
            # Always K4-free
            new_score = calc_score(new_adj)
            
            if new_score < current_score:
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
            else:
                # Accept with probability
                delta = new_score - current_score
                if delta > 0:
                    if random.random() < pow(2.718, -delta / T):
                        current_adj = new_adj
                        current_score = new_score
                elif random.random() < 0.5:
                    current_adj = new_adj
                    current_score = new_score
        else:
            # Add edge
            # Check if adding (u,v) creates a K4
            # Common neighbors C = N(u) & N(v)
            C = current_adj[u] & current_adj[v]
            
            creates_k4 = False
            # Check if there is an edge within C
            # Iterate w in C
            m = C
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                m ^= lsb
                # Check if N(w) & C has any bit set (excluding w itself, but w is not in N(w))
                # Actually, we just need to check if there is x in C, x != w, such that (w,x) is edge.
                # N(w) & C will contain x if (w,x) is edge and x in C.
                # Since w is not in N(w), we don't need to mask out w.
                if current_adj[w] & C:
                    creates_k4 = True
                    break
            
            if creates_k4:
                # Skip move
                continue
                
            # Add edge
            new_adj = current_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
            new_score = calc_score(new_adj)
            
            if new_score < current_score:
                current_adj = new_adj
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
            else:
                delta = new_score - current_score
                if delta > 0:
                    if random.random() < pow(2.718, -delta / T):
                        current_adj = new_adj
                        current_score = new_score
                elif random.random() < 0.5:
                    current_adj = new_adj
                    current_score = new_score
                    
        T *= cooling
        
    return adj_to_edges(best_adj)

# EVOLVE-BLOCK-END

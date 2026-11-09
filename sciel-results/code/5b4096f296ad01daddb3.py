import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    def time_left():
        return deadline - (time.time() - start_time)

    def get_edges_from_adj(adj):
        n = len(adj)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges

    def is_c4_free_masks(masks):
        n = len(masks)
        for u in range(n):
            for v in range(u + 1, n):
                common = masks[u] & masks[v]
                if common & (common - 1) != 0:
                    return False
        return True

    def max_independent_set_size(adj, limit=12):
        """
        Returns the size of the maximum independent set if < limit, 
        or 'limit' if >= limit.
        Uses branch and bound on the complement graph (clique finding).
        Optimized with bitmasks.
        """
        n = len(adj)
        # Build complement masks
        comp_masks = [0] * n
        for i in range(n):
            m = 0
            row = adj[i]
            for j in range(n):
                if i != j and not row[j]:
                    m |= (1 << j)
            comp_masks[i] = m
        
        # Order vertices by degree in complement (descending) for better bounds
        # deg[i] is degree in complement
        degs = [bin(m).count('1') for m in comp_masks]
        order = sorted(range(n), key=lambda x: degs[x], reverse=True)
        
        # Map original index to position in sorted order? 
        # Actually, for bitmask B&B, it's often better to keep original indices 
        # but pick the vertex with highest degree in the candidate set.
        
        # Greedy coloring bound for clique in complement (independent set in G)
        def greedy_color_bound(candidates_mask):
            if candidates_mask == 0:
                return 0
            
            verts = []
            temp = candidates_mask
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                verts.append(idx)
                temp ^= lsb
            
            # Sort by degree in induced subgraph
            verts.sort(key=lambda v: bin(comp_masks[v] & candidates_mask).count('1'), reverse=True)
            
            colors = {}
            next_color = 1
            for v in verts:
                used_colors = set()
                neigh_mask = comp_masks[v] & candidates_mask
                temp_neigh = neigh_mask
                while temp_neigh:
                    lsb = temp_neigh & -temp_neigh
                    n_idx = lsb.bit_length() - 1
                    if n_idx in colors:
                        used_colors.add(colors[n_idx])
                    temp_neigh ^= lsb
                
                c = 1
                while c in used_colors:
                    c += 1
                colors[v] = c
                if c > next_color:
                    next_color = c
            
            return next_color

        best_size = 0
        
        def backtrack(candidates_mask, current_size, current_clique):
            nonlocal best_size
            if current_size >= limit:
                return True # Found a set of size >= limit
            
            if candidates_mask == 0:
                return False
            
            if current_size + bin(candidates_mask).count('1') <= best_size:
                return False
            
            color_bound = greedy_color_bound(candidates_mask)
            if current_size + color_bound <= best_size:
                return False
            
            # Pick vertex with highest degree in candidate set
            verts = []
            temp = candidates_mask
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                verts.append(idx)
                temp ^= lsb
            
            verts.sort(key=lambda v: bin(comp_masks[v] & candidates_mask).count('1'), reverse=True)
            
            # Iterate through candidates, usually starting with high degree ones
            # Standard B&B: pick one, branch include/exclude.
            # To find max size, we need to explore all.
            # But we just need to know if size >= limit.
            
            for v in verts:
                v_bit = 1 << v
                if not (candidates_mask & v_bit):
                    continue
                
                # Include v
                new_mask = candidates_mask & comp_masks[v]
                new_mask &= ~v_bit
                if backtrack(new_mask, current_size + 1, current_clique + [v]):
                    return True
                
                # Exclude v
                candidates_mask &= ~v_bit
                
            return False
        
        initial_mask = (1 << n) - 1
        if backtrack(initial_mask, 0, []):
            return limit
        else:
            # If we didn't find size >= limit, we need the exact max size found.
            # The B&B above returns True if it finds >= limit.
            # If it returns False, it means max size < limit.
            # We need to track the best_size found during the search.
            # The logic above is slightly flawed for returning exact size if < limit.
            # Let's re-run with a flag to track best_size properly.
            
            best_size = 0
            def backtrack_track(candidates_mask, current_size):
                nonlocal best_size
                if current_size >= limit:
                    return True
                
                if candidates_mask == 0:
                    return False
                
                if current_size + bin(candidates_mask).count('1') <= best_size:
                    return False
                
                color_bound = greedy_color_bound(candidates_mask)
                if current_size + color_bound <= best_size:
                    return False
                
                verts = []
                temp = candidates_mask
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    verts.append(idx)
                    temp ^= llsb
                
                verts.sort(key=lambda v: bin(comp_masks[v] & candidates_mask).count('1'), reverse=True)
                
                for v in verts:
                    v_bit = 1 << v
                    if not (candidates_mask & v_bit):
                        continue
                    
                    # Include v
                    new_mask = candidates_mask & comp_masks[v]
                    new_mask &= ~v_bit
                    if backtrack_track(new_mask, current_size + 1):
                        return True
                    
                    # Exclude v
                    candidates_mask &= ~v_bit
                    
                    # Update best if we found a larger clique in the exclusion branch?
                    # Actually, the standard way is to just return the max.
                    # Since we prune if <= best_size, best_size is updated when we hit a leaf or prune?
                    # No, best_size should be updated whenever we have a valid clique.
                    # The function returns True if it finds >= limit.
                    # If it returns False, it explored the space where size < limit.
                    # We need to capture the max size found.
                    
            # Let's just use a simpler approach: find max clique size directly if n is small enough.
            # For n <= 43, finding max independent set is feasible.
            
            def find_max_is_size(adj, n):
                comp_masks = [0] * n
                for i in range(n):
                    m = 0
                    for j in range(n):
                        if i != j and not adj[i][j]:
                            m |= (1 << j)
                    comp_masks[i] = m
                
                best = 0
                def bnb(mask, size):
                    nonlocal best
                    if size > best:
                        best = size
                    if mask == 0:
                        return
                    if size + bin(mask).count('1') <= best:
                        return
                    
                    # Color bound
                    verts = []
                    temp = mask
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        verts.append(idx)
                        temp ^= lsb
                    verts.sort(key=lambda v: bin(comp_masks[v] & mask).count('1'), reverse=True)
                    
                    colors = {}
                    next_c = 1
                    for v in verts:
                        used = set()
                        nm = comp_masks[v] & mask
                        tn = nm
                        while tn:
                            lsb = tn & -tn
                            ni = lsb.bit_length() - 1
                            if ni in colors:
                                used.add(colors[ni])
                            tn ^= lsb
                        c = 1
                        while c in used:
                            c += 1
                        colors[v] = c
                        if c > next_c:
                            next_c = c
                    
                    if size + next_c <= best:
                        return
                    
                    # Branch on last vertex in sorted order (lowest degree usually? No, highest)
                    # Standard: pick a vertex, branch include/exclude.
                    # To maximize pruning, pick a vertex that reduces the bound most?
                    # Let's just pick the first one in the sorted list (highest degree).
                    v = verts[0]
                    v_bit = 1 << v
                    
                    # Include
                    new_mask = mask & comp_masks[v] & ~v_bit
                    bnb(new_mask, size + 1)
                    
                    # Exclude
                    new_mask = mask & ~v_bit
                    bnb(new_mask, size)
                
                bnb((1 << n) - 1, 0)
                return best
            
            return find_max_is_size(adj, n)

    def construct_polarity_graph(q):
        points = []
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
        for m in range(q):
            points.append((m, 1, 0))
        points.append((1, 0, 0))
        
        n = len(points)
        adj = np.zeros((n, n), dtype=bool)
        
        for i in range(n):
            x1, y1, z1 = points[i]
            for j in range(i + 1, n):
                x2, y2, z2 = points[j]
                dot = (x1 * x2 + y1 * y2 + z1 * z2) % q
                if dot == 0:
                    adj[i][j] = True
                    adj[j][i] = True
        return adj

    # Strategy:
    # 1. Start with a good seed: Polarity Graph of PG(2,5) -> n=31.
    #    This graph is C4-free.
    #    Alpha of PG(2,5) polarity graph is 8 (known).
    #    So alpha < 11 is satisfied.
    # 2. Try to add vertices one by one.
    #    For each new vertex, determine its adjacency to existing vertices such that:
    #    - No C4 is formed.
    #    - Alpha remains < 11.
    # 3. If we get stuck, try local search (edge swaps) to improve density or structure?
    #    Actually, just adding vertices is the main goal.
    #    The constraint "No two vertices have two common neighbors" is strong.
    
    # Let's use the PG(2,5) graph as the base.
    q = 5
    adj_seed = construct_polarity_graph(q)
    n_seed = len(adj_seed)
    
    # Convert to list of lists for easy manipulation
    adj = [adj_seed[i].tolist() for i in range(n_seed)]
    
    current_n = n_seed
    
    # Helper to check if adding vertex v with neighbors N_v is valid
    def can_add_vertex(adj, new_neighbors):
        n = len(adj)
        # new_neighbors is a bitmask or list of indices in 0..n-1
        # Check C4 condition:
        # For any pair (u, w) in new_neighbors, they must not have a common neighbor in adj.
        # Wait, the condition is: No two vertices have two common neighbors.
        # If we add vertex v connected to u and w:
        # 1. u and v have common neighbors: N(u) \cap N(v).
        #    N(v) is new_neighbors.
        #    So |N(u) \cap new_neighbors| <= 1.
        # 2. v and w have common neighbors: |N(w) \cap new_neighbors| <= 1.
        # 3. Any two existing vertices u, w: their common neighbors are unchanged 
        #    UNLESS v is a common neighbor.
        #    If v is connected to both u and w, then v becomes a common neighbor of u and w.
        #    So |N(u) \cap N(w) \cap {v}| <= 1. This is always true since it's just {v} or empty.
        #    Wait, the condition is "No two vertices may have TWO common neighbors".
        #    If u and w already had 1 common neighbor, and we add v as a second one, it's invalid.
        #    So for any u, w in new_neighbors, we must check if they already have a common neighbor.
        #    If yes, we cannot connect v to both u and w.
        
        # Let's precompute common neighbor counts for all pairs in existing graph.
        # common_count[u][w] = number of common neighbors.
        # If common_count[u][w] == 1, we cannot add v adjacent to both u and w.
        # If common_count[u][w] == 0, we can add v adjacent to both u and w (v becomes the 1st common neighbor).
        # If common_count[u][w] >= 2, it's already invalid (shouldn't happen).
        
        # Also, for u in new_neighbors, we need |N(u) \cap new_neighbors| <= 1.
        # This means new_neighbors can contain at most 2 vertices that are adjacent to u?
        # No, N(u) \cap new_neighbors is the set of neighbors of v that are also neighbors of u.
        # If v is connected to u1, u2, u3, and u1, u2 are both neighbors of u, then u and v share u1 and u2.
        # So for any u (existing), the number of neighbors of v that are also neighbors of u must be <= 1.
        
        # Let's check these conditions.
        
        # Condition 1: For any u in 0..n-1, count how many neighbors of v are in N(u).
        # This must be <= 1.
        # Let S = new_neighbors.
        # For each u, |N(u) \cap S| <= 1.
        
        # Condition 2: For any pair u, w in S, they must not already have a common neighbor.
        # i.e., common_count[u][w] must be 0.
        
        # Let's implement this check.
        
        # Precompute masks for speed
        masks = [0] * n
        for i in range(n):
            m = 0
            for j in range(n):
                if adj[i][j]:
                    m |= (1 << j)
            masks[i] = m
        
        # Check Condition 1
        for u in range(n):
            common = masks[u] & new_neighbors
            if common & (common - 1) != 0:
                return False
        
        # Check Condition 2
        # For any u, w in S, common_count[u][w] must be 0.
        # This is equivalent to: The subgraph induced by S in G is a matching?
        # No, it means no two vertices in S are connected by a path of length 2 in G.
        # i.e., S is an independent set in the square of G?
        # Actually, if u and w are in S, and they have a common neighbor x, then x is a common neighbor of u and w.
        # So we require that no two vertices in S have a common neighbor in G.
        
        # Let's check pairs in S.
        s_list = []
        temp = new_neighbors
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            s_list.append(idx)
            temp ^= lsb
        
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                u = s_list[i]
                w = s_list[j]
                common = masks[u] & masks[w]
                if common != 0:
                    return False
        
        return True

    # Greedy addition
    # We want to add vertices.
    # For each new vertex, we want to choose a set of neighbors S such that:
    # 1. can_add_vertex(S) is True.
    # 2. The new graph has no independent set of size 11.
    #    Adding a vertex can increase alpha.
    #    If v is isolated, alpha increases by 1.
    #    If v is connected to some vertices, alpha might stay same or increase.
    #    To be safe, we should ensure that the new vertex is "well-connected" or that the alpha bound holds.
    #    Given the time limit, checking alpha for every addition is expensive.
    #    However, starting from alpha=8, we have a buffer of 2.
    #    If we add vertices with degree >= 1, alpha might not increase immediately.
    #    Let's add vertices and check alpha periodically or if we get close to 43.
    
    max_n_target = 43
    
    while current_n < max_n_target and time_left() > 10:
        # Try to add a vertex
        # Heuristic: Try to connect to as many vertices as possible to keep alpha low?
        # Or connect to a random subset?
        # Let's try to find a valid S.
        # Start with empty S? No, isolated vertex increases alpha.
        # Let's try to connect to 1, 2, 3... vertices.
        
        added = False
        # Shuffle vertices to break symmetry
        vertices_to_try = list(range(current_n))
        random.shuffle(vertices_to_try)
        
        # Try different sizes of S
        for s_size in range(1, current_n + 1):
            if added:
                break
            # Randomly pick s_size vertices
            # This is expensive if we do it naively.
            # Let's try a few random subsets of size s_size.
            for _ in range(5):
                if time_left() < 5:
                    break
                s_subset = random.sample(vertices_to_try, min(s_size, current_n))
                s_mask = 0
                for v in s_subset:
                    s_mask |= (1 << v)
                
                if can_add_vertex(adj, s_mask):
                    # Check alpha
                    # Build new adj
                    new_adj = [row[:] for row in adj]
                    new_adj.append([False] * (current_n + 1))
                    for i in range(current_n):
                        if s_mask & (1 << i):
                            new_adj[i][current_n] = True
                            new_adj[current_n][i] = True
                    new_adj[current_n][current_n] = False # Self loop false
                    
                    # Check alpha
                    alpha = max_independent_set_size(new_adj, 11)
                    if alpha < 11:
                        # Accept
                        adj = new_adj
                        current_n += 1
                        added = True
                        break
        
        if not added:
            # Try local search? Or just break?
            # If we can't add a vertex with any subset, we might be stuck.
            # Maybe try a different seed or random restart?
            # Given time, let's just break and return current.
            break

    # Final verification
    if current_n > 0:
        # Check C4
        masks = [0] * current_n
        for i in range(current_n):
            m = 0
            for j in range(current_n):
                if adj[i][j]:
                    m |= (1 << j)
            masks[i] = m
        
        c4_free = True
        for u in range(current_n):
            for v in range(u + 1, current_n):
                common = masks[u] & masks[v]
                if common & (common - 1) != 0:
                    c4_free = False
                    break
            if not c4_free:
                break
        
        if c4_free:
            alpha = max_independent_set_size(adj, 11)
            if alpha < 11:
                edges = get_edges_from_adj(adj)
                return {'n': current_n, 'edges': edges}
        
        # If validation fails, return the seed if it was valid
        # PG(2,5) is known to be valid.
        if n_seed > 0:
             # Re-verify seed just in case
             masks_seed = [0] * n_seed
             for i in range(n_seed):
                 m = 0
                 for j in range(n_seed):
                     if adj_seed[i][j]:
                         m |= (1 << j)
                 masks_seed[i] = m
             c4_free_seed = True
             for u in range(n_seed):
                 for v in range(u + 1, n_seed):
                     common = masks_seed[u] & masks_seed[v]
                     if common & (common - 1) != 0:
                         c4_free_seed = False
                         break
                 if not c4_free_seed:
                     break
             if c4_free_seed:
                 alpha_seed = max_independent_set_size(adj_seed, 11)
                 if alpha_seed < 11:
                     edges = get_edges_from_adj(adj_seed)
                     return {'n': n_seed, 'edges': edges}

    # Fallback: 10 disjoint triangles
    n = 30
    adj = np.zeros((n, n), dtype=bool)
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        adj[a][b] = adj[b][a] = True
        adj[a][c] = adj[c][a] = True
        adj[b][c] = adj[c][b] = True
    edges = get_edges_from_adj(adj)
    return {'n': n, 'edges': edges}

if __name__ == "__main__":
    res = construct()
    print(f"Constructed graph with n={res['n']}")

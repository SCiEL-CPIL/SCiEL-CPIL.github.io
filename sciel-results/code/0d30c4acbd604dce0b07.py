import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Helper to check if a coloring is valid (no mono K2,2,2)
    # and count violations for local search.
    
    def count_violations(n, colors):
        """
        Returns the number of monochromatic K2,2,2 subgraphs.
        colors is a flat list of 0/1 for pairs (i,j) i<j.
        """
        # Build adjacency bitmasks for each color
        # adj[c][i] is a bitmask of vertices j such that (i,j) is color c
        adj = [[0] * n for _ in range(2)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        total_violations = 0
        
        for c in [0, 1]:
            # We want to find triangles in the "pair graph" H_c
            # Nodes of H_c are pairs (u,v) with u<v.
            # Edge in H_c between (u,v) and (x,y) if all 4 edges between {u,v} and {x,y} are color c.
            # This is equivalent to: {x,y} is a subset of common neighbors of u and v in G_c.
            
            # To find triangles in H_c, we can iterate over all triples of pairs?
            # No, that's C(435, 3) which is ~14M. Too slow.
            
            # Instead, we can iterate over the middle pair?
            # Or just iterate over all pairs (u,v) and (x,y) that form a K2,2 in G_c.
            # Let's build the adjacency list of H_c.
            # H_adj[pair_idx] = set of pair_idx connected to it.
            
            # Mapping from (u,v) to index
            # We can just use (u,v) tuples or encode them.
            # Let's use a dictionary or list of sets.
            # Number of pairs is M = n(n-1)/2.
            
            # For each pair (u,v):
            #   Find common neighbors C = adj[c][u] & adj[c][v]
            #   For each pair (x,y) in C:
            #       Add edge between (u,v) and (x,y) in H_c.
            
            # Then find triangles in H_c.
            # A triangle in H_c corresponds to 3 pairs P1, P2, P3.
            # We must check if P1, P2, P3 are disjoint.
            
            # Optimization:
            # Instead of building full H_c and finding triangles,
            # we can iterate over all pairs of pairs that form a K2,2.
            # Let S be the list of "K2,2 blocks" (u,v,x,y) where all cross edges are c.
            # A K2,2,2 is formed by 3 such blocks sharing the same 6 vertices?
            # No.
            
            # Let's stick to the H_c triangle approach but optimize it.
            # M = 435 for n=30.
            # Building H_c:
            # For each (u,v), C has size ~ n/4 = 7.
            # Pairs in C: ~21.
            # Total edges in H_c ~ 435 * 21 / 2 ~ 4500.
            # Finding triangles in a graph with 435 nodes and 4500 edges.
            # Algorithm:
            #   For each node u in H_c:
            #     For each neighbor v of u, v > u:
            #       For each neighbor w of v, w > v:
            #         If w is neighbor of u: Triangle (u,v,w).
            
            # This is fast.
            
            # We need to map pair (u,v) to an integer index 0..M-1.
            # And back.
            
            # Let's create the mapping.
            pair_to_idx = {}
            idx_to_pair = []
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    pair_to_idx[(i, j)] = idx
                    idx_to_pair.append((i, j))
                    idx += 1
            
            M = len(idx_to_pair)
            
            # Build H_c adjacency list
            # H_adj[i] is a set of j such that pair i and pair j form a K2,2 in G_c
            H_adj = [set() for _ in range(M)]
            
            for u in range(n):
                for v in range(u + 1, n):
                    # Common neighbors of u and v in G_c
                    mask_u = adj[c][u]
                    mask_v = adj[c][v]
                    common_mask = mask_u & mask_v
                    # Remove u and v from common_mask? 
                    # u and v are not neighbors of themselves in simple graph, so bits are 0.
                    # But we need x,y distinct from u,v.
                    # Since adj[c][u] has 0 at u, and adj[c][v] has 0 at v,
                    # common_mask will have 0 at u and v automatically?
                    # Yes, if graph is simple.
                    
                    # Get list of common neighbors
                    # We need to generate all pairs (x,y) from common neighbors.
                    # To be efficient, we can iterate over bits.
                    cn_list = []
                    m = common_mask
                    while m:
                        # Get lowest set bit
                        lsb = m & -m
                        idx_cn = lsb.bit_length() - 1
                        cn_list.append(idx_cn)
                        m ^= lsb
                    
                    if len(cn_list) < 2:
                        continue
                    
                    idx_uv = pair_to_idx[(u, v)]
                    
                    # Generate all pairs (x, y) from cn_list
                    for i_cn in range(len(cn_list)):
                        x = cn_list[i_cn]
                        for j_cn in range(i_cn + 1, len(cn_list)):
                            y = cn_list[j_cn]
                            # Ensure x < y for consistent pair key
                            if x > y:
                                x, y = y, x
                            # Check if x,y are distinct from u,v
                            # They are in common neighbors, so they are neighbors of u and v.
                            # They are not u or v (since u,v not in their own neighbor sets).
                            # So {u,v} and {x,y} are disjoint.
                            
                            idx_xy = pair_to_idx[(x, y)]
                            H_adj[idx_uv].add(idx_xy)
                            H_adj[idx_xy].add(idx_uv)
            
            # Now find triangles in H_c
            # To avoid counting each triangle 6 times, we enforce ordering:
            # u < v < w in terms of their indices in H_c?
            # Or just count unique sets.
            
            # We iterate u, then v in H_adj[u] with v > u,
            # then w in H_adj[v] with w > v,
            # and check if w in H_adj[u].
            
            # To make "v > u" work, we need the indices.
            
            for u_idx in range(M):
                neighbors_u = H_adj[u_idx]
                for v_idx in neighbors_u:
                    if v_idx <= u_idx:
                        continue
                    neighbors_v = H_adj[v_idx]
                    for w_idx in neighbors_v:
                        if w_idx <= v_idx:
                            continue
                        if w_idx in neighbors_u:
                            # Triangle found: (u_idx, v_idx, w_idx)
                            # Check if the 3 pairs are disjoint
                            p1 = idx_to_pair[u_idx]
                            p2 = idx_to_pair[v_idx]
                            p3 = idx_to_pair[w_idx]
                            
                            # Check disjointness
                            # p1, p2, p3 are tuples (a,b), (c,d), (e,f)
                            # All 6 vertices must be distinct.
                            if (p1[0] != p2[0] and p1[0] != p2[1] and p1[1] != p2[0] and p1[1] != p2[1] and
                                p1[0] != p3[0] and p1[0] != p3[1] and p1[1] != p3[0] and p1[1] != p3[1] and
                                p2[0] != p3[0] and p2[0] != p3[1] and p2[1] != p3[0] and p2[1] != p3[1]):
                                total_violations += 1
        
        return total_violations

    # Try to find a valid coloring for n=30
    # Start with Paley(29) + random vertex 30
    
    def get_paley_29_colors():
        n = 29
        squares = {x * x % n for x in range(1, n)}
        colors = [0 if (j - i) % n in squares else 1 for i in range(n) for j in range(i + 1, n)]
        return colors

    base_colors_29 = get_paley_29_colors()
    
    # Construct initial 30-vertex coloring
    # Vertex 29 is the new one.
    # Edges (29, i) for i in 0..28.
    # There are 29 edges.
    
    best_n = 29
    best_colors = base_colors_29
    
    # Try local search for n=30
    # We will use a random restart strategy if the first attempt fails quickly.
    # But 140s is enough for one good attempt.
    
    # Initial random edges for vertex 29
    random.seed(42)
    extra_edges = [random.randint(0, 1) for _ in range(29)]
    
    # Combine
    # The order of edges in the flat list:
    # (0,1), (0,2), ..., (0,28), (0,29), (1,2), ..., (1,29), ..., (28,29)
    # Wait, lexicographic order:
    # i=0: j=1..29
    # i=1: j=2..29
    # ...
    # i=28: j=29
    
    # We need to insert the new edges into the correct positions.
    # Or just build the full list from scratch.
    
    def build_colors_30(extra_edges):
        n = 30
        colors = []
        idx_extra = 0
        for i in range(n):
            for j in range(i + 1, n):
                if i == 29:
                    # Edge (29, j) doesn't exist in i<j loop for i=29?
                    # j > 29, but max j is 29. So no edges with i=29 in this loop?
                    # Wait, i < j.
                    # If i=28, j=29.
                    # If i=29, no j > 29.
                    # So the edges involving 29 are (0,29), (1,29), ..., (28,29).
                    # In the loop:
                    # i=0, j=29 -> index 28 in j-loop?
                    # Let's trace:
                    # i=0: j=1,2,...,29. (29 edges)
                    # i=1: j=2,...,29. (28 edges)
                    # ...
                    # i=k: j=k+1,...,29. (29-k edges)
                    
                    # The edges (i, 29) appear when j=29.
                    # For i=0, j=29 is the last edge of i=0.
                    # For i=1, j=29 is the last edge of i=1.
                    
                    # So we can just fill them in.
                    pass
                
                if j == 29:
                    # This is an edge (i, 29)
                    colors.append(extra_edges[i])
                else:
                    # This is an edge (i, j) where j < 29
                    # We need to get the color from base_colors_29
                    # Index in base_colors_29 for (i, j):
                    # Sum of edges for rows 0..i-1 + (j - i - 1)
                    # Row k has 29 - 1 - k edges? No, row k has (29-1) - k edges?
                    # For n=29, row i has 29 - i - 1 edges.
                    # Cumulative sum for row i:
                    # S(i) = sum_{k=0}^{i-1} (29 - k - 1) = sum_{k=0}^{i-1} (28 - k)
                    # S(i) = 28*i - i(i-1)/2
                    idx_base = 28 * i - i * (i - 1) // 2 + (j - i - 1)
                    colors.append(base_colors_29[idx_base])
        return colors

    # Initial colors
    colors_30 = build_colors_30(extra_edges)
    
    # Check initial violations
    v = count_violations(30, colors_30)
    
    if v == 0:
        return {'n': 30, 'colors': colors_30}
    
    # Local Search
    # We will flip edges randomly and accept if it reduces violations.
    # To speed up, we can try flipping edges incident to vertex 29 first, 
    # then all edges.
    
    # Number of edges: 435
    # We can precompute the indices of edges incident to 29.
    # Edges (i, 29) for i=0..28.
    # In the flat list, where are they?
    # i=0, j=29: idx = 28*0 - 0 + 28 = 28?
    # Let's re-verify index formula for n=30.
    # Row i has 30 - i - 1 edges.
    # Edges are (i, i+1), ..., (i, 29).
    # The edge (i, 29) is the last one in row i.
    # Index = Sum_{k=0}^{i-1} (30 - k - 1) + (29 - i - 1)
    # Sum_{k=0}^{i-1} (29 - k) = 29*i - i(i-1)/2
    # Offset in row i: 29 - i - 1 = 28 - i
    # Total idx = 29*i - i(i-1)/2 + 28 - i = 28*i - i(i-1)/2 + 28
    
    edge_indices_29 = []
    for i in range(29):
        idx = 28 * i - i * (i - 1) // 2 + 28
        edge_indices_29.append(idx)
    
    # All edge indices
    all_edges = list(range(435))
    
    # Local search loop
    # We'll use a simple hill climbing with random restarts if stuck.
    # But 140s is a lot. Let's just do one long run.
    
    # To make it faster, we can restrict the search space initially to edges incident to 29.
    # If that fails, expand to all edges.
    
    search_space = edge_indices_29
    # If stuck after some time, expand search_space to all_edges
    
    current_v = v
    best_v = current_v
    best_colors = colors_30[:]
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        # Pick random edge from search space
        e_idx = random.choice(search_space)
        
        # Flip it
        colors_30[e_idx] = 1 - colors_30[e_idx]
        
        # Calculate new violations
        new_v = count_violations(30, colors_30)
        
        if new_v < current_v:
            current_v = new_v
            if new_v < best_v:
                best_v = new_v
                best_colors = colors_30[:]
                if best_v == 0:
                    return {'n': 30, 'colors': best_colors}
        else:
            # Revert
            colors_30[e_idx] = 1 - colors_30[e_idx]
            
        # Every 1000 steps, if stuck, expand search space
        if steps % 1000 == 0 and search_space is not all_edges:
            if current_v == best_v: # Stuck
                search_space = all_edges
    
    # If no solution found for 30, return 29
    return {'n': 29, 'colors': base_colors_29}

# EVOLVE-BLOCK-END

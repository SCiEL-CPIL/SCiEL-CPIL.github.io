import time
import sys
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s

    def time_exceeded():
        return time.time() - start_time > deadline

    # Precompute for q=29
    q = 29
    
    def build_paley_colors(n):
        """Build Paley coloring for prime n ≡ 1 (mod 4)."""
        squares = set()
        for x in range(1, n):
            squares.add((x * x) % n)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    colors_29 = build_paley_colors(q)
    
    def edge_index(n, i, j):
        """Get the index of edge (i,j) in the colors list, where i < j."""
        if i > j:
            i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def build_adj(n, colors):
        """Build bitmask adjacency for each color."""
        adj = [[0] * n for _ in range(2)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj

    def has_k222(adj_color, n):
        """
        Check if adj_color (list of bitmasks) contains K_{2,2,2}.
        Returns (vertices) if found, else None.
        Uses bitmask operations for speed.
        """
        for i in range(n):
            for j in range(i + 1, n):
                if adj_color[i] & (1 << j):
                    continue  # skip pairs with an edge in this color
                
                # Common neighbors of i and j in this color
                cn_mask = (adj_color[i] & adj_color[j]) & ~(1 << i) & ~(1 << j)
                
                if cn_mask.bit_count() < 4:
                    continue
                
                # Need to find two non-adjacent pairs in cn_mask
                # that share at least 2 common neighbors (also non-adjacent to each other)
                # More precisely: need k,l in cn_mask with no edge, and m,o in cn_mask
                # with no edge, such that m,o are both adjacent to i,j,k,l
                
                # Iterate over pairs (k,l) in cn_mask that are NOT adjacent
                cn_list = []
                mask = cn_mask
                while mask:
                    lsb = mask & (-mask)
                    v = lsb.bit_length() - 1
                    cn_list.append(v)
                    mask ^= lsb
                
                if len(cn_list) < 4:
                    continue
                
                # For each non-edge pair (k,l) in cn_list
                for ai in range(len(cn_list)):
                    k = cn_list[ai]
                    for bi in range(ai + 1, len(cn_list)):
                        l = cn_list[bi]
                        if adj_color[k] & (1 << l):
                            continue  # k,l are adjacent, skip
                        
                        # Now need m,o that are common neighbors of i,j,k,l
                        # and m,o are NOT adjacent
                        cn2_mask = adj_color[i] & adj_color[j] & adj_color[k] & adj_color[l]
                        cn2_mask &= ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)
                        
                        if cn2_mask.bit_count() < 2:
                            continue
                        
                        # Check if cn2_mask contains a non-edge pair
                        cn2_list = []
                        mask2 = cn2_mask
                        while mask2:
                            lsb2 = mask2 & (-mask2)
                            v2 = lsb2.bit_length() - 1
                            cn2_list.append(v2)
                            mask2 ^= lsb2
                        
                        for ci in range(len(cn2_list)):
                            m = cn2_list[ci]
                            for di in range(ci + 1, len(cn2_list)):
                                o = cn2_list[di]
                                if not (adj_color[m] & (1 << o)):
                                    return (i, j, k, l, m, o)
        
        return None

    def find_k222(n, colors):
        """Find a K_{2,2,2} in either color class. Returns (color, vertices) or (None, None)."""
        adj = build_adj(n, colors)
        
        result = has_k222(adj[0], n)
        if result is not None:
            return (0, result)
        
        result = has_k222(adj[1], n)
        if result is not None:
            return (1, result)
        
        return (None, None)

    def find_k222_fast(n, adj):
        """Find K2,2,2 given pre-built adjacency. Returns (color, vertices) or (None, None)."""
        result = has_k222(adj[0], n)
        if result is not None:
            return (0, result)
        result = has_k222(adj[1], n)
        if result is not None:
            return (1, result)
        return (None, None)

    def update_adj(adj, n, i, j, old_color, new_color):
        """Update adjacency bitmasks when flipping edge (i,j)."""
        adj[old_color][i] &= ~(1 << j)
        adj[old_color][j] &= ~(1 << i)
        adj[new_color][i] |= (1 << j)
        adj[new_color][j] |= (1 << i)

    def flip_edge(n, colors, adj, i, j):
        """Flip edge (i,j) in colors and update adj."""
        idx = edge_index(n, i, j)
        old_c = colors[idx]
        new_c = 1 - old_c
        colors[idx] = new_c
        update_adj(adj, n, i, j, old_c, new_c)

    # Verify Paley(29) is valid
    color, vertices = find_k222(q, colors_29)
    
    if color is None:
        # Paley(29) is valid!
        best_n = q
        best_colors = colors_29[:]
        
        if not time_exceeded():
            random.seed(42)
            
            # Try to extend to n=30
            # Build adj for n=29 first
            adj_29 = build_adj(29, colors_29)
            
            # For n=30, we need to assign colors for edges (i, 29) for i=0..28
            # and verify no K2,2,2 in either color
            
            # Strategy: use a greedy + local search approach
            # For each vertex i, choose color for edge (i,29) to minimize
            # the number of "dangerous" configurations
            
            max_attempts = 20
            for attempt in range(max_attempts):
                if time_exceeded():
                    break
                
                colors_30 = colors_29[:]
                
                # Initialize new vertex edges
                if attempt < 5:
                    # Deterministic patterns
                    pattern = [0]*29
                    if attempt == 0:
                        for i in range(29):
                            pattern[i] = 0
                    elif attempt == 1:
                        for i in range(29):
                            pattern[i] = 1
                    elif attempt == 2:
                        for i in range(29):
                            pattern[i] = i % 2
                    elif attempt == 3:
                        for i in range(29):
                            pattern[i] = (i // 5) % 2
                    else:
                        for i in range(29):
                            pattern[i] = (i * 13) % 2
                else:
                    pattern = [random.randint(0, 1) for _ in range(29)]
                
                for i in range(29):
                    colors_30.append(pattern[i])
                
                # Build adj for n=30
                adj_30 = build_adj(30, colors_30)
                
                # Local search: flip edges to eliminate K2,2,2
                max_iter = 200
                for iteration in range(max_iter):
                    if time_exceeded():
                        break
                    
                    color, verts = find_k222_fast(30, adj_30)
                    if color is None:
                        # Valid n=30 coloring found!
                        return {'n': 30, 'colors': colors_30}
                    
                    # verts = (i, j, k, l, m, o) where (i,j), (k,l), (m,o) are non-edges
                    # and all cross-edges are color `color`
                    # To break: flip one cross-edge
                    pairs = [(verts[0], verts[1]), 
                             (verts[2], verts[3]), 
                             (verts[4], verts[5])]
                    
                    cross_edges = []
                    for p1 in range(3):
                        for p2 in range(p1 + 1, 3):
                            for v1 in pairs[p1]:
                                for v2 in pairs[p2]:
                                    cross_edges.append((v1, v2))
                    
                    # Prefer flipping edges involving vertex 29 (new vertex)
                    # as they're "easier" to change
                    edges_with_29 = [(a,b) for a,b in cross_edges if 29 in (a,b)]
                    if edges_with_29 and random.random() < 0.7:
                        v1, v2 = random.choice(edges_with_29)
                    else:
                        v1, v2 = random.choice(cross_edges)
                    
                    flip_edge(30, colors_30, adj_30, v1, v2)
                
                # Final check
                color, verts = find_k222_fast(30, adj_30)
                if color is None:
                    return {'n': 30, 'colors': colors_30}
            
            # If we couldn't find n=30, return the best valid coloring
            return {'n': best_n, 'colors': best_colors}
        else:
            return {'n': q, 'colors': colors_29}
    else:
        # Paley(29) is NOT valid - need local search on n=29
        random.seed(42)
        
        colors = colors_29[:]
        adj = build_adj(29, colors)
        
        max_iter = 3000
        for iteration in range(max_iter):
            if time_exceeded():
                break
            
            color, verts = find_k222_fast(29, adj)
            if color is None:
                return {'n': 29, 'colors': colors}
            
            pairs = [(verts[0], verts[1]), 
                     (verts[2], verts[3]), 
                     (verts[4], verts[5])]
            
            cross_edges = []
            for p1 in range(3):
                for p2 in range(p1 + 1, 3):
                    for v1 in pairs[p1]:
                        for v2 in pairs[p2]:
                            cross_edges.append((v1, v2))
            
            v1, v2 = random.choice(cross_edges)
            flip_edge(29, colors, adj, v1, v2)
        
        # Return best effort
        return {'n': 29, 'colors': colors}

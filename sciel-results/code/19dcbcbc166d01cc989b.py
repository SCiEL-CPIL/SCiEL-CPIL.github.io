import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    def verify_coloring_fast(n, colors):
        """
        Verify that a 4-coloring of K_n has no monochromatic C6.
        colors: list of length n*(n-1)//2, values in {0,1,2,3}.
        """
        num_edges = n * (n - 1) // 2
        if len(colors) != num_edges:
            return False
        
        # Build adjacency bitmasks for each color
        adj = [[0] * n for _ in range(4)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        # Check for C6 in each color class
        for c in range(4):
            if has_c6(adj[c], n):
                return False
        return True

    def has_c6(adj, n):
        """
        Check if the graph defined by bitmask adjacency 'adj' contains a C6.
        A C6 is a cycle of length 6.
        We iterate over pairs of non-adjacent vertices (u, v) and check if there is a path of length 5 between them?
        No, standard C6 detection: 
        A cycle v0-v1-v2-v3-v4-v5-v0.
        We can detect by checking for paths of length 5 between adjacent vertices?
        Actually, it's easier to check for "chords" or use the property:
        A graph has a C6 if and only if there exist vertices u, v such that the number of common neighbors is ... no.
        
        Efficient method:
        For each edge (u, v), check if there is a path of length 4 between u and v that does not use u or v internally?
        No, that would form a C6 if u and v are connected?
        Cycle: u - a - b - c - d - v - u.
        Yes, if (u,v) is an edge, and there is a path u-a-b-c-d-v of length 5 (4 internal vertices), then we have a C6.
        Wait, path length 5 means 5 edges. u-a (1), a-b (2), b-c (3), c-d (4), d-v (5). Plus v-u (6). Total 6 edges.
        So: For each edge (u, v), count paths of length 4 from u to v?
        No, path of length 4 from u to v is u-a-b-c-v (4 edges). Plus v-u makes 5 edges (C5).
        So for C6, we need a path of length 5 from u to v.
        u-a-b-c-d-v.
        
        Let's use the property: 
        A C6 exists iff there is a pair of vertices (u, v) at distance 2, 3, or 4 that are connected by a path of the appropriate length?
        
        Simpler robust check for n=18:
        Iterate all 6-tuples? 18P6 is too big (18*17*16*15*14*13 ~ 13 million). With pruning it might work in Python? 
        13M * 4 colors * 150s... maybe too slow in pure Python if not optimized.
        
        Let's use the "Path of length 5 between adjacent vertices" approach with bitsets.
        For each edge (u, v):
          We want to know if there exists a path u -> a -> b -> c -> d -> v.
          a in N(u) \ {v}
          d in N(v) \ {u}
          b in N(a) \ {u, v, d}
          c in N(d) \ {u, v, a}
          And (b, c) is an edge.
          
          This is O(E * deg(u) * deg(v) * deg(a) * deg(d) * ...) which can be heavy.
          
          Alternative: 
          Precompute `common_neighbors`?
          
          Let's stick to the bitset intersection method used in the seed, which was:
          For edge (u, v):
            a in N(u)\{v}, d in N(v)\{u}, a != d
            b in N(a)\{u,v,d}, c in N(d)\{u,v,a}, b != c
            Check if (b,c) in E.
            
          This is what `has_c6_bitset` in the seed did. It was fast enough for verification.
        """
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    continue
                # u and v are adjacent. Look for path u-a-b-c-d-v of length 5.
                neighbors_u = adj[u] & ~(1 << v) & ~(1 << u)
                neighbors_v = adj[v] & ~(1 << u) & ~(1 << v)
                
                # Iterate a in N(u)\{v}
                a_mask = neighbors_u
                while a_mask:
                    a = (a_mask & -a_mask).bit_length() - 1
                    a_mask &= a_mask - 1
                    
                    # d in N(v)\{u, a}
                    d_mask = neighbors_v & ~(1 << a)
                    while d_mask:
                        d = (d_mask & -d_mask).bit_length() - 1
                        d_mask &= d_mask - 1
                        
                        # b in N(a)\{u, v, d}
                        # c in N(d)\{u, v, a}
                        # and (b,c) in E
                        
                        b_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << d) & ~(1 << a)
                        c_mask = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                        
                        # We need b in b_mask, c in c_mask, and (b,c) in E.
                        # This is equivalent to: exists b in b_mask such that adj[b] & c_mask is non-zero.
                        # Note: b != c is guaranteed if we are careful? 
                        # If b == c, then (b,c) is a self-loop, which doesn't exist in simple graphs.
                        # So adj[b] & c_mask != 0 implies there is an edge (b,c) with b != c.
                        
                        b_iter = b_mask
                        while b_iter:
                            b = (b_iter & -b_iter).bit_length() - 1
                            b_iter &= b_iter - 1
                            if adj[b] & c_mask:
                                return True
        return False

    def get_known_17():
        n = 17
        dist_assign = [0, 1, 2, 3, 1, 0, 3, 2]
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                d_min = min(d, n - d)
                c = dist_assign[d_min - 1]
                colors.append(c)
        if verify_coloring_fast(n, colors):
            return {'n': n, 'colors': colors}
        # Fallback
        n = 9
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                d = j - i
                d_min = min(d, n - d)
                colors.append((d_min - 1) % 4)
        return {'n': n, 'colors': colors}

    def backtrack_solve(n, colors, adj, current_time_limit):
        """
        CSP Backtracking with Forward Checking.
        colors: list of length n*(n-1)//2. -1 means unassigned.
        adj: list of 4 lists of bitmasks.
        """
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        
        # Precompute edge index for (i, j)
        edge_idx = {}
        for idx, (i, j) in enumerate(edges):
            edge_idx[(i, j)] = idx
            
        def get_edge_idx(i, j):
            if i > j: i, j = j, i
            return edge_idx[(i, j)]

        def check_c6_in_color(c):
            return has_c6(adj[c], n)

        # Recursive search
        def search(k, time_limit):
            if time.time() > time_limit:
                return False
            
            if k == num_edges:
                # All assigned, final check (should be redundant if FC works perfectly, but safe)
                for c in range(4):
                    if check_c6_in_color(c):
                        return False
                return True

            i, j = edges[k]
            # Try each color
            for c in range(4):
                # Assign
                colors[k] = c
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                
                # Forward Check: 
                # Adding (i,j) to color c might create a C6 in color c.
                # If it does, this branch is invalid.
                if check_c6_in_color(c):
                    # Undo
                    adj[c][i] &= ~(1 << j)
                    adj[c][j] &= ~(1 << i)
                    colors[k] = -1
                    continue
                
                # Recurse
                if search(k + 1, time_limit):
                    return True
                
                # Undo
                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
                colors[k] = -1
                
            return False

        return search(0, current_time_limit)

    # Strategy:
    # 1. Secure n=17
    # 2. Try to solve n=18 using Backtracking CSP
    #    We will use a randomized edge ordering to help escape symmetries and find solutions faster.
    #    Standard lexicographic ordering might be bad. 
    #    However, implementing a dynamic variable ordering (MRV) is complex.
    #    Let's try a few random permutations of edge assignments for n=18.
    
    result_17 = get_known_17()
    if result_17['n'] < 17:
        return result_17
        
    n_target = 18
    time_for_18 = deadline - time.time() - 5
    
    if time_for_18 > 10:
        # Prepare data structures for n=18
        edges = [(i, j) for i in range(n_target) for j in range(i + 1, n_target)]
        num_edges = len(edges)
        
        # Try a few random edge orderings
        # We can't just shuffle the `colors` list easily with the `edges` list in the backtrack function 
        # unless we pass the permutation.
        
        # Let's modify the approach: 
        # Create a permutation of edge indices.
        # The backtrack function should use `order[k]` to pick the k-th edge to assign.
        
        def backtrack_solve_with_order(n, order, current_time_limit):
            colors = [-1] * (n * (n - 1) // 2)
            adj = [[0] * n for _ in range(4)]
            
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            edge_idx = {}
            for idx, (i, j) in enumerate(edges):
                edge_idx[(i, j)] = idx

            def check_c6_in_color(c):
                return has_c6(adj[c], n)

            def search(k, time_limit):
                if time.time() > time_limit:
                    return False
                
                if k == len(order):
                    # All assigned
                    for c in range(4):
                        if check_c6_in_color(c):
                            return False
                    return True

                idx = order[k]
                i, j = edges[idx]
                
                # Heuristic: Try colors in a random order? Or 0-3?
                # Random color order can help with symmetry breaking.
                color_order = [0, 1, 2, 3]
                random.shuffle(color_order)
                
                for c in color_order:
                    # Assign
                    colors[idx] = c
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    
                    # Check if this creates a C6 in color c
                    if check_c6_in_color(c):
                        adj[c][i] &= ~(1 << j)
                        adj[c][j] &= ~(1 << i)
                        colors[idx] = -1
                        continue
                    
                    if search(k + 1, time_limit):
                        return True
                    
                    adj[c][i] &= ~(1 << j)
                    adj[c][j] &= ~(1 << i)
                    colors[idx] = -1
                    
                return False

            success = search(0, current_time_limit)
            if success:
                return colors
            return None

        # Try multiple random orderings
        # Each attempt gets a slice of the remaining time
        num_attempts = 10
        time_per_attempt = time_for_18 / num_attempts
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
            
            remaining = deadline - time.time() - 5
            if remaining < 2:
                break
            
            # Random permutation of edge indices
            order = list(range(num_edges))
            random.shuffle(order)
            
            # Run solver
            result_colors = backtrack_solve_with_order(n_target, order, time.time() + min(time_per_attempt, remaining))
            
            if result_colors is not None:
                # Verify just in case
                if verify_coloring_fast(n_target, result_colors):
                    return {'n': n_target, 'colors': result_colors}
            
    return result_17

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")

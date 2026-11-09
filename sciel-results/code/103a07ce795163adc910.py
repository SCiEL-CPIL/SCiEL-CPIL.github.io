import time
import random
import sys
from collections import deque

def construct():
    start_time = time.time()
    deadline = 115  # Leave 5s margin for safety
    
    def solve(n, seed_val=0):
        """
        Attempt to construct a valid 4-coloring for K_n.
        Returns col_matrix if successful, None otherwise.
        """
        random.seed(seed_val)
        
        # adj_c[c][i] is a set of neighbors j of i with edge (i,j) colored c
        adj_c = [[set() for _ in range(n)] for _ in range(4)]
        
        # Using a 2D array for color storage
        col_matrix = [[-1]*n for _ in range(n)]
        
        # Precompute: for each color c, maintain adjacency bitmasks for fast subset operations
        # adj_mask[c][i] = bitmask of neighbors of i in color c
        adj_mask = [[0] * n for _ in range(4)]
        
        def update_adj_mask(c, u, v):
            """Add edge (u,v) to color c in adj_mask."""
            adj_mask[c][u] |= (1 << v)
            adj_mask[c][v] |= (1 << u)
        
        def remove_adj_mask(c, u, v):
            """Remove edge (u,v) from color c in adj_mask."""
            adj_mask[c][u] &= ~(1 << v)
            adj_mask[c][v] &= ~(1 << u)
        
        def has_path_len_5(c, u, v):
            """
            Check if there is a path of exactly length 5 from u to v in G_c.
            Path: u - x1 - x2 - x3 - x4 - v
            All intermediate vertices distinct and not u,v.
            """
            # Use bitmasks for efficient subset operations
            # We need to find x1, x2, x3, x4 such that:
            # u-x1, x1-x2, x2-x3, x3-x4, x4-v all in color c
            # with all vertices distinct
            
            u_mask = adj_mask[c][u]
            v_mask = adj_mask[c][v]
            
            # Remove u and v from consideration for intermediates
            u_mask &= ~(1 << v)  # x1 != v
            v_mask &= ~(1 << u)  # x4 != u
            
            if u_mask == 0 or v_mask == 0:
                return False
            
            # Enumerate x1 from neighbors of u
            x1_list = []
            mask = u_mask
            while mask:
                bit = mask & (-mask)
                x1 = bit.bit_length() - 1
                x1_list.append(x1)
                mask -= bit
            
            if not x1_list:
                return False
            
            # Enumerate x4 from neighbors of v
            x4_list = []
            mask = v_mask
            while mask:
                bit = mask & (-mask)
                x4 = bit.bit_length() - 1
                x4_list.append(x4)
                mask -= bit
            
            if not x4_list:
                return False
            
            # For each (x1, x4), check if there's a path of length 3 from x1 to x4
            # avoiding u, v, x1, x4
            for x1 in x1_list:
                # Neighbors of x1, excluding u, v, x1, and we'll exclude x4 later
                x1_neigh_mask = adj_mask[c][x1]
                x1_neigh_mask &= ~(1 << u)
                x1_neigh_mask &= ~(1 << v)
                x1_neigh_mask &= ~(1 << x1)
                
                if x1_neigh_mask == 0:
                    continue
                
                # Enumerate x2 from neighbors of x1
                x2_list = []
                mask = x1_neigh_mask
                while mask:
                    bit = mask & (-mask)
                    x2 = bit.bit_length() - 1
                    x2_list.append(x2)
                    mask -= bit
                
                if not x2_list:
                    continue
                
                for x4 in x4_list:
                    if x4 == x1:
                        continue
                    
                    # For this x4, check if any x2 connects to any x3 which connects to x4
                    # x3 must be neighbor of x4, excluding u, v, x1, x4
                    x4_neigh_mask = adj_mask[c][x4]
                    x4_neigh_mask &= ~(1 << u)
                    x4_neigh_mask &= ~(1 << v)
                    x4_neigh_mask &= ~(1 << x1)
                    x4_neigh_mask &= ~(1 << x4)
                    
                    if x4_neigh_mask == 0:
                        continue
                    
                    # Enumerate x3 from neighbors of x4
                    x3_list = []
                    mask = x4_neigh_mask
                    while mask:
                        bit = mask & (-mask)
                        x3 = bit.bit_length() - 1
                        x3_list.append(x3)
                        mask -= bit
                    
                    if not x3_list:
                        continue
                    
                    # Check if any x2 in x2_list is connected to any x3 in x3_list
                    # and x2 != x3
                    x2_set = set(x2_list)
                    for x3 in x3_list:
                        if x3 in x2_set:
                            continue
                        # Check if x3 is neighbor of x2
                        if adj_mask[c][x2] & (1 << x3):
                            return True
            
            return False
        
        def check_no_6_cycle(c, u, v):
            """
            Check if adding edge (u,v) with color c creates a 6-cycle.
            This happens iff there is a path of length 5 from u to v in G_c (without the new edge).
            """
            return has_path_len_5(c, u, v)
        
        # Backtracking with vertex-by-vertex construction
        def build_vertex(k):
            if time.time() > deadline:
                return False
            if k == n:
                return True
            
            # Assign colors to edges (i, k) for i in 0..k-1
            def assign_edge(i):
                if time.time() > deadline:
                    return False
                if i == k:
                    return build_vertex(k + 1)
                
                u = i
                v = k
                
                # Try colors in a smart order
                # Prefer colors that are less "constrained"
                c_list = [0, 1, 2, 3]
                random.shuffle(c_list)
                
                for c in c_list:
                    if not check_no_6_cycle(c, u, v):
                        col_matrix[u][v] = c
                        col_matrix[v][u] = c
                        adj_c[c][u].add(v)
                        adj_c[c][v].add(u)
                        update_adj_mask(c, u, v)
                        
                        if assign_edge(i + 1):
                            return True
                        
                        adj_c[c][u].remove(v)
                        adj_c[c][v].remove(u)
                        remove_adj_mask(c, u, v)
                        col_matrix[u][v] = -1
                        col_matrix[v][u] = -1
                return False
            
            return assign_edge(0)
        
        return build_vertex(0)
    
    # Target n values - try from high to low
    best_result = None
    for target_n in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline:
            break
            
        # Try multiple restarts for this target_n
        max_attempts = 30 if target_n <= 14 else 10
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            seed_val = random.randint(0, 2**31 - 1)
            res = solve(target_n, seed_val)
            if res is not None:
                # Extract colors in lexicographic order
                colors = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        colors.append(res[i][j])
                return {'n': target_n, 'colors': colors}
        
        # If we get here, we failed for this target_n, try lower
    
    # Fallback: try to find any valid coloring
    for target_n in [9, 8, 7, 6, 5, 4, 3, 2]:
        if time.time() > deadline:
            break
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            seed_val = random.randint(0, 2**31 - 1)
            res = solve(target_n, seed_val)
            if res is not None:
                colors = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        colors.append(res[i][j])
                return {'n': target_n, 'colors': colors}
    
    # Final fallback: return a trivial valid coloring
    # For n=1, no edges, trivially valid
    return {'n': 1, 'colors': []}

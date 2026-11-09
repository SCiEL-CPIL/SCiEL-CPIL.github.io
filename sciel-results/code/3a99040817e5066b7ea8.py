import time
import random

def construct():
    """
    Constructs a 2-coloring of K_n with no monochromatic K2,2,2.
    Returns a dictionary with 'n' and 'colors'.
    
    Attempts to find a valid 30-vertex coloring by extending a 29-vertex
    Paley graph. If unsuccessful within the time limit, returns the
    valid 29-vertex coloring.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate base n=29 coloring (Paley graph)
    n_base = 29
    p = 29
    residues = set()
    for x in range(1, p):
        residues.add((x * x) % p)
    
    base_colors = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            diff = (j - i) % p
            if diff in residues:
                base_colors.append(0)
            else:
                base_colors.append(1)
    
    # Helper to convert list to dict for fast lookup
    def colors_to_dict(colors, n):
        d = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                d[(i, j)] = colors[idx]
                idx += 1
        return d

    base_dict = colors_to_dict(base_colors, n_base)
    
    # Helper to check if adding edge (u,v) with color c creates a mono K2,2,2
    # This check is expensive, so we only call it during search
    def has_mono_k222_with_edge(d, n, u, v, c):
        """
        Checks if there is a monochromatic K2,2,2 of color c containing edge (u,v).
        d: dict of edges
        n: number of vertices
        """
        # Case 1: u, v are in the same part (A)
        # Need to find w, x, y, z in V \ {u, v}
        # such that {w,x} is part B, {y,z} is part C
        # Edges (u,w),(u,x),(v,w),(v,x) must be c
        # Edges (u,y),(u,z),(v,y),(v,z) must be c
        # Edges (w,y),(w,z),(x,y),(x,z) must be c
        
        # Pre-filter candidates for B: vertices w such that (u,w)==c and (v,w)==c
        # Pre-filter candidates for C: vertices y such that (u,y)==c and (v,y)==c
        
        # Since u,v are in A, they are not in B or C.
        # We need 4 distinct vertices w,x,y,z from the other n-2 vertices.
        
        # Let's iterate over pairs for B and C.
        # To optimize, we can iterate over w, x for B and check if there are y, z for C.
        
        # Get list of all other vertices
        others = [k for k in range(n) if k != u and k != v]
        
        # Candidates for B (must connect to u and v with color c)
        cand_b = []
        for w in others:
            if d[(u, w) if u < w else (w, u)] == c and d[(v, w) if v < w else (w, v)] == c:
                cand_b.append(w)
        
        # Candidates for C (must connect to u and v with color c)
        cand_c = [w for w in cand_b] # C is subset of B candidates? No, C is distinct from B.
        # Actually, cand_b and cand_c are the same set of vertices that connect to u,v with color c.
        # Let's call this set S.
        S = cand_b
        
        # We need to pick two disjoint pairs {w,x} and {y,z} from S
        # such that all edges between {w,x} and {y,z} are color c.
        
        # If |S| < 4, no solution in this case.
        if len(S) < 4:
            pass
        else:
            # Iterate over pairs in S
            s_list = S
            for i in range(len(s_list)):
                for j in range(i+1, len(s_list)):
                    w = s_list[i]
                    x = s_list[j]
                    # Check if w,x can be part B.
                    # We need y,z in S \ {w,x} such that edges between {w,x} and {y,z} are c.
                    # Let's find candidates for C from S \ {w,x}
                    remaining = [k for k in S if k != w and k != x]
                    if len(remaining) < 2:
                        continue
                    
                    # Find y, z in remaining such that:
                    # (w,y)==c, (w,z)==c, (x,y)==c, (x,z)==c
                    # This means y and z must be in the intersection of neighbors of w and x (in color c)
                    # restricted to 'remaining'.
                    
                    # Since w,x are in S, they connect to u,v with c.
                    # We just need to check edges between w,x and y,z.
                    
                    # Build list of valid y's
                    valid_ys = []
                    for y in remaining:
                        if d[(w, y) if w < y else (y, w)] == c and d[(x, y) if x < y else (y, x)] == c:
                            valid_ys.append(y)
                    
                    # If there are at least 2 valid y's, say y1, y2, we need to check if (y1, y2) is c?
                    # No, y1, y2 are in the SAME part C. So edge (y1, y2) is NOT in K2,2,2.
                    # So we just need two distinct vertices y, z in valid_ys.
                    if len(valid_ys) >= 2:
                        return True
        
        # Case 2: u, v are in different parts (u in A, v in B)
        # Need w in A (completing u), x in B (completing v), y,z in C.
        # Edges:
        # A(u,w) to B(v,x): all 4 edges c.
        # A(u,w) to C(y,z): all 4 edges c.
        # B(v,x) to C(y,z): all 4 edges c.
        
        # w must satisfy: (u,w)==c (since u,w in A? No, edge (u,w) is NOT in K2,2,2. Wait.)
        # In K2,2,2, there are no edges within parts.
        # So (u,w) is NOT an edge in the subgraph.
        # But (u,v) IS an edge (A-B).
        # So w is in A. v is in B.
        # Edges from w (in A) to v (in B) must be c.
        # Edges from w (in A) to x (in B) must be c.
        # Edges from u (in A) to x (in B) must be c.
        
        # So w must connect to v and x with c.
        # x must connect to u and w with c.
        
        # Let's find candidates for w (in A \ {u}):
        # w must connect to v with c.
        # Let's find candidates for x (in B \ {v}):
        # x must connect to u with c.
        
        # Then for a pair (w,x), we need y,z in C \ {y,z} such that:
        # w connects to y,z with c.
        # x connects to y,z with c.
        # u connects to y,z with c.
        # v connects to y,z with c.
        # So y,z must connect to u, v, w, x with c.
        
        cand_w = []
        for k in range(n):
            if k == u or k == v: continue
            if d[(v, k) if v < k else (k, v)] == c:
                cand_w.append(k)
                
        cand_x = []
        for k in range(n):
            if k == u or k == v: continue
            if d[(u, k) if u < k else (k, u)] == c:
                cand_x.append(k)
                
        # Iterate over w in cand_w, x in cand_x
        # w and x must be distinct and not u, v.
        # Also w != x.
        
        for w in cand_w:
            for x in cand_x:
                if w == x: continue
                # Check if w and x are compatible (edge w-x not required to be c, but w in A, x in B, so (w,x) IS an edge in K2,2,2 and must be c)
                if d[(w, x) if w < x else (x, w)] != c:
                    continue
                
                # Now find y, z such that they connect to u, v, w, x with c.
                # y, z must be distinct from u, v, w, x.
                
                valid_ys = []
                for k in range(n):
                    if k in (u, v, w, x): continue
                    # Check connections to u, v, w, x
                    if (d[(u, k) if u < k else (k, u)] == c and
                        d[(v, k) if v < k else (k, v)] == c and
                        d[(w, k) if w < k else (k, w)] == c and
                        d[(x, k) if x < k else (k, x)] == c):
                        valid_ys.append(k)
                
                if len(valid_ys) >= 2:
                    return True
        
        return False

    # Backtracking search to extend base_colors to n=30
    def try_extend(base_colors, n_base, target_n, time_limit):
        """
        Tries to extend a valid coloring of K_n_base to K_target_n.
        Returns the extended colors if successful, else None.
        """
        if n_base == target_n:
            return base_colors
        
        # We add vertices one by one.
        # Current colors list
        cur_colors = list(base_colors)
        cur_n = n_base
        
        # We will add vertex cur_n, then cur_n+1, etc.
        # Edges for new vertex v are (v, 0), (v, 1), ..., (v, v-1)
        
        # Recursion depth is target_n - n_base
        # Here we only add 1 vertex (30-29=1), so it's simple.
        
        # Try all 2^29 combinations? No.
        # Use DFS with pruning.
        
        # State: list of colors for edges of the new vertex
        # new_edge_colors[i] is color of (new_vertex, i)
        
        new_v = cur_n
        edges_to_assign = list(range(cur_n))
        # Shuffle for randomization?
        random.shuffle(edges_to_assign)
        
        assignment = {}
        
        def dfs(idx):
            if time.time() > time_limit:
                return None
            
            if idx == len(edges_to_assign):
                # All edges assigned. Check if valid.
                # Since we pruned at each step, it should be valid.
                # But we should double check just in case.
                full_colors = list(cur_colors)
                # Append colors in lexicographic order
                # Lex order for new vertex v: (v, 0), (v, 1)... 
                # Wait, the interface requires lexicographic order of ALL pairs.
                # The base colors are for 0..28.
                # The new edges are (29, 0), (29, 1)... (29, 28).
                # These come AFTER all pairs in 0..28 in lexicographic order?
                # Pairs (i,j) with i<j.
                # Max pair in base is (28, 29)? No, base is 0..28. Max is (27, 28).
                # New vertex is 29.
                # Pairs involving 29 are (0,29), (1,29)... (28,29).
                # In lexicographic order, (0,29) comes after (0,28)?
                # (0,1)...(0,28), (1,2)...(1,28)... (28,29)?
                # Yes, all pairs (i, 29) come after all pairs (i, j) with j < 29.
                # So we can just append the new colors in order i=0..28.
                
                # However, my `assignment` dict maps edge index to color.
                # edges_to_assign is shuffled.
                # I need to reconstruct the list in order 0..28.
                
                new_edge_list = []
                for i in range(cur_n):
                    new_edge_list.append(assignment[i])
                    
                return full_colors + new_edge_list

            v = edges_to_assign[idx]
            
            for c in [0, 1]:
                # Tentatively assign (new_v, v) = c
                assignment[v] = c
                
                # Check if this creates a mono K2,2,2
                # We need to check if adding this edge creates a violation.
                # The check `has_mono_k222_with_edge` requires a dict.
                # Creating a dict every time is slow.
                # Optimization: Maintain the dict incrementally?
                # Or just pass the current state.
                
                # For simplicity and given the small number of steps (29),
                # we can construct the dict for the check.
                # But `has_mono_k222_with_edge` is O(n^3) or so.
                # 29 steps * 2 colors * O(29^3) is fast.
                
                # Create a temporary dict including the new edge
                # and the previous assignments.
                
                # To avoid creating a full dict of size 435+29 every time,
                # we can modify `has_mono_k222_with_edge` to accept a "partial" dict
                # or just use the base dict and the current assignment.
                
                # Let's create a helper that checks using base_dict and assignment.
                
                def check_violation(base_d, assign_dict, new_v, v, c, n):
                    # Check if edge (new_v, v) with color c creates a mono K2,2,2.
                    # This function is similar to `has_mono_k222_with_edge` 
                    # but uses `base_d` for existing edges and `assign_dict` for new ones.
                    
                    def get_color(u, w):
                        if u == new_v or w == new_v:
                            other = v if u == new_v else w
                            # If other is new_v, then u is v? No.
                            # If u==new_v, other=v. Edge is (new_v, v).
                            # But we are checking edges involving new_v.
                            # The edge being checked IS (new_v, v).
                            # Other edges involving new_v are in assign_dict.
                            # Edges not involving new_v are in base_d.
                            
                            if u == new_v:
                                # Edge is (new_v, w). w must be in assign_dict or w==v (current).
                                if w == v:
                                    return c
                                else:
                                    return assign_dict.get(w, None) # Should exist if assigned
                            else:
                                # w == new_v
                                if u == v:
                                    return c
                                else:
                                    return assign_dict.get(u, None)
                        else:
                            # Both in base
                            if u > w: u, w = w, u
                            return base_d.get((u, w), None)
                    
                    # The logic in has_mono_k222_with_edge relied on a full dict.
                    # I will reuse the logic but replace d[...] with get_color(...)
                    # This is too slow to rewrite inline.
                    
                    # Alternative: Just build the full dict. It's small.
                    temp_d = dict(base_d)
                    for k, val in assign_dict.items():
                        temp_d[(new_v, k) if new_v < k else (k, new_v)] = val
                    temp_d[(new_v, v) if new_v < v else (v, new_v)] = c
                    
                    return has_mono_k222_with_edge(temp_d, n, new_v, v, c)
                
                if not check_violation(base_dict, assignment, new_v, v, c, target_n):
                    res = dfs(idx + 1)
                    if res is not None:
                        return res
            
            del assignment[v]
            return None

        return dfs(0)

    # Attempt to extend
    result_colors = try_extend(base_colors, n_base, 30, deadline)
    
    if result_colors is not None:
        return {'n': 30, 'colors': result_colors}
    else:
        return {'n': 29, 'colors': base_colors}

# EVOLVE-BLOCK-END

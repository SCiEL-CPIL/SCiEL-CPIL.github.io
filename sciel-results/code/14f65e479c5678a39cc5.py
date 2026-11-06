import time
import sys
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 115.0
    
    prev_solution = None
    
    # Try increasing n
    for n in range(10, 18):
        if time.time() - start_time > time_limit:
            break
        
        # Allocate time based on n
        # n=10,11,12 should be fast. n=13,14 harder. n=15,16,17 hardest.
        if n <= 12:
            limit_for_n = 5.0
        elif n <= 14:
            limit_for_n = 15.0
        else:
            limit_for_n = 100.0 - (time.time() - start_time)
            
        if limit_for_n < 1.0:
            break

        sol = solve_c6(n, prev_solution, limit_for_n)
        if sol is not None:
            prev_solution = sol
        else:
            # If we fail for n, we might still want to return the last successful one?
            # The task says "Maximize n". So if we found n=15, and fail n=16, return n=15.
            # But we should keep trying if time permits?
            # Usually if n fails, n+1 will also fail or take longer.
            # But let's stick to the loop. If sol is None, we break or continue?
            # If n=10 fails, something is wrong. 
            # If n=17 fails, we return the best found so far.
            
            # Optimization: If we failed for n, it's unlikely n+1 will succeed quickly 
            # unless we have a very good seed. 
            # However, if we have prev_solution for n-1, extending to n is easier than from scratch.
            # So we continue the loop, but the time limit will likely cut us off.
            
            pass

    if prev_solution is None:
        # Fallback to a small known valid coloring if n=10 failed (should not happen)
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
    else:
        # Find the max n for which we have a solution
        # prev_solution corresponds to the last successful n
        # We need to track the n
        # Actually, the loop variable n is the one we just tried.
        # If sol was not None, prev_solution is for that n.
        # Let's restructure to track best_n.
        
        pass

def solve_c6(n, initial_colors, time_limit):
    """
    Solves 4-coloring of K_n without monochromatic C6.
    initial_colors: list of colors for K_{n-1} or None.
    """
    start_time = time.time()
    
    # Bitmasks for each color: mask[c][u] is a bitmask of vertices v such that (u,v) is color c
    masks = [[0] * n for _ in range(4)]
    
    # unassigned edges
    # We need to know which edges are unassigned.
    # For MRV, we iterate through all pairs (u,v) where u < v.
    
    # Initialize from initial_colors if provided
    if initial_colors is not None:
        # initial_colors is for n-1 vertices
        idx = 0
        for i in range(n - 1):
            for j in range(i + 1, n - 1):
                c = initial_colors[idx]
                masks[c][i] |= (1 << j)
                masks[c][j] |= (1 << i)
                idx += 1
        # Edges involving n-1 are unassigned
    else:
        # All unassigned
        pass

    # To support MRV efficiently, we might just iterate all edges.
    # For n=17, 136 edges.
    
    def is_safe(u, v, c):
        """
        Check if adding edge (u,v) with color c creates a C6.
        This is true if there is a path of length 5 in G_c between u and v.
        Path: u - a - b - d - e - v
        """
        # Neighbors of u in color c
        nu = masks[c][u]
        # Neighbors of v in color c
        nv = masks[c][v]
        
        # We need to find a - b - d such that a in nu, d in nv, 
        # and a-b-d is a path in G_c.
        # a != v, d != u (guaranteed by unassigned edge? 
        # If u-v is unassigned, u is not in nv and v is not in nu.
        # But other vertices might be connected.
        
        # Iterate a in nu
        # For each a, iterate b in neighbors of a (color c)
        # For each b, check if any neighbor d of b (color c) is in nv
        # AND d is not u, v, a, b.
        
        # Optimization: Precompute neighbors? 
        # masks[c][x] gives neighbors.
        
        # We must ensure distinct vertices: u, a, b, d, v.
        # a != u (true, no self loops)
        # a != v (true, uv unassigned)
        # b != u, v, a
        # d != u, v, a, b
        
        # Let's iterate a
        # nu is a bitmask
        while nu:
            lsb = nu & -nu
            a = lsb.bit_length() - 1
            nu ^= lsb
            
            # Neighbors of a
            na = masks[c][a]
            # b must be in na, b != u, b != v, b != a
            # Also b must not be such that the path closes early? 
            # The path is u-a-b-d-v.
            # b cannot be u or v.
            # b cannot be a.
            
            # Filter na
            # mask_for_b = na & ~(1<<u) & ~(1<<v) & ~(1<<a)
            mask_for_b = na & ~(1 << u) & ~(1 << v) & ~(1 << a)
            
            while mask_for_b:
                lsb_b = mask_for_b & -mask_for_b
                b = lsb_b.bit_length() - 1
                mask_for_b ^= lsb_b
                
                # Neighbors of b
                nb = masks[c][b]
                # d must be in nb, d in nv
                # d != u, v, a, b
                # d in nv implies d != v (since uv unassigned, v not in nv? 
                # Wait, nv is neighbors of v. d is a neighbor of v. So d != v.
                # d != u? If d=u, then u is neighbor of v in G_c. But uv is unassigned, so u not in nv.
                # So d != u and d != v are guaranteed if we just intersect nb and nv.
                # We must exclude a and b.
                
                # Intersection of nb and nv
                candidates = nb & nv
                # Exclude a and b
                candidates &= ~(1 << a)
                candidates &= ~(1 << b)
                
                if candidates:
                    # Found a path u-a-b-d-v
                    return False
        
        return True

    # MRV Backtracking
    # State: masks
    # We need to track which edges are assigned?
    # No, we can just try to assign any unassigned edge.
    # How to know if an edge is assigned?
    # If masks[c][u] has bit v set for some c, it's assigned.
    # Since it's a proper coloring, it has exactly one color.
    
    def find_mrv_edge():
        """
        Returns (u, v, valid_colors_list) for the edge with fewest valid colors.
        Returns None if no unassigned edges (solution found).
        Returns (None, None, []) if dead end (some edge has 0 valid colors).
        """
        min_count = 5
        best_u = -1
        best_v = -1
        best_colors = []
        
        for u in range(n):
            for v in range(u + 1, n):
                # Check if (u,v) is assigned
                assigned = False
                for c in range(4):
                    if masks[c][u] & (1 << v):
                        assigned = True
                        break
                
                if assigned:
                    continue
                
                # Count valid colors
                valid_colors = []
                for c in range(4):
                    if is_safe(u, v, c):
                        valid_colors.append(c)
                
                if len(valid_colors) < min_count:
                    min_count = len(valid_colors)
                    best_u = u
                    best_v = v
                    best_colors = valid_colors
                    
                    if min_count == 0:
                        return (u, v, []) # Dead end
                    if min_count == 1:
                        # Can't get better than 1 (unless 0)
                        # But we should check if any other edge has 0? 
                        # MRV usually picks min. If min is 1, we pick it.
                        # But if another edge has 0, we should detect that.
                        # So we can't break early on 1.
                        pass
        
        if best_u == -1:
            return None # No unassigned edges
        
        return (best_u, best_v, best_colors)

    def backtrack():
        if time.time() - start_time > time_limit:
            return False
        
        res = find_mrv_edge()
        if res is None:
            return True # All edges assigned
        
        u, v, colors = res
        
        if len(colors) == 0:
            return False # Dead end
        
        # Try colors
        # Symmetry breaking for the first few edges?
        # If we are extending from n-1, the first n-1 vertices are fixed.
        # The new edges are those involving n-1.
        # Symmetry breaking is tricky with MRV.
        # Let's rely on MRV.
        
        for c in colors:
            # Assign
            masks[c][u] |= (1 << v)
            masks[c][v] |= (1 << u)
            
            if backtrack():
                return True
            
            # Unassign
            masks[c][u] &= ~(1 << v)
            masks[c][v] &= ~(1 << u)
            
        return False

    if backtrack():
        # Extract colors
        colors_list = []
        for i in range(n):
            for j in range(i + 1, n):
                for c in range(4):
                    if masks[c][i] & (1 << j):
                        colors_list.append(c)
                        break
        return colors_list
    else:
        return None

# We need to fix the construct function to properly track the best n
def construct():
    start_time = time.time()
    time_limit = 115.0
    
    best_n = 0
    best_colors = None
    
    # Try increasing n
    for n in range(10, 18):
        if time.time() - start_time > time_limit:
            break
        
        # Allocate time
        if n <= 12:
            limit_for_n = 5.0
        elif n <= 14:
            limit_for_n = 15.0
        else:
            limit_for_n = 115.0 - (time.time() - start_time)
            
        if limit_for_n < 1.0:
            break

        # Use best_colors as initial if it matches n-1
        init = None
        if best_n == n - 1:
            init = best_colors
            
        sol = solve_c6(n, init, limit_for_n)
        if sol is not None:
            best_n = n
            best_colors = sol
            
    if best_colors is None:
        # Fallback
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
    else:
        return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END

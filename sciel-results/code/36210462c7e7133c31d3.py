import time
import sys
import random

def construct():
    deadline = time.time() + 145  # Safety margin
    
    # Try n=27 first with a small budget
    # Then n=26 with the remaining budget
    # If n=26 fails, try n=25, etc.
    
    # We know n=26 is possible (Lazebnik-Woldar).
    # We want to maximize n.
    
    # Fallback result
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    targets = [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]
    
    # Allocate time
    # n=27: 15s
    # n=26: 100s
    # others: remaining
    
    for n in targets:
        if time.time() > deadline - 5:
            break
            
        if n == 27:
            t_limit = 15
        elif n == 26:
            t_limit = 100
        else:
            t_limit = 10
            
        actual_limit = min(deadline - time.time() - 2, t_limit)
        if actual_limit < 2:
            continue
            
        success, colors = _solve_csp(n, time.time() + actual_limit)
        
        if success:
            best_n = n
            best_colors = colors
            break  # Found the largest n
            
    return {'n': best_n, 'colors': best_colors}

def _solve_csp(n, deadline):
    m = n * (n - 1) // 2
    # color[i][j] is the color of edge (i, j), or -1 if unassigned
    # Using a 2D list for fast access
    color = [[-1] * n for _ in range(n)]
    
    # common_counts[c][u][v] = number of common neighbors of u and v in color c
    # This is symmetric: common_counts[c][u][v] == common_counts[c][v][u]
    # Initialize to 0
    common_counts = [[[0] * n for _ in range(n)] for _ in range(5)]
    
    # List of edges to assign
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            
    # We will use a dynamic variable ordering: pick the edge with the fewest available colors
    # But calculating MRV for all edges is O(m * n).
    # Instead, we can use a static ordering that is known to work well, or a simple heuristic.
    # For C4-free colorings, a "concentric" or "layered" assignment often works.
    # Let's stick to lexicographic but with a strong propagation.
    
    # Actually, dynamic MRV is crucial for CSPs.
    # To make it fast, we only update MRV for edges affected by the last assignment.
    
    # Let's try a simpler approach first: Lexicographic order with backtracking.
    # If it's too slow, we'll switch to MRV.
    
    # Given n=26, m=325.
    # Branching factor 5. Depth 325.
    # 5^325 is huge. Pruning is essential.
    
    # The propagation logic:
    # When assigning (u, v) = c:
    # 1. Check if common_counts[c][u][v] >= 2. If yes, fail.
    # 2. For each w in [0, n):
    #    if color[w][v] == c:
    #       common_counts[c][u][w] += 1
    #       if common_counts[c][u][w] >= 2: fail
    #    if color[w][u] == c:
    #       common_counts[c][v][w] += 1
    #       if common_counts[c][v][w] >= 2: fail
    # 3. Update color[u][v] = c, color[v][u] = c
    
    # Undo logic is symmetric.
    
    # To improve performance, we can order edges by "constraint tightness".
    # Edges connecting vertices with many assigned neighbors are more constrained.
    
    # Let's implement a recursive solver with dynamic edge selection.
    
    assigned_count = 0
    
    # To avoid recursion limit issues and overhead, we can use an iterative stack, 
    # but recursion is fine for depth ~325 if optimized.
    
    # We need a list of unassigned edges.
    # Instead of maintaining a list and removing, we can just iterate through all edges 
    # and pick the first unassigned one with min domain?
    # That's O(m) per step. 325 * 325 = 100k ops per level? No, 325 steps * 325 edges = 100k checks total per path?
    # 5^d is the tree size.
    
    # Let's try to be smarter.
    # We'll maintain a list of unassigned edges.
    unassigned_edges = edges[:]
    
    # We'll use a stack for backtracking
    # Each state: (edge_index_in_unassigned, color_tried, ...)
    # This is complex.
    
    # Alternative: Use a SAT solver style approach? No, standard library only.
    
    # Let's stick to simple backtracking with lexicographic order first, 
    # but add a time check. If it's too slow, we might need MRV.
    
    # Actually, for n=26, lexicographic backtracking might be too slow in Python.
    # We need MRV.
    
    # Implementation of MRV:
    # For each unassigned edge (u, v), count how many colors c are valid.
    # A color c is invalid if:
    #   common_counts[c][u][v] >= 2
    #   OR there exists w such that color[w][v] == c and common_counts[c][u][w] >= 1 (would become 2)
    #   OR there exists w such that color[w][u] == c and common_counts[c][v][w] >= 1 (would become 2)
    
    # This check is O(n) per color per edge.
    # 5 colors * 325 edges * 26 n = ~40,000 operations per step to find the best edge.
    # 325 steps * 40,000 = 13 million operations per path.
    # If we explore 1000 paths, 13 billion ops. Too slow for 100s.
    
    # We need to optimize the check.
    # We can precompute for each (u, v) and color c, the "threat" count.
    # threat[c][u][v] = 1 if assigning c to (u, v) would create a C4, 0 otherwise.
    # Updating threat is O(n) per assignment.
    
    # Let's implement this optimized version.
    
    # threat[c][u][v] is 1 if:
    # 1. common_counts[c][u][v] >= 2
    # 2. There exists w such that color[w][v] == c and common_counts[c][u][w] >= 1
    # 3. There exists w such that color[w][u] == c and common_counts[c][v][w] >= 1
    
    # Actually, condition 2 and 3 are the propagation conditions.
    # If we assign c to (u, v), we increment common_counts[c][u][w] for all w where color[w][v]==c.
    # If any of those becomes 2, it's a conflict.
    # So, before assigning, we can check if any such w exists with common_counts[c][u][w] >= 1.
    
    # Let's define `valid_colors(u, v)`:
    # Returns a bitmask of valid colors.
    
    def get_valid_colors(u, v):
        valid = 0
        for c in range(5):
            # Check 1: Direct C4 on (u, v)
            if common_counts[c][u][v] >= 2:
                continue
            
            # Check 2: Indirect C4 via w
            conflict = False
            # w is neighbor of v in color c
            # We need to iterate w. To speed this up, we can maintain lists of neighbors.
            # But iterating 0..n is only 26 iterations. 5 * 26 = 130 ops per edge check.
            # 325 edges * 130 = 42k ops to find MRV.
            # This is acceptable.
            
            for w in range(n):
                if w == u or w == v: continue
                if color[w][v] == c:
                    if common_counts[c][u][w] >= 1:
                        conflict = True
                        break
                if color[w][u] == c:
                    if common_counts[c][v][w] >= 1:
                        conflict = True
                        break
            if not conflict:
                valid |= (1 << c)
        return valid
        
    # We will pick the edge with the minimum number of valid colors (MRV).
    # If min is 0, backtrack.
    
    # To make backtracking efficient, we store the state changes.
    
    # Since Python recursion is slow, and we might go deep, we should be careful.
    # However, n=26 depth is 325. Recursion limit is usually 1000. So it fits.
    
    # We'll use a recursive function.
    
    # To handle the time limit, we'll check time at the start of each call.
    
    # Optimization: If n is small, just return the known solution or use a faster method.
    # But we want to find the max n.
    
    # Let's add a "random restart" if we get stuck?
    # No, CSP backtracking is deterministic. If it doesn't find a solution, there might not be one 
    # (or the heuristic is bad). 
    # For n=26, a solution exists. A good MRV heuristic should find it.
    
    # One issue: Symmetry.
    # We can break symmetry by fixing the color of edge (0, 1) to 0.
    # And maybe (0, 2) to 0 or 1?
    # This reduces the search space by factor of 5.
    
    # Let's implement the solver.
    
    # State:
    # color: 2D list
    # common_counts: 3D list
    # unassigned_edges: list of (u, v)
    
    # We'll manage unassigned_edges as a list, and pop/append during backtracking.
    
    # Actually, modifying a list during recursion is tricky.
    # Better: Pass the list of unassigned edges.
    # But copying the list is O(m).
    
    # Alternative: Use a boolean array `assigned` and iterate through all edges to find the best one.
    # This avoids list management overhead.
    
    assigned = [[False] * n for _ in range(n)]
    
    def solve(idx):
        # idx is not used if we scan all edges, but we can use it as a hint?
        # No, we scan all edges to find MRV.
        
        if time.time() > deadline:
            return False, None
            
        # Find edge with min valid colors
        best_u = -1
        best_v = -1
        best_valid = 0
        min_count = 6 # More than 5
        
        # Iterate through all edges
        for i in range(n):
            for j in range(i + 1, n):
                if assigned[i][j]:
                    continue
                
                valid_mask = get_valid_colors(i, j)
                count = bin(valid_mask).count('1')
                
                if count < min_count:
                    min_count = count
                    best_u = i
                    best_v = j
                    best_valid = valid_mask
                    
                    if count == 0:
                        return False, None
                    if count == 1:
                        # Can't get better than 1 (unless 0)
                        # We can break early if we find a 0, but 1 is good.
                        # However, there might be a 0 later.
                        # So we must check all?
                        # Yes, if we find a 0, we fail.
                        # If we find a 1, we keep looking for 0.
                        pass
        
        if best_u == -1:
            # All edges assigned
            return True, [color[i][j] for i in range(n) for j in range(i + 1, n)]
            
        # Try colors for best_u, best_v
        u = best_u
        v = best_v
        
        # Order colors: try those that cause least propagation?
        # For now, try 0..4
        
        for c in range(5):
            if not (best_valid & (1 << c)):
                continue
                
            # Assign (u, v) = c
            color[u][v] = c
            color[v][u] = c
            assigned[u][v] = True
            assigned[v][u] = True
            
            # Propagate
            valid = True
            changes = [] # List of (u, v, c, old_count) to undo
            
            # Update common_counts and check for conflicts
            # 1. Check (u, v) itself
            if common_counts[c][u][v] >= 2:
                valid = False
            else:
                # 2. Update for w where color[w][v] == c
                for w in range(n):
                    if w == u or w == v: continue
                    if color[w][v] == c:
                        old_cnt = common_counts[c][u][w]
                        common_counts[c][u][w] += 1
                        common_counts[c][w][u] += 1
                        changes.append((u, w, c, old_cnt))
                        if common_counts[c][u][w] >= 2:
                            valid = False
                            break
                            
                if valid:
                    # 3. Update for w where color[w][u] == c
                    for w in range(n):
                        if w == u or w == v: continue
                        if color[w][u] == c:
                            old_cnt = common_counts[c][v][w]
                            common_counts[c][v][w] += 1
                            common_counts[c][w][v] += 1
                            changes.append((v, w, c, old_cnt))
                            if common_counts[c][v][w] >= 2:
                                valid = False
                                break
                                
            if valid:
                # Recurse
                success, res = solve(idx + 1)
                if success:
                    return True, res
            else:
                # Undo changes
                for (cu, cv, cc, old_cnt) in changes:
                    common_counts[cc][cu][cv] = old_cnt
                    common_counts[cc][cv][cu] = old_cnt
                    
            # Undo assignment
            color[u][v] = -1
            color[v][u] = -1
            assigned[u][v] = False
            assigned[v][u] = False
            
        return False, None
        
    # Symmetry breaking: Fix color(0, 1) = 0
    # This is safe because colors are symmetric.
    if n >= 2:
        color[0][1] = 0
        color[1][0] = 0
        assigned[0][1] = True
        assigned[1][0] = True
        
        # Propagate this initial assignment
        # common_counts[0][0][1] is 0.
        # For w where color[w][1] == 0: none (only 0,1 assigned).
        # For w where color[w][0] == 0: none.
        # So no changes.
        
    success, colors = solve(0)
    
    if success:
        return True, colors
    return False, []

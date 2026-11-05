"""Rewritten program for R_4(C6) maximization with a deadline-guarded search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _has_c6_in_color(edges_color, n, color):
    """Check if the subgraph of edges with given color contains a C6 (not necessarily induced).
    edges_color is a set of frozenset or tuple (i,j) with i<j.
    """
    # Build adjacency for this color
    adj = [set() for _ in range(n)]
    for (i, j) in edges_color:
        adj[i].add(j)
        adj[j].add(i)
    
    # Look for a 6-cycle: v0-v1-v2-v3-v4-v5-v0
    # We can do a bounded search: pick v0, then try paths of length 5 that return to v0
    # To avoid duplicates, fix v0 < v1 < ... but for detection we just need any.
    # For small n, we can enumerate.
    # A faster way: for each pair (v0, v1), try to extend.
    # But let's use a simple backtracking with pruning for C6.
    
    # Actually, for n up to ~20, checking all 6-cycles is feasible if we're smart.
    # Number of 6-cycles in K_n is n!/(2*6) which is huge for n=20.
    # We need a smarter check.
    
    # Use the fact that a C6 exists iff there's a path of length 5 from v0 to v0 (cycle).
    # Let's use DFS with visited set for each starting node, limiting depth to 5.
    for v0 in range(n):
        # Try all paths of length 5 starting at v0, ending at a neighbor of v0
        # Path: v0 -> v1 -> v2 -> v3 -> v4 -> v5 -> v0
        # v1, v2, v3, v4, v5 must be distinct and not v0 (except the last step back)
        # To reduce symmetry, fix v1 > v0? No, we just need existence.
        
        # DFS
        def dfs(path, visited):
            if len(path) == 6:
                # Check if last node connects to first
                if path[0] in adj[path[5]]:
                    return True
                return False
            for nxt in adj[path[-1]]:
                if nxt not in visited and nxt != path[0]:
                    visited.add(nxt)
                    path.append(nxt)
                    if dfs(path, visited):
                        return True
                    path.pop()
                    visited.remove(nxt)
            return False
        
        # To speed up, we can fix the second node to be greater than v0 to avoid
        # checking the same cycle twice, but it's not strictly necessary for correctness.
        # Let's just do it for all neighbors.
        for v1 in adj[v0]:
            if dfs([v0, v1], {v0, v1}):
                return True
    return False

def _is_valid_coloring(colors, n):
    """Check if the coloring has no monochromatic C6."""
    edges_by_color = {c: [] for c in range(4)}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            edges_by_color[c].append((i, j))
            idx += 1
    for c in range(4):
        if _has_c6_in_color(edges_by_color[c], n, c):
            return False
    return True

def _try_extend(current_colors, n_current, target_n, deadline):
    """Try to extend a valid coloring of n_current vertices to target_n vertices.
    Returns (success, new_colors) or (False, None).
    """
    if n_current >= target_n:
        return True, current_colors
    
    # We need to add (target_n - n_current) vertices.
    # This is a complex constraint satisfaction problem.
    # For now, let's try a simple greedy or random approach for small extensions.
    # This is a placeholder for a more robust solver.
    
    # A better approach: use backtracking with the new vertices.
    # Let's implement a simple backtracking for adding one vertex at a time.
    
    colors = list(current_colors)
    n = n_current
    
    for _ in range(target_n - n_current):
        new_vertex = n
        n += 1
        # We need to assign colors to edges (i, new_vertex) for i in 0..n-2
        # There are n-1 edges to assign.
        
        # Backtracking search
        def bt(edges_to_assign, assigned_colors):
            if time.time() > deadline:
                return None
            
            if not edges_to_assign:
                # Check if the new coloring is valid
                # We only need to check for new C6s that involve the new vertex
                # But for simplicity, check all (expensive but correct)
                # Actually, we can incrementally check.
                # For now, let's do a full check for correctness, but it's slow.
                # Let's optimize: only check cycles that include the new vertex.
                # A C6 including the new vertex must use at least one edge to the new vertex.
                # Let's just do a full check for now, as n is small.
                all_colors = assigned_colors + colors
                if _is_valid_coloring(all_colors, n):
                    return assigned_colors
                return None
            
            i = edges_to_assign[0]
            remaining = edges_to_assign[1:]
            
            for c in range(4):
                # Assign color c to edge (i, new_vertex)
                # Check if this creates a C6 in color c
                # We can do a quick check: does adding this edge create a C6?
                # This is complex. Let's just try all and check at the end.
                # To prune, we can check if the partial assignment is already invalid.
                # This is hard. Let's just do full search.
                
                new_assigned = assigned_colors + [c]
                result = bt(remaining, new_assigned)
                if result is not None:
                    return result
            return None
        
        edges_to_assign = list(range(n - 1))  # vertices 0..n-2
        result = bt(edges_to_assign, [])
        if result is None:
            return False, None
        colors.extend(result)
    
    return True, colors

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with the known K9 coloring
    n_start = 9
    colors_start = [min(j - i, n_start - j + i) - 1 for i in range(n_start) for j in range(i + 1, n_start)]
    
    best_n = n_start
    best_colors = colors_start
    
    # Try to extend to larger n
    for target_n in range(n_start + 1, 21):
        if time.time() > deadline:
            break
        success, new_colors = _try_extend(best_colors, best_n, target_n, deadline)
        if success:
            best_n = target_n
            best_colors = new_colors
        else:
            # Try a different starting point or random restarts?
            # For now, if we can't extend, try the next target with a different strategy
            # or give up on this lineage.
            pass
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END

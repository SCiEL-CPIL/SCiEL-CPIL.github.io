"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _mul(a, b):
    r = 0
    for i in range(4):
        if b >> i & 1:
            r ^= a << i
    for i in range(7, 3, -1):
        if r >> i & 1:
            r ^= 0b10011 << (i - 4)
    return r

def _build_k32_colors():
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = _mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return colors

def _build_adjacency(colors, n):
    """Build adjacency lists for each color class."""
    adj = {0: [set() for _ in range(n)], 1: [set() for _ in range(n)], 
           2: [set() for _ in range(n)], 3: [set() for _ in range(n)]}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
    return adj

def _check_no_triangle_adj(adj, n):
    """Check that no color class contains a triangle."""
    for c in range(4):
        for i in range(n):
            neighbors = adj[c][i]
            for j in neighbors:
                if j > i:
                    for k in neighbors:
                        if k > j:
                            if k in adj[c][j]:
                                return False
    return True

def _count_triangles_in_color(adj, c, n):
    """Count triangles in color class c."""
    count = 0
    for i in range(n):
        neighbors = adj[c][i]
        for j in neighbors:
            if j > i:
                for k in neighbors:
                    if k > j and k in adj[c][j]:
                        count += 1
    return count

def _try_extend(colors, n, max_n, deadline, rng):
    """Try to extend coloring from n to max_n vertices via local search."""
    # Start with the base coloring
    cur_colors = colors[:]
    cur_n = n
    
    # Build adjacency for current coloring
    adj = _build_adjacency(cur_colors, cur_n)
    
    # Try to add vertices one at a time
    for target_n in range(cur_n + 1, max_n + 1):
        if time.time() > deadline:
            break
        
        # Try to add vertex target_n-1
        # We need to assign colors to edges from new vertex to all existing vertices
        # such that no monochromatic triangle is formed
        
        # For each existing vertex i, the new edge (i, target_n-1) can be any color
        # such that it doesn't form a triangle with any pair (i, j) where j < target_n-1
        # and edge (i,j) has the same color
        
        # Strategy: try all possible color assignments using backtracking with time limit
        new_vertex = target_n - 1
        assignments = [None] * new_vertex  # assignments[i] = color of edge (i, new_vertex)
        
        # Precompute: for each vertex i and color c, check if adding edge (i, new_vertex) 
        # with color c would create a triangle
        # A triangle is formed if there exists j < i such that:
        #   edge(i,j) has color c AND edge(j, new_vertex) has color c
        # But since we're assigning colors for the new vertex, we need to check:
        #   For color c on edge (i, new_vertex), we need no j < i such that
        #   edge(i,j) has color c AND we assign color c to edge (j, new_vertex)
        
        # Use a simpler approach: try random assignments and check
        best_assignment = None
        attempts = 0
        max_attempts = 5000
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            # Random assignment
            assignment = [rng.randint(0, 3) for _ in range(new_vertex)]
            
            # Check if this assignment is valid
            valid = True
            for i in range(new_vertex):
                ci = assignment[i]
                # Check all j < i
                for j in range(i):
                    cj = assignment[j]
                    if ci == cj:
                        # Check if edge (i, j) has color ci
                        # Find index of edge (i, j) in cur_colors
                        # Edge (i,j) where i > j: index = sum_{k=0}^{i-1} (n-1-k) + (j)
                        # Actually, let's use the adjacency structure
                        if j in adj[ci][i]:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                best_assignment = assignment
                break
        
        if best_assignment is None:
            # Try a more systematic approach: greedy with backtracking
            # Assign colors one by one, checking constraints
            assignment = [None] * new_vertex
            success = True
            
            for i in range(new_vertex):
                # Determine which colors are forbidden for edge (i, new_vertex)
                forbidden = [False, False, False, False]
                for j in range(i):
                    if assignment[j] is not None:
                        cj = assignment[j]
                        if j in adj[cj][i]:
                            forbidden[cj] = True
                
                # Try each color
                found = False
                for c in range(4):
                    if not forbidden[c]:
                        # Check if this creates issues with future vertices
                        # For now, just accept if not forbidden
                        assignment[i] = c
                        found = True
                        break
                
                if not found:
                    success = False
                    break
            
            if success:
                best_assignment = assignment
        
        if best_assignment is not None:
            # Add the new vertex's edges
            cur_colors.extend(best_assignment)
            cur_n = target_n
            
            # Rebuild adjacency
            adj = _build_adjacency(cur_colors, cur_n)
            
            # Verify no triangles
            if not _check_no_triangle_adj(adj, cur_n):
                # This shouldn't happen if our check was correct, but rollback
                cur_colors = cur_colors[:len(cur_colors) - new_vertex]
                cur_n = cur_n - 1
                break
        else:
            # Cannot extend further with this approach
            break
    
    return cur_colors, cur_n

def construct():
    deadline = time.time() + 140  # Safety margin: 10s before 150s kill
    rng = random.Random(42)
    
    # Start with 32-vertex coloring
    base_colors = _build_k32_colors()
    best_colors = base_colors[:]
    best_n = 32
    
    # Try to extend to larger n
    try:
        extended_colors, extended_n = _try_extend(base_colors, 32, 62, deadline, rng)
        if extended_n > best_n:
            best_colors = extended_colors
            best_n = extended_n
    except Exception:
        pass
    
    # If we have time, try local search to improve
    if time.time() < deadline - 5:
        # Try random restarts with different seeds
        for seed in range(1, 10):
            if time.time() > deadline - 2:
                break
            rng2 = random.Random(seed)
            try:
                ext_colors, ext_n = _try_extend(base_colors, 32, 62, deadline, rng2)
                if ext_n > best_n:
                    best_colors = ext_colors
                    best_n = ext_n
            except Exception:
                pass
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END

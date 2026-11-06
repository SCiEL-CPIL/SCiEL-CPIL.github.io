"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

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

def _build_edge_color_matrix(colors, n):
    """Build an n x n matrix where mat[i][j] = color of edge (i,j)."""
    mat = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            mat[i][j] = c
            mat[j][i] = c
            idx += 1
    return mat

def _check_no_triangle_mat(mat, n):
    """Check that no color class contains a triangle using the matrix."""
    for c in range(4):
        for i in range(n):
            for j in range(i + 1, n):
                if mat[i][j] != c:
                    continue
                for k in range(j + 1, n):
                    if mat[i][k] == c and mat[j][k] == c:
                        return False
    return True

def _backtrack_add_vertex(mat, cur_n, deadline):
    """
    Use backtracking to assign colors to edges from new vertex (cur_n) to vertices 0..cur_n-1.
    Returns the assignment list (colors for edges (i, cur_n) for i in 0..cur_n-1) or None.
    """
    new_v = cur_n
    assignment = [0] * new_v  # assignment[i] = color of edge (i, new_v)
    
    # For backtracking efficiency, process vertices in order 0, 1, ..., new_v-1
    # When assigning color to edge (i, new_v), we need to check:
    #   For all j < i: if assignment[j] == c and mat[i][j] == c, then triangle (j, i, new_v)
    # So for vertex i, color c is forbidden if exists j < i with assignment[j] == c and mat[i][j] == c
    
    def bt(i):
        if time.time() > deadline:
            return False
        if i == new_v:
            return True
        # Try colors 0,1,2,3 for edge (i, new_v)
        for c in range(4):
            # Check if color c is valid for edge (i, new_v)
            valid = True
            for j in range(i):
                if assignment[j] == c and mat[i][j] == c:
                    valid = False
                    break
            if valid:
                assignment[i] = c
                if bt(i + 1):
                    return True
        return False
    
    if bt(0):
        return assignment
    return None

def _try_extend(colors, n, max_n, deadline):
    """Try to extend coloring from n to max_n vertices via backtracking."""
    cur_colors = colors[:]
    cur_n = n
    mat = _build_edge_color_matrix(cur_colors, cur_n)
    
    for target_n in range(cur_n + 1, max_n + 1):
        if time.time() > deadline:
            break
        
        new_v = cur_n  # The new vertex index
        assignment = _backtrack_add_vertex(mat, cur_n, deadline)
        
        if assignment is not None:
            # Add the new vertex's edges to cur_colors in lexicographic order
            # Edges are (i, new_v) for i in 0..new_v-1, which come after all existing edges
            cur_colors.extend(assignment)
            cur_n = target_n
            # Extend matrix
            new_row = [-1] * cur_n
            for i in range(new_v):
                new_row[i] = assignment[i]
                new_row[new_v] = -1  # self
            # Add column
            for i in range(new_v):
                mat[i].append(assignment[i])
            mat.append(new_row)
        else:
            break
    
    return cur_colors, cur_n

def _local_search_polish(colors, n, deadline):
    """
    Local search: try to flip colors of edges to allow adding more vertices.
    This is a heuristic approach - we try random edge flips and see if we can then extend.
    """
    if n < 2:
        return colors, n
    
    mat = _build_edge_color_matrix(colors, n)
    best_colors = colors[:]
    best_n = n
    
    # We'll try to extend from current n up to max_n using backtracking
    # If backtracking fails, we might try some local modifications
    # For now, just use the backtracking extension
    extended_colors, extended_n = _try_extend(colors, n, min(62, n + 20), deadline)
    
    if extended_n > best_n:
        best_colors = extended_colors
        best_n = extended_n
    
    return best_colors, best_n

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin: 10s before 150s kill
    
    # Start with 32-vertex coloring
    base_colors = _build_k32_colors()
    best_colors = base_colors[:]
    best_n = 32
    
    # Try to extend to larger n using backtracking
    try:
        extended_colors, extended_n = _try_extend(base_colors, 32, 62, deadline)
        if extended_n > best_n:
            best_colors = extended_colors
            best_n = extended_n
    except Exception:
        pass
    
    # If we have time, try multiple random restarts of the extension process
    # by slightly modifying the base coloring
    if time.time() < deadline - 10:
        rng = random.Random(42)
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            # Try a slightly different approach: modify some edges in the base
            # and see if we can extend further
            mod_colors = base_colors[:]
            # Flip a few random edges
            num_flips = rng.randint(1, 5)
            for _ in range(num_flips):
                idx = rng.randint(0, len(mod_colors) - 1)
                mod_colors[idx] = (mod_colors[idx] + 1) % 4
            
            # Check if modified coloring is still valid
            mat = _build_edge_color_matrix(mod_colors, 32)
            if not _check_no_triangle_mat(mat, 32):
                continue
            
            try:
                ext_colors, ext_n = _try_extend(mod_colors, 32, 62, deadline)
                if ext_n > best_n:
                    best_colors = ext_colors
                    best_n = ext_n
            except Exception:
                continue
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END

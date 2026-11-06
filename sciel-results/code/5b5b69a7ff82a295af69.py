"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Greenwood-Gleason colouring of K16 by cubic-residue classes of GF(16)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    # Build 16-vertex 3-colouring (no monochromatic triangles)
    def color16(i, j):
        return log[i ^ j] % 3
    
    # Try to extend: start with 32 vertices (two copies of 16, cross edges = colour 3)
    # Then attempt to add more vertices via local search
    
    def make_colors(n, base_n=16):
        """Build colouring for n vertices using copies of the 16-vertex graph"""
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                ci = i // base_n
                cj = j // base_n
                pi = i % base_n
                pj = j % base_n
                if ci == cj:
                    colors.append(color16(pi, pj))
                else:
                    # Cross edges: use colour 3 for all cross edges (bipartite)
                    colors.append(3)
        return colors
    
    def has_mono_triangle(n, colors):
        """Check if any colour class contains a triangle"""
        # Build adjacency per colour
        adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i, j] = True
                adj[c][j, i] = True
                idx += 1
        for c in range(4):
            A = adj[c]
            # Triangle exists if there is (i,j,k) with all three edges present
            # Efficient check: for each i, look at neighbors of i, check if any pair are connected
            for i in range(n):
                nbrs = np.where(A[i, :])[0]
                nbrs = nbrs[nbrs > i]  # only j > i
                for j in nbrs:
                    # Check if k > j has edges to both i and j
                    common = A[j, j+1:] & A[i, j+1:]
                    if np.any(common):
                        return True
        return False
    
    # Start with 32 vertices
    best_n = 32
    best_colors = make_colors(32)
    
    # Verify
    if has_mono_triangle(32, best_colors):
        # Should not happen, but fallback to 16
        best_n = 16
        best_colors = make_colors(16)
    
    # Try to extend by adding one vertex at a time
    # For each new vertex, try all 4^16 assignments for edges to existing vertices
    # and check if no monochromatic triangle is formed
    
    def try_extend(n, colors):
        """Try to add one vertex to the n-vertex colouring"""
        if time.time() > deadline - 5.0:
            return None
        # Number of edges from new vertex to existing vertices: n
        # Each edge gets a colour in {0,1,2,3}
        # Constraint: for any pair (i,j) among existing vertices, if color(i,j) = c,
        # then new vertex cannot have both edges to i and j coloured c (would form mono triangle)
        
        # Build constraint matrix: for each colour c and each edge (i,j) of colour c,
        # the new vertex cannot assign colour c to both i and j.
        
        # Use backtracking with pruning
        new_colors = [None] * n  # colour of edge (new, v) for v in 0..n-1
        
        # Precompute: for each colour c, list of pairs (i,j) where color(i,j) = c
        pairs_by_color = [[] for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                pairs_by_color[c].append((i, j))
                idx += 1
        
        # Backtracking search
        def backtrack(v):
            if v == n:
                return True
            for c in range(4):
                # Check if assigning colour c to edge (new, v) creates a mono triangle
                # Need: no pair (i,j) with i,j < v (since we assign in order) such that
                # color(i,j) = c AND new_colors[i] = c AND new_colors[j] = c
                valid = True
                for (i, j) in pairs_by_color[c]:
                    if j < v:  # both i and j already assigned
                        if new_colors[i] == c and new_colors[j] == c:
                            valid = False
                            break
                    elif i < v and j >= v:
                        # j is v or greater; if j == v, not yet assigned; if j > v, not yet assigned
                        pass
                    elif i >= v:
                        # both not yet assigned, skip
                        pass
                if not valid:
                    continue
                new_colors[v] = c
                if backtrack(v + 1):
                    return True
                new_colors[v] = None
            return False
        
        if backtrack(0):
            return new_colors
        return None
    
    # Try to extend from 32 to 33, 34, ...
    for target_n in range(33, 40):  # Try up to 39
        if time.time() > deadline - 10.0:
            break
        if target_n > best_n + 1:
            break
        # Extend best_n vertex colouring to target_n
        if target_n == best_n + 1:
            new_vc = try_extend(best_n, best_colors)
            if new_vc is not None:
                # Build new colouring
                new_colors = list(best_colors)
                for v in range(best_n):
                    new_colors.append(new_vc[v])
                # Verify
                if not has_mono_triangle(target_n, new_colors):
                    best_n = target_n
                    best_colors = new_colors
    
    # Also try a different approach: use 3 copies with smarter cross-colouring
    # Actually, let's try the Paley-based approach for larger n
    # For now, return what we have
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END

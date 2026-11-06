"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _has_c6_in_color(n, colors, color):
    """Check if a specific color class contains a C6."""
    # Build adjacency matrix for this color
    adj = np.zeros((n, n), dtype=bool)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == color:
                adj[i, j] = adj[j, i] = True
            idx += 1
    
    # Check for C6 using DFS from each vertex
    for start in range(n):
        # Find all 6-cycles starting from 'start'
        # A C6 is: start -> v1 -> v2 -> v3 -> v4 -> v5 -> start
        # where all edges exist and all vertices distinct
        # Use backtracking
        def backtrack(path, visited):
            if len(path) == 6:
                # Check if edge back to start exists
                if adj[path[-1], start]:
                    return True
                return False
            for v in range(n):
                if not visited[v] and adj[path[-1], v]:
                    if v == start and len(path) < 5:
                        continue  # Can't return to start before completing cycle
                    visited[v] = True
                    path.append(v)
                    if backtrack(path, visited):
                        return True
                    path.pop()
                    visited[v] = False
            return False
        
        visited = [False] * n
        visited[start] = True
        if backtrack([start], visited):
            return True
    return False

def _is_valid(n, colors):
    """Check if the coloring is valid (no C6 in any color class)."""
    for c in range(4):
        if _has_c6_in_color(n, colors, c):
            return False
    return True

def _spectral_coloring(n, seed=42):
    """Generate a 4-coloring using spectral methods."""
    rng = np.random.default_rng(seed)
    
    # Create a random symmetric matrix with zero diagonal
    M = rng.random((n, n))
    M = (M + M.T) / 2
    np.fill_diagonal(M, 0)
    
    # Get eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(M)
    except:
        return None
    
    # Use top 4 eigenvectors (most negative eigenvalues for eigh, so last 4)
    # Actually, for random matrices, we want the eigenvectors with largest absolute eigenvalues
    # or we can just use 4 random linear combinations
    # Let's use the first 4 eigenvectors corresponding to largest |eigenvalues|
    abs_eigs = np.abs(eigenvalues)
    top4_idx = np.argsort(abs_eigs)[-4:]
    V = eigenvectors[:, top4_idx]
    
    # Assign colors based on quadrant of 4D vector
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            # Use difference of vertex embeddings
            diff = V[i] - V[j]
            # Map to 4 quadrants based on signs of first two components
            # Actually, let's use a more robust mapping
            # Color based on which octant/region in 4D space
            c = 0
            if diff[0] >= 0:
                c += 1
            if diff[1] >= 0:
                c += 2
            # This gives 4 colors (0,1,2,3) based on signs of two components
            colors.append(c)
    
    return colors

def _random_coloring(n, seed=0):
    """Generate a random 4-coloring."""
    rng = np.random.default_rng(seed)
    m = n * (n - 1) // 2
    return list(rng.integers(0, 4, size=m))

def _greedy_improve(n, colors, max_iter=1000, deadline=None):
    """Try to improve a coloring by local search."""
    colors = list(colors)
    best_colors = list(colors)
    best_valid = _is_valid(n, best_colors)
    
    if best_valid:
        return best_colors
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Pick a random edge and try changing its color
        idx = np.random.randint(0, len(colors))
        old_c = colors[idx]
        new_c = (old_c + 1 + np.random.randint(0, 3)) % 4
        
        colors[idx] = new_c
        if _is_valid(n, colors):
            best_colors = list(colors)
            best_valid = True
            return best_colors
        else:
            colors[idx] = old_c
    
    return best_colors

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Start with known valid n=9 coloring
    n_base = 9
    colors_base = [min(j - i, n_base - j + i) - 1 for i in range(n_base) for j in range(i + 1, n_base)]
    best_result = {'n': n_base, 'colors': colors_base}
    
    # Try to find larger n using spectral approach
    for target_n in range(10, 18):
        if time.time() > deadline:
            break
        
        # Try multiple seeds for spectral coloring
        found = False
        for seed in range(20):
            if time.time() > deadline:
                break
            
            colors = _spectral_coloring(target_n, seed=seed)
            if colors is None:
                continue
            
            if _is_valid(target_n, colors):
                best_result = {'n': target_n, 'colors': colors}
                found = True
                break
        
        if found:
            # Try to go even larger
            continue
        else:
            # If we can't find valid coloring for this n, try smaller
            # But first try random + greedy
            for seed in range(10):
                if time.time() > deadline:
                    break
                colors = _random_coloring(target_n, seed=seed)
                if _is_valid(target_n, colors):
                    best_result = {'n': target_n, 'colors': colors}
                    found = True
                    break
            
            if not found:
                break  # Can't find valid coloring for this n, stop
    
    # If we found a valid coloring, try to extend it
    # Try adding vertices one at a time
    current_n = best_result['n']
    current_colors = list(best_result['colors'])
    
    for new_n in range(current_n + 1, 21):
        if time.time() > deadline:
            break
        
        # Try to extend by adding one vertex
        # We need to add n_new - n_old edges from new vertex to old vertices
        # Plus edges among new vertices if adding multiple
        # For simplicity, add one vertex at a time
        
        new_colors_found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Randomly assign colors to new edges
            rng = np.random.default_rng(attempt * 1000 + new_n)
            new_edges = list(rng.integers(0, 4, size=current_n))
            
            candidate_colors = current_colors + new_edges
            
            if _is_valid(new_n, candidate_colors):
                best_result = {'n': new_n, 'colors': candidate_colors}
                current_n = new_n
                current_colors = candidate_colors
                new_colors_found = True
                break
        
        if not new_colors_found:
            break
    
    return best_result
# EVOLVE-BLOCK-END

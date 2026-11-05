"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _has_mono_k222(adj, n):
    """Check if adjacency matrix adj (n x n boolean) contains K2,2,2.
    Returns True if K2,2,2 exists (bad), False if not (good).
    Uses the fact that K2,2,2 exists iff there are 3 disjoint pairs with complete
    bipartite connections between them.
    """
    # For each pair (i,j), define common neighborhood: vertices k where adj[i,k] and adj[j,k]
    # K2,2,2: pairs (a,b), (c,d), (e,f) with all cross-edges present.
    # Equivalent: (a,b) is an edge, (c,d) is an edge in N(a,b), (e,f) is an edge in N(a,b) ∩ N(c,d)
    
    # Precompute for each edge (i,j), the set of common neighbors as a bitmask or boolean array
    # For n up to 30, use boolean arrays
    
    # adj[i,j] is True if edge (i,j) exists in this color
    
    # For each pair (i,j) with adj[i,j]:
    #   CN(i,j) = {k : k != i, k != j, adj[i,k] and adj[j,k]}
    #   For each pair (c,d) in CN(i,j) with adj[c,d]:
    #     CN2 = CN(i,j) ∩ CN(c,d)
    #     If there exists a pair (e,f) in CN2 with adj[e,f]:
    #       Then (a,b), (c,d), (e,f) form K2,2,2
    
    # This is still potentially expensive but let's try with numpy
    
    # Precompute common neighbors for all pairs
    # CN[i,j] = adj[i,:] & adj[j,:] (excluding i and j themselves)
    
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i, j]:
                continue
            # Common neighbors of i and j
            cn = adj[i, :] & adj[j, :]
            cn[i] = False
            cn[j] = False
            
            # Find all pairs (c,d) in cn with adj[c,d]
            cn_indices = np.where(cn)[0]
            if len(cn_indices) < 4:
                continue
            
            for idx_c in range(len(cn_indices)):
                c = cn_indices[idx_c]
                for idx_d in range(idx_c + 1, len(cn_indices)):
                    d = cn_indices[idx_d]
                    if not adj[c, d]:
                        continue
                    # Now check if there's a pair (e,f) in cn ∩ CN(c,d) with adj[e,f]
                    cn_cd = adj[c, :] & adj[d, :]
                    cn_cd[c] = False
                    cn_cd[d] = False
                    cn2 = cn & cn_cd
                    cn2_indices = np.where(cn2)[0]
                    if len(cn2_indices) < 2:
                        continue
                    for idx_e in range(len(cn2_indices)):
                        e = cn2_indices[idx_e]
                        for idx_f in range(idx_e + 1, len(cn2_indices)):
                            f = cn2_indices[idx_f]
                            if adj[e, f]:
                                return True
    return False

def _count_violations_fast(adj, n):
    """Count monochromatic K2,2,2 in adjacency matrix adj."""
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i, j]:
                continue
            cn = adj[i, :] & adj[j, :]
            cn[i] = False
            cn[j] = False
            cn_indices = np.where(cn)[0]
            if len(cn_indices) < 4:
                continue
            for idx_c in range(len(cn_indices)):
                c = cn_indices[idx_c]
                for idx_d in range(idx_c + 1, len(cn_indices)):
                    d = cn_indices[idx_d]
                    if not adj[c, d]:
                        continue
                    cn_cd = adj[c, :] & adj[d, :]
                    cn_cd[c] = False
                    cn_cd[d] = False
                    cn2 = cn & cn_cd
                    cn2_indices = np.where(cn2)[0]
                    if len(cn2_indices) < 2:
                        continue
                    for idx_e in range(len(cn2_indices)):
                        e = cn2_indices[idx_e]
                        for idx_f in range(idx_e + 1, len(cn2_indices)):
                            f = cn2_indices[idx_f]
                            if adj[e, f]:
                                count += 1
    return count

def _try_local_search(n, max_time=100.0, seed=42):
    start_time = time.time()
    deadline = start_time + max_time
    rng = np.random.default_rng(seed)
    
    colors = rng.integers(0, 2, size=(n, n)).astype(np.int8)
    for i in range(n):
        for j in range(i+1, n):
            colors[j, i] = colors[i, j]
    np.fill_diagonal(colors, 0)
    
    adj0 = (colors == 0)
    adj1 = (colors == 1)
    
    v0 = _count_violations_fast(adj0, n)
    v1 = _count_violations_fast(adj1, n)
    best_violations = v0 + v1
    best_colors = colors.copy()
    
    if best_violations == 0:
        return colors
    
    T = 1.0
    cooling = 0.9995
    
    while time.time() < deadline:
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        old = colors[i, j]
        colors[i, j] = 1 - old
        colors[j, i] = colors[i, j]
        
        adj0 = (colors == 0)
        adj1 = (colors == 1)
        v0 = _count_violations_fast(adj0, n)
        v1 = _count_violations_fast(adj1, n)
        new_violations = v0 + v1
        
        if new_violations < best_violations:
            best_violations = new_violations
            best_colors = colors.copy()
            if best_violations == 0:
                return colors
        else:
            delta = new_violations - best_violations
            if delta > 0 and rng.random() < np.exp(-delta / T):
                pass
            else:
                colors[i, j] = old
                colors[j, i] = old
        
        T *= cooling
        if T < 0.005:
            T = 0.005
    
    return best_colors

def _colors_from_matrix(color_matrix, n):
    result = []
    for i in range(n):
        for j in range(i+1, n):
            result.append(int(color_matrix[i, j]))
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    for n in [30, 29]:
        if time.time() > deadline:
            break
        for seed in [42, 123, 456, 789, 1000, 2024, 31415, 27182, 1, 2, 3]:
            if time.time() > deadline:
                break
            remaining = deadline - time.time()
            if remaining < 2:
                break
            colors = _try_local_search(n, max_time=min(10.0, remaining), seed=seed)
            adj0 = (colors == 0)
            adj1 = (colors == 1)
            if not _has_mono_k222(adj0, n) and not _has_mono_k222(adj1, n):
                return {'n': n, 'colors': _colors_from_matrix(colors, n)}
    
    # Fallback: try random colorings for smaller n
    for n in [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]:
        if time.time() > deadline:
            break
        for seed in range(200):
            if time.time() > deadline:
                break
            rng = np.random.default_rng(seed)
            colors = rng.integers(0, 2, size=(n, n)).astype(np.int8)
            for i in range(n):
                for j in range(i+1, n):
                    colors[j, i] = colors[i, j]
            np.fill_diagonal(colors, 0)
            adj0 = (colors == 0)
            adj1 = (colors == 1)
            if not _has_mono_k222(adj0, n) and not _has_mono_k222(adj1, n):
                return {'n': n, 'colors': _colors_from_matrix(colors, n)}
    
    # Last resort: Paley K17
    n = 17
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': n, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(n) for j in range(i + 1, n)]}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _build_adj_matrices(colors, n):
    """Build 4 numpy adjacency matrices (one per color) from the color list."""
    m = n * (n - 1) // 2
    adj = [np.zeros((n, n), dtype=np.int8) for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i, j] = 1
            adj[c][j, i] = 1
            idx += 1
    return adj

def _count_c6s_fast(adj_matrices, n):
    """Count C6s in each color class using numpy. Returns list of 4 counts."""
    counts = [0] * 4
    for c in range(4):
        A = adj_matrices[c]
        # A^2[i,j] = number of common neighbors of i and j
        A2 = A @ A
        # A^3[i,j] = number of walks of length 3 from i to j
        A3 = A2 @ A
        # A^4[i,j] = number of walks of length 4 from i to j
        A4 = A3 @ A
        # A^5[i,j] = number of walks of length 5 from i to j
        A5 = A4 @ A
        
        # For a C6 through vertices v0,v1,v2,v3,v4,v5 (in order),
        # we need edges (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v5),(v5,v0)
        # Count = sum over all 6-cycles. 
        # Each C6 is counted 12 times in the trace of A^6 (2 directions * 6 starting points)
        # But we want to count each distinct C6 once.
        # 
        # Alternative: count using the formula for number of 6-cycles.
        # Number of C6 = (1/12) * (trace(A^6) - 6*trace(A^3) - 3*sum_{i!=j}(A^2[i,j])^2 + ... )
        # This is complex. Let's use a simpler approach for small n.
        
        # For n <= 20, we can use a direct but optimized approach:
        # For each pair (i,j) with i<j, count paths of length 4 from i to j that don't use i or j internally,
        # then check if (i,j) is an edge.
        # Actually, let's just do the DFS but with numpy for speed where possible.
        
        # Better: use the fact that for small graphs, we can enumerate
        # Number of 6-cycles = sum_{i<j, A[i,j]=1} (number of paths of length 4 from i to j avoiding i,j)
        # But this is still O(n^2 * something).
        
        # For n <= 20, let's use a bitmask-based DFS which is fast in Python
        adj_list = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j]:
                    adj_list[i].append(j)
                    adj_list[j].append(i)
        
        count = 0
        for start in range(n):
            # DFS to find paths of length 5 starting from start, with all vertices > start
            def dfs(current, path, visited_mask):
                nonlocal count
                if len(path) == 6:
                    if A[path[0], path[5]]:
                        count += 1
                    return
                for nxt in adj_list[current]:
                    if nxt > start and not (visited_mask >> nxt) & 1:
                        path.append(nxt)
                        if dfs(nxt, path, visited_mask | (1 << nxt)):
                            return
                        path.pop()
            
            dfs(start, [start], 1 << start)
        counts[c] = count
    return counts

def _has_c6_fast(adj_matrices, n):
    """Check if any color class contains a C6. Returns True if found."""
    for c in range(4):
        A = adj_matrices[c]
        adj_list = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j]:
                    adj_list[i].append(j)
                    adj_list[j].append(i)
        
        for start in range(n):
            def dfs(current, path, visited_mask):
                if len(path) == 6:
                    return A[path[0], path[5]]
                for nxt in adj_list[current]:
                    if nxt > start and not (visited_mask >> nxt) & 1:
                        path.append(nxt)
                        if dfs(nxt, path, visited_mask | (1 << nxt)):
                            return True
                        path.pop()
                return False
            
            if dfs(start, [start], 1 << start):
                return True
    return False

def _local_search(initial_colors, n, deadline):
    """Perform local search to minimize C6 count using fast numpy-based checks."""
    colors = list(initial_colors)
    best_colors = list(colors)
    
    # Build initial adjacency matrices
    adj = _build_adj_matrices(colors, n)
    best_c6_count = sum(_count_c6s_fast(adj, n))
    
    if best_c6_count == 0:
        return colors
    
    m = n * (n - 1) // 2
    step = 0
    max_steps = 200000
    
    while step < max_steps:
        if time.time() > deadline:
            break
        
        # Pick a random edge to flip
        idx = random.randint(0, m - 1)
        old_color = colors[idx]
        new_color = random.choice([c for c in range(4) if c != old_color])
        
        # Find which pair (i,j) this index corresponds to
        # Index = i*(n-1) - i*(i-1)//2 + (j-i-1)
        # We need to invert this. Precompute a lookup.
        # For efficiency, precompute the mapping once.
        
        colors[idx] = new_color
        adj_new = _build_adj_matrices(colors, n)
        new_c6_count = sum(_count_c6s_fast(adj_new, n))
        
        if new_c6_count < best_c6_count:
            best_c6_count = new_c6_count
            best_colors = list(colors)
            adj = adj_new
            if best_c6_count == 0:
                return colors
        else:
            colors[idx] = old_color
        
        step += 1
    
    return best_colors

# Precompute index-to-(i,j) mapping for a given n
def _build_index_map(n):
    """Return a list where idx -> (i,j) for lexicographic ordering."""
    m = n * (n - 1) // 2
    mapping = [None] * m
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            mapping[idx] = (i, j)
            idx += 1
    return mapping

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (150s limit)
    
    # Start with a known good small solution
    n0 = 9
    colors0 = [min(j - i, n0 - j + i) - 1 for i in range(n0) for j in range(i + 1, n0)]
    
    best_n = n0
    best_colors = list(colors0)
    
    # Precompute index maps for each n we might use
    index_maps = {}
    for target_n in range(n0 + 1, 18):
        index_maps[target_n] = _build_index_map(target_n)
    
    # Try to extend from n=9 up to n=17
    for target_n in range(n0 + 1, 18):
        if time.time() > deadline - 15:
            break
        
        # Initialize new colors
        colors = []
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if j < n0:
                    idx0 = i * (n0 - 1) - i * (i - 1) // 2 + (j - i - 1)
                    colors.append(colors0[idx0])
                else:
                    colors.append(random.randint(0, 3))
        
        # Run local search
        result = _local_search(colors, target_n, deadline - 5)
        
        # Verify
        adj_check = _build_adj_matrices(result, target_n)
        if not _has_c6_fast(adj_check, target_n):
            best_n = target_n
            best_colors = result
        else:
            # Try a few more random restarts if we have time
            for attempt in range(3):
                if time.time() > deadline - 10:
                    break
                colors2 = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if j < n0:
                            idx0 = i * (n0 - 1) - i * (i - 1) // 2 + (j - i - 1)
                            colors2.append(colors0[idx0])
                        else:
                            colors2.append(random.randint(0, 3))
                result2 = _local_search(colors2, target_n, deadline - 5)
                adj_check2 = _build_adj_matrices(result2, target_n)
                if not _has_c6_fast(adj_check2, target_n):
                    best_n = target_n
                    best_colors = result2
                    break
            else:
                break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END

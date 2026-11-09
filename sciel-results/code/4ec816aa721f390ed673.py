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

def _has_c6_in_adj(A, n):
    """Check if adjacency matrix A (n x n) contains a C6. Returns True if found."""
    # Build adjacency list for faster traversal
    adj_list = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                adj_list[i].append(j)
                adj_list[j].append(i)
    
    # For each starting vertex, do DFS to find paths of length 5
    # where all vertices except start are > start (to avoid counting same cycle multiple times)
    for start in range(n):
        # DFS state: (current_vertex, path_so_far, visited_mask)
        stack = [(start, [start], 1 << start)]
        while stack:
            current, path, visited = stack.pop()
            if len(path) == 6:
                # Check if path[0] and path[5] are connected
                if A[path[0], path[5]]:
                    return True
                continue
            for nxt in adj_list[current]:
                if nxt > start and not (visited >> nxt) & 1:
                    stack.append((nxt, path + [nxt], visited | (1 << nxt)))
    return False

def _has_c6_fast(adj_matrices, n):
    """Check if any color class contains a C6. Returns True if found."""
    for c in range(4):
        if _has_c6_in_adj(adj_matrices[c], n):
            return True
    return False

def _count_c6s_fast(adj_matrices, n):
    """Count C6s in each color class. Returns list of 4 counts."""
    counts = [0] * 4
    for c in range(4):
        A = adj_matrices[c]
        adj_list = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j]:
                    adj_list[i].append(j)
                    adj_list[j].append(i)
        
        count = 0
        for start in range(n):
            stack = [(start, [start], 1 << start)]
            while stack:
                current, path, visited = stack.pop()
                if len(path) == 6:
                    if A[path[0], path[5]]:
                        count += 1
                    continue
                for nxt in adj_list[current]:
                    if nxt > start and not (visited >> nxt) & 1:
                        stack.append((nxt, path + [nxt], visited | (1 << nxt)))
        counts[c] = count
    return counts

def _local_search(initial_colors, n, deadline, max_steps=50000):
    """Perform local search to minimize C6 count."""
    colors = list(initial_colors)
    best_colors = list(colors)
    
    # Build initial adjacency matrices
    adj = _build_adj_matrices(colors, n)
    best_c6_count = sum(_count_c6s_fast(adj, n))
    
    if best_c6_count == 0:
        return colors
    
    m = n * (n - 1) // 2
    step = 0
    
    while step < max_steps:
        if time.time() > deadline:
            break
        
        # Pick a random edge to flip
        idx = random.randint(0, m - 1)
        old_color = colors[idx]
        new_color = random.choice([c for c in range(4) if c != old_color])
        
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

def _simulated_annealing(initial_colors, n, deadline):
    """Perform simulated annealing to minimize C6 count."""
    colors = list(initial_colors)
    best_colors = list(colors)
    best_c6_count = sum(_count_c6s_fast(_build_adj_matrices(colors, n), n))
    
    if best_c6_count == 0:
        return colors
    
    m = n * (n - 1) // 2
    T = 1.0
    cooling = 0.999
    min_T = 0.001
    
    while T > min_T:
        if time.time() > deadline:
            break
        
        # Multiple iterations per temperature
        for _ in range(100):
            if time.time() > deadline:
                break
            
            idx = random.randint(0, m - 1)
            old_color = colors[idx]
            new_color = random.choice([c for c in range(4) if c != old_color])
            
            colors[idx] = new_color
            new_c6_count = sum(_count_c6s_fast(_build_adj_matrices(colors, n), n))
            delta = new_c6_count - best_c6_count
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                if new_c6_count < best_c6_count:
                    best_c6_count = new_c6_count
                    best_colors = list(colors)
                    if best_c6_count == 0:
                        return colors
            else:
                colors[idx] = old_color
        
        T *= cooling
    
    return best_colors

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with a known good small solution
    n0 = 9
    colors0 = [min(j - i, n0 - j + i) - 1 for i in range(n0) for j in range(i + 1, n0)]
    
    best_n = n0
    best_colors = list(colors0)
    
    # Try to extend from n=9 up to n=17
    for target_n in range(n0 + 1, 18):
        if time.time() > deadline - 15:
            break
        
        success = False
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Initialize new colors
            colors = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if j < n0:
                        # Use the known good coloring for the first n0 vertices
                        # Need to find the correct index in colors0
                        # Index in colors0 for pair (i,j) where i,j < n0:
                        # The lexicographic index is i*(n0-1) - i*(i-1)//2 + (j-i-1)
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
                success = True
                break
        
        if not success:
            # Try simulated annealing
            for attempt in range(2):
                if time.time() > deadline - 10:
                    break
                
                colors = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if j < n0:
                            idx0 = i * (n0 - 1) - i * (i - 1) // 2 + (j - i - 1)
                            colors.append(colors0[idx0])
                        else:
                            colors.append(random.randint(0, 3))
                
                result = _simulated_annealing(colors, target_n, deadline - 5)
                adj_check = _build_adj_matrices(result, target_n)
                if not _has_c6_fast(adj_check, target_n):
                    best_n = target_n
                    best_colors = result
                    success = True
                    break
        
        if not success:
            # If we failed at this n, try the next one with different initialization
            continue
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END

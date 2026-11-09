"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _build_adj_matrices(colors, n):
    """Build 4 adjacency matrices (one per color) from the color list."""
    adj = [np.zeros((n, n), dtype=np.int32) for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i, j] = 1
            adj[c][j, i] = 1
            idx += 1
    return adj

def _count_c6s_vectorized(adj_mats, n):
    """
    Count C6s in each color class using matrix operations.
    For each color, count cycles of length 6.
    A C6 is counted by: for each edge (u,v), count paths of length 4 from u to v
    that don't use u or v internally. But this is tricky with matrix powers.
    
    Alternative: Use the formula that the number of C6s in a graph can be computed
    from the eigenvalues, but that's complex.
    
    For n <= 20, let's use a direct but vectorized approach:
    Count C6s by iterating over all 6-tuples (a,b,c,d,e,f) where a<b and the
    cycle a-b-c-d-e-f-a exists. To avoid overcounting, require a to be the
    smallest vertex.
    
    Actually, for speed, let's use the following approach:
    For each pair of adjacent vertices (u,v) in the graph, the number of C6s
    containing edge (u,v) equals the number of paths of length 4 from u to v
    that don't use u or v internally.
    
    We can compute this as: (A^4)[u,v] - (number of paths that use u or v internally)
    This is getting complex. Let's just do a semi-vectorized count.
    
    For n <= 20, C(n,6) is at most C(20,6) = 38760. For each 6-subset, we check
    if it contains a C6. But checking all permutations is expensive.
    
    Let's use the backtracking approach but with numpy for neighbor lookup.
    """
    total = 0
    for c in range(4):
        adj = adj_mats[c]
        count = 0
        for s in range(n):
            # Only count cycles where s is the smallest vertex
            # DFS from s, only visiting vertices > s
            # We need paths of length 5 (6 vertices) where last connects back to s
            neighbors = np.where(adj[s, s+1:] > 0)[0] + s + 1
            for start in neighbors:
                # Path: s -> start -> ... -> ? -> s
                # Need 4 more vertices after start, all > s, all distinct
                # Then check if last connects to s
                path = [start]
                _dfs_count_c6(adj, s, path, 1, n)
                # We'll use a helper that returns the count
        # Let me redo this with a proper function
        count = _count_c6s_for_color(adj, n)
        total += count
    return total

def _count_c6s_for_color(adj, n):
    """Count C6s in a single graph, counting each cycle once (smallest vertex first)."""
    count = 0
    for s in range(n):
        # Start from s, only go to vertices > s
        # We need a path of 5 edges: s -> v1 -> v2 -> v3 -> v4 -> v5 -> s
        # where s < v1 < ... is NOT required, but all vi > s and all distinct
        # Actually, we just need all vi > s and all distinct from each other and from s
        # Then check adj[v5][s]
        
        # For efficiency, use iterative deepening
        # Path length 5 from s, visiting only vertices > s
        # Use a stack-based DFS
        
        # To avoid Python recursion overhead, let's use a different approach
        # Count by: for each pair (v1, v5) where v1 > s, v5 > s, v1 != v5,
        # adj[s][v1] and adj[v5][s], count paths of length 3 from v1 to v5
        # using only vertices in {s+1, ..., n-1} \ {v1, v5}
        
        # This is still complex. For n <= 20, let's just do a direct enumeration
        # with some pruning.
        
        # Get neighbors of s that are > s
        nbrs = [v for v in range(s + 1, n) if adj[s, v]]
        if len(nbrs) < 2:
            continue
        
        # For each pair of neighbors (a, b) of s with a < b, both > s
        # Count paths of length 3 from a to b using vertices > s, not using s
        # Wait, the path is a -> x -> y -> b, all in {s+1,...,n-1}, all distinct
        # Then the cycle is s-a-x-y-b-s
        
        for i in range(len(nbrs)):
            a = nbrs[i]
            for j in range(i + 1, len(nbrs)):
                b = nbrs[j]
                # Count paths of length 3 from a to b using vertices > s, excluding a and b
                # Path: a -> x -> y -> b where x, y in {s+1,...,n-1} \ {a, b}, x != y
                # adj[a][x] and adj[x][y] and adj[y][b]
                count += _count_paths_len3(adj, a, b, s, n)
    
    return count

def _count_paths_len3(adj, a, b, min_vertex, n):
    """Count paths a->x->y->b where x,y > min_vertex, x!=y, x,y != a,b."""
    count = 0
    for x in range(min_vertex + 1, n):
        if x == a or x == b:
            continue
        if not adj[a, x]:
            continue
        for y in range(min_vertex + 1, n):
            if y == a or y == b or y == x:
                continue
            if adj[x, y] and adj[y, b]:
                count += 1
    return count

def _has_c6(colors, n):
    """Check if any color class has a C6."""
    adj_mats = _build_adj_matrices(colors, n)
    return _count_c6s_vectorized(adj_mats, n) > 0

def _count_c6s_fast(colors, n):
    """Fast count of C6s."""
    adj_mats = _build_adj_matrices(colors, n)
    return _count_c6s_vectorized(adj_mats, n)

def _simulate_annealing(colors, n, deadline, max_iter=200000):
    """Simulated annealing to minimize C6 count."""
    colors = list(colors)
    adj_mats = _build_adj_matrices(colors, n)
    current_c6 = _count_c6s_vectorized(adj_mats, n)
    best_c6 = current_c6
    best_colors = list(colors)
    
    if best_c6 == 0:
        return best_colors
    
    T = 1.0
    T_min = 0.001
    cooling = 0.99995
    
    step = 0
    while step < max_iter and time.time() < deadline:
        # Pick a random edge to change color
        idx = random.randint(0, len(colors) - 1)
        old_color = colors[idx]
        new_color = random.randint(0, 3)
        if new_color == old_color:
            step += 1
            continue
        
        colors[idx] = new_color
        adj_mats = _build_adj_matrices(colors, n)
        new_c6 = _count_c6s_vectorized(adj_mats, n)
        
        delta = new_c6 - current_c6
        if delta <= 0 or random.random() < np.exp(-delta / max(T, T_min)):
            current_c6 = new_c6
            if new_c6 < best_c6:
                best_c6 = new_c6
                best_colors = list(colors)
                if best_c6 == 0:
                    return best_colors
        else:
            colors[idx] = old_color
        
        T = max(T * cooling, T_min)
        step += 1
    
    return best_colors

def _local_search(colors, n, deadline, max_iter=100000):
    """Local search: only accept improvements."""
    colors = list(colors)
    adj_mats = _build_adj_matrices(colors, n)
    current_c6 = _count_c6s_vectorized(adj_mats, n)
    best_c6 = current_c6
    best_colors = list(colors)
    
    if best_c6 == 0:
        return best_colors
    
    step = 0
    while step < max_iter and time.time() < deadline:
        idx = random.randint(0, len(colors) - 1)
        old_color = colors[idx]
        new_color = random.randint(0, 3)
        if new_color == old_color:
            step += 1
            continue
        
        colors[idx] = new_color
        adj_mats = _build_adj_matrices(colors, n)
        new_c6 = _count_c6s_vectorized(adj_mats, n)
        
        if new_c6 < current_c6:
            current_c6 = new_c6
            if new_c6 < best_c6:
                best_c6 = new_c6
                best_colors = list(colors)
                if best_c6 == 0:
                    return best_colors
        else:
            colors[idx] = old_color
        
        step += 1
    
    return best_colors

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Start with a random valid small coloring and try to grow
    # First, find a valid coloring for small n
    best_n = 2
    best_colors = [0]  # Just one edge for n=2
    
    # Try to build up from n=2
    for target_n in range(3, 18):
        if time.time() > deadline - 10:
            break
        
        # Initialize: use existing colors for pairs within best_n, random for new
        colors = []
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if i < best_n and j < best_n:
                    # Find index in best_colors
                    idx = i * (best_n - 1) - i * (i - 1) // 2 + (j - i - 1)
                    colors.append(best_colors[idx])
                else:
                    colors.append(random.randint(0, 3))
        
        # Try multiple restarts with different search strategies
        found = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            # Randomize new edges
            if attempt > 0:
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if i >= best_n or j >= best_n:
                            idx = i * (target_n - 1) - i * (i - 1) // 2 + (j - i - 1)
                            colors[idx] = random.randint(0, 3)
            
            # Run local search
            result = _local_search(colors, target_n, deadline - 2)
            if _has_c6(result, target_n):
                continue
            
            # Verify
            if not _has_c6(result, target_n):
                best_n = target_n
                best_colors = result
                found = True
                break
        
        if not found:
            # Try simulated annealing for this n
            if time.time() < deadline - 5:
                result = _simulate_annealing(colors, target_n, deadline - 2)
                if not _has_c6(result, target_n):
                    best_n = target_n
                    best_colors = result
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END

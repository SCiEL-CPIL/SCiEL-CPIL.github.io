import time
import random
import math
import numpy as np

def construct():
    deadline = time.time() + 110  # safety margin under 120s
    
    # We'll try to find a valid 5-coloring for increasing n.
    # The seed gives n=11. Lazebnik-Woldar gives n=26.
    # We'll use simulated annealing with a bitset-based violation counter.
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Target sizes to try, in increasing order
    target_sizes = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    
    for n in target_sizes:
        if time.time() > deadline - 10:
            break
        
        # Try to find a valid coloring for this n
        success, colors = _sa_search(n, time_limit=deadline - 15)
        if success:
            best_n = n
            best_colors = colors
            # Continue trying larger n
            continue
        else:
            # Failed to find valid coloring for this n
            # Try next size but with less time
            continue
    
    return {'n': best_n, 'colors': best_colors}


def _count_c4_violations(n, colors):
    """Count the number of C4 violations in the coloring.
    
    A C4 violation occurs when two vertices share two common neighbors in the same color.
    For each pair of vertices (u, v), for each color c, count the number of vertices w
    such that (u,w) has color c and (v,w) has color c. If this count is >= 2, it's a violation.
    
    But wait - the constraint is: no two vertices have two common neighbors in ONE color.
    So for each color c, the subgraph of edges with color c must be C4-free.
    A C4 in color c exists if there are vertices a, b, c, d such that edges ab, bc, cd, da all have color c.
    Equivalently, in the color-c subgraph, no two vertices share two common neighbors.
    
    For efficiency, for each color c, for each pair of vertices (u,v), we can count common neighbors in color c.
    If any pair has >= 2 common neighbors in color c, that's a violation.
    
    Actually, let me think about this more carefully. The condition "no two vertices with two common neighbours in one colour" means:
    For any color c and any pair of vertices (u, v), the number of vertices w such that edge (u,w) is color c AND edge (v,w) is color c must be at most 1.
    
    This is equivalent to saying the color-c subgraph is C4-free.
    """
    if len(colors) != n * (n - 1) // 2:
        return float('inf')
    
    # Build adjacency representation for each color
    # adj[c][i] = set of neighbors of i in color c
    # Using numpy bitsets for speed
    
    total_violations = 0
    
    # For each color, check for C4s
    # A C4 exists if there are two vertices u, v that share >= 2 common neighbors in that color
    
    # Let's use a different approach: for each color c, build the adjacency matrix,
    # then for each pair of vertices, count common neighbors.
    
    # Index into colors array: for pair (i, j) with i < j, index = i * n - i * (i+1) // 2 + (j - i - 1)
    # Wait, let me re-derive. The colors are in lexicographic order: (0,1), (0,2), ..., (0,n-1), (1,2), (1,3), ..., (n-2, n-1)
    
    # Let's build a 2D array for quick lookup
    edge_color = np.full((n, n), -1, dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            edge_color[i, j] = c
            edge_color[j, i] = c
            idx += 1
    
    # For each color, check C4-freeness
    for c in range(5):
        # Build adjacency list for color c
        # adj[c][i] = list of neighbors
        # For counting common neighbors, we can use bitsets
        
        # Using numpy: for each vertex i, a bitmask of neighbors in color c
        neighbors = [0] * n  # bitmask as Python int or numpy array
        for i in range(n):
            mask = 0
            for j in range(n):
                if i != j and edge_color[i, j] == c:
                    mask |= (1 << j)
            neighbors[i] = mask
        
        # For each pair (u, v), count common neighbors in color c
        # common = popcount(neighbors[u] & neighbors[v])
        # If common >= 2, violation
        for u in range(n):
            for v in range(u + 1, n):
                common = bin(neighbors[u] & neighbors[v]).count('1')
                if common >= 2:
                    total_violations += (common - 1)
    
    return total_violations


def _sa_search(n, time_limit):
    """Simulated annealing to find a valid 5-coloring of K_n with no C4 in any color."""
    if n < 3:
        return True, []
    
    m = n * (n - 1) // 2
    
    # Initialize with random coloring
    colors = [random.randint(0, 4) for _ in range(m)]
    
    # Precompute for fast violation counting
    # We'll maintain a violation count and update incrementally
    
    # Build edge_color matrix
    edge_color = np.full((n, n), -1, dtype=np.int8)
    for idx in range(m):
        i = idx // (n - 1)  # This isn't right for lexicographic order
        # Let me compute the (i,j) for each index
        # Lexicographic: (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2,n-1)
        # For index k, find i, j such that k = i*(n-1) - i*(i+1)//2 + (j - i - 1)
        # This is getting complex. Let me precompute.
        pass
    
    # Precompute edge index to (i, j) mapping
    edge_to_pair = []
    pair_to_edge = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_pair.append((i, j))
            pair_to_edge[(i, j)] = idx
            pair_to_edge[(j, i)] = idx
            idx += 1
    
    # Build edge_color matrix
    def build_edge_color(colors):
        ec = np.full((n, n), -1, dtype=np.int8)
        for k in range(m):
            i, j = edge_to_pair[k]
            c = colors[k]
            ec[i, j] = c
            ec[j, i] = c
        return ec
    
    # Count violations
    def count_violations(colors):
        ec = build_edge_color(colors)
        total = 0
        for c in range(5):
            # For each color, build neighbor bitmasks
            neighbors = [0] * n
            for i in range(n):
                mask = 0
                for j in range(n):
                    if i != j and ec[i, j] == c:
                        mask |= (1 << j)
                neighbors[i] = mask
            
            for u in range(n):
                for v in range(u + 1, n):
                    common = bin(neighbors[u] & neighbors[v]).count('1')
                    if common >= 2:
                        total += (common - 1)
        return total
    
    current_violations = count_violations(colors)
    best_violations = current_violations
    best_colors = colors[:]
    
    if current_violations == 0:
        return True, colors
    
    T = 1.0
    alpha = 0.999
    min_T = 0.001
    
    start_time = time.time()
    
    while time.time() < time_limit:
        # Pick a random edge and change its color
        k = random.randint(0, m - 1)
        i, j = edge_to_pair[k]
        old_color = colors[k]
        new_color = random.randint(0, 4)
        
        if new_color == old_color:
            continue
        
        # Calculate delta violations efficiently
        # This is complex, so for now, just recount
        # Optimization: only recalculate violations involving edges (i, *) or (j, *)
        
        colors[k] = new_color
        new_violations = count_violations(colors)
        delta = new_violations - current_violations
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors[:]
                if new_violations == 0:
                    return True, best_colors
        else:
            colors[k] = old_color  # revert
        
        T *= alpha
        if T < min_T:
            T = min_T
    
    if best_violations == 0:
        return True, best_colors
    
    return False, best_colors


# Optimize with numpy-based approach for larger n
def _sa_search_v2(n, time_limit):
    """Faster simulated annealing using numpy for violation counting."""
    if n < 3:
        return True, []
    
    m = n * (n - 1) // 2
    
    colors = np.random.randint(0, 5, size=m, dtype=np.int8)
    
    # Precompute edge to pair
    edge_to_i = np.zeros(m, dtype=np.int32)
    edge_to_j = np.zeros(m, dtype=np.int32)
    pair_to_edge = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_i[idx] = i
            edge_to_j[idx] = j
            pair_to_edge[(i, j)] = idx
            pair_to_edge[(j, i)] = idx
            idx += 1
    
    def build_ec(colors):
        ec = np.full((n, n), -1, dtype=np.int8)
        i_arr = edge_to_i
        j_arr = edge_to_j
        ec[i_arr, j_arr] = colors
        ec[j_arr, i_arr] = colors
        return ec
    
    def count_violations_fast(ec):
        total = 0
        for c in range(5):
            # Adjacency for color c
            adj = (ec == c).astype(np.int8)
            np.fill_diagonal(adj, 0)
            
            # For each pair (u,v), count common neighbors
            # Common neighbors of u and v in color c = (adj[u] & adj[v]).sum()
            # Using matrix multiply: adj @ adj.T gives for each (u,v) the number of common neighbors
            common = adj @ adj.T  # n x n matrix, common[u,v] = number of common neighbors
            
            # Count pairs with >= 2 common neighbors
            # For each u < v, if common[u,v] >= 2, add (common[u,v] - 1)
            for u in range(n):
                for v in range(u + 1, n):
                    if common[u, v] >= 2:
                        total += (common[u, v] - 1)
        return total
    
    ec = build_ec(colors)
    current_violations = count_violations_fast(ec)
    best_violations = current_violations
    best_colors = colors.copy()
    
    if current_violations == 0:
        return True, colors.tolist()
    
    T = 1.0
    alpha = 0.9995
    min_T = 0.001
    
    start_time = time.time()
    iter_count = 0
    
    while time.time() < time_limit:
        k = random.randint(0, m - 1)
        i = edge_to_i[k]
        j = edge_to_j[k]
        old_color = colors[k]
        new_color = random.randint(0, 4)
        
        if new_color == old_color:
            continue
        
        colors[k] = new_color
        ec = build_ec(colors)
        new_violations = count_violations_fast(ec)
        delta = new_violations - current_violations
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors.copy()
                if new_violations == 0:
                    return True, colors.tolist()
        else:
            colors[k] = old_color
            ec = build_ec(colors)
        
        T *= alpha
        if T < min_T:
            T = min_T
        
        iter_count += 1
        # Restart if stuck
        if iter_count % 10000 == 0 and best_violations > 0:
            # Try to kick out of local minimum
            for _ in range(10):
                k2 = random.randint(0, m - 1)
                colors[k2] = random.randint(0, 4)
            ec = build_ec(colors)
            current_violations = count_violations_fast(ec)
            T = max(T, 0.5)
    
    if best_violations == 0:
        return True, best_colors.tolist()
    
    return False, best_colors.tolist()


def construct():
    deadline = time.time() + 110
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to improve from n=12 upward
    target_sizes = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    
    for n in target_sizes:
        if time.time() > deadline - 15:
            break
        
        # Allocate decreasing time for larger n
        time_remaining = deadline - time.time() - 5
        if time_remaining < 5:
            break
        
        # Try SA search
        success, colors = _sa_search_v2(n, time.time() + time_remaining * 0.6)
        if success:
            best_n = n
            best_colors = colors
            # Continue to try larger n with remaining time
            continue
    
    return {'n': best_n, 'colors': best_colors}

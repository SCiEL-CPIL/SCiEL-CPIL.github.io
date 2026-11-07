"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _is_bicolored_forest(colors, n, edges):
    """Check if for every pair of colors, the induced subgraph is a forest."""
    # Group edges by color pair
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        if cu == cv:
            return False  # Adjacent same color
        key = (min(cu, cv), max(cu, cv))
        pair_edges[key].append((u, v))
    
    # For each pair, check if it's a forest using union-find
    for pair, elist in pair_edges.items():
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True
        
        for u, v in elist:
            if not union(u, v):
                return False  # Cycle found
    return True

def _count_distinct_colors(colors):
    return len(set(colors))

def _try_periodic_coloring(n, edges, period, pattern):
    """Try a periodic coloring with given period and pattern."""
    colors = [pattern[i % period] for i in range(n)]
    if _is_bicolored_forest(colors, n, edges):
        return colors
    return None

def _local_search(colors, n, edges, adj, max_iters=100000, deadline=None):
    """Local search to minimize distinct colors while maintaining validity."""
    if deadline is None:
        deadline = time.time() + 5
    
    def is_valid(c):
        # Check adjacent different
        for u, v in edges:
            if c[u] == c[v]:
                return False
        # Check bicolored forest
        return _is_bicolored_forest(c, n, edges)
    
    def num_colors(c):
        return len(set(c))
    
    best = colors[:]
    best_score = num_colors(best)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try to change a random vertex's color to an existing color
        v = it % n  # Cycle through vertices
        current_color = best[v]
        
        # Get candidate colors: existing colors used in the graph
        used_colors = set(best)
        
        # Try each used color (and maybe 0 if not used)
        candidates = list(used_colors)
        if 0 not in candidates:
            candidates.append(0)
        
        # Shuffle for randomness
        import random
        random.shuffle(candidates)
        
        for new_color in candidates:
            if new_color == current_color:
                continue
            
            # Check if new_color is adjacent to v
            if any(best[nb] == new_color for nb in adj[v]):
                continue
            
            # Try the change
            trial = best[:]
            trial[v] = new_color
            
            if is_valid(trial):
                new_score = num_colors(trial)
                if new_score < best_score:
                    best = trial
                    best_score = new_score
                    break
                elif new_score == best_score and random.random() < 0.1:
                    # Accept equal-score moves to escape local optima
                    best = trial
                    break
    
    return best

def _solve_with_k_colors(n, edges, k, deadline):
    """Try to find a valid k-coloring using backtracking with pruning."""
    adj = _build_adjacency(n, edges)
    colors = [-1] * n
    
    # Order vertices by degree (highest first)
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: (-degrees[x], x))
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        v = order[idx]
        
        # Forbidden colors from neighbors
        forbidden = set()
        for nb in adj[v]:
            if colors[nb] != -1:
                forbidden.add(colors[nb])
        
        for color in range(k):
            if color in forbidden:
                continue
            
            # Tentatively assign
            colors[v] = color
            
            # Check bicolored forest constraint incrementally
            # For each neighbor with a different color, check if adding this edge creates a cycle
            valid = True
            for nb in adj[v]:
                if colors[nb] == -1:
                    continue
                if colors[nb] == color:
                    valid = False
                    break
                # Check if the edge (v, nb) would create a cycle in the bicolored subgraph
                # We need to check if v and nb are already connected in the subgraph of colors {color, colors[nb]}
                # This is expensive to check incrementally, so we'll do a full check after assignment
            if not valid:
                colors[v] = -1
                continue
            
            # Full check is expensive, so we'll rely on the final verification
            # For now, just proceed with backtracking
            if backtrack(idx + 1):
                return True
            
            colors[v] = -1
        
        return False
    
    if backtrack(0):
        return colors
    return None

def construct():
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(n, edges)
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # First, try to find a valid coloring with few colors using periodic patterns
    # Try various periods and patterns
    
    best_coloring = None
    best_num_colors = n  # Start with worst case
    
    # Try small periods
    for period in range(2, 10):
        if time.time() > deadline:
            break
        
        # Try all possible patterns of this period using k colors
        # For efficiency, try specific patterns first
        patterns_to_try = []
        
        # Pattern: 0,1,0,1,... (2 colors)
        if period == 2:
            patterns_to_try.append([0, 1])
        
        # Pattern: 0,1,2,0,1,2,... (3 colors)
        if period == 3:
            patterns_to_try.append([0, 1, 2])
        
        # Pattern: 0,1,2,3,... (4 colors)
        if period == 4:
            patterns_to_try.append([0, 1, 2, 3])
        
        # Pattern: 0,1,0,2,... 
        if period == 4:
            patterns_to_try.append([0, 1, 0, 2])
            patterns_to_try.append([0, 1, 2, 1])
        
        for pattern in patterns_to_try:
            colors = _try_periodic_coloring(n, edges, period, pattern)
            if colors is not None:
                nc = _count_distinct_colors(colors)
                if nc < best_num_colors:
                    best_num_colors = nc
                    best_coloring = colors[:]
    
    # If we found a good coloring, try to improve with local search
    if best_coloring is not None:
        improved = _local_search(best_coloring, n, edges, adj, max_iters=50000, deadline=deadline)
        if _is_bicolored_forest(improved, n, edges):
            nc = _count_distinct_colors(improved)
            if nc < best_num_colors:
                best_num_colors = nc
                best_coloring = improved
    
    # If still no good coloring, try backtracking with increasing k
    if best_coloring is None or best_num_colors > 6:
        for k in range(3, min(8, n + 1)):
            if time.time() > deadline:
                break
            result = _solve_with_k_colors(n, edges, k, deadline)
            if result is not None and _is_bicolored_forest(result, n, edges):
                nc = _count_distinct_colors(result)
                if nc < best_num_colors:
                    best_num_colors = nc
                    best_coloring = result[:]
    
    # Fallback: ensure we have a valid coloring
    if best_coloring is None:
        # Use a simple valid coloring: assign colors based on position
        best_coloring = [i % 4 for i in range(n)]
        # Verify and fix if needed
        if not _is_bicolored_forest(best_coloring, n, edges):
            # Last resort: unique colors
            best_coloring = list(range(n))
    
    # Final verification
    if not _is_bicolored_forest(best_coloring, n, edges):
        # Force a valid coloring
        best_coloring = list(range(n))
    
    return [int(c) for c in best_coloring]
# EVOLVE-BLOCK-END

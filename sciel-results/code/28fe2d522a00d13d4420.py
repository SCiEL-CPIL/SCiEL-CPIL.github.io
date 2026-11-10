"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def has_bicolored_cycle(coloring, edges):
    """Check if any pair of colors forms a cycle using union-find."""
    n = len(coloring)
    # Group edges by color pair
    from collections import defaultdict
    color_pairs = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu != cv:
            key = (min(cu, cv), max(cu, cv))
            color_pairs[key].append((u, v))
    
    for (c1, c2), edge_list in color_pairs.items():
        # Union-find for this color pair's subgraph
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False  # cycle detected
            parent[rx] = ry
            return True
        for u, v in edge_list:
            if not union(u, v):
                return True  # bicolored cycle found
    return False

def count_colors(coloring):
    return len(set(coloring))

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Try to find a good coloring
    # The graph appears to be a grid-like structure
    # Let's try a systematic approach
    
    # First, try a simple pattern-based coloring
    # The graph seems to have rows and columns
    # Looking at the structure, it appears to be a 7x7 or similar grid with diagonals
    
    # Let me try a 4-coloring based on position
    # Assign colors based on (row + col) % k pattern
    
    # Actually, let's try to use a backtracking search with the constraint
    # that no two colors form a cycle
    
    # Start with a reasonable initial coloring
    # Try coloring by vertex index patterns
    
    best_coloring = list(range(n))  # fallback: all distinct
    best_num_colors = n
    
    # Try small number of colors first
    for num_colors in [2, 3, 4, 5, 6, 7, 8]:
        if time.time() > deadline:
            break
        
        # Try to find a valid coloring with exactly num_colors colors
        # Use backtracking with constraint propagation
        
        def backtrack(vertex, current_coloring, color_usage):
            if time.time() > deadline:
                return None
            
            if vertex == n:
                # Check for bicolored cycles
                if not has_bicolored_cycle(current_coloring, edges):
                    return current_coloring[:]
                return None
            
            # Get available colors (not used by neighbors)
            neighbor_colors = set()
            for nb in adj[vertex]:
                if current_coloring[nb] is not None:
                    neighbor_colors.add(current_coloring[nb])
            
            for c in range(num_colors):
                if c not in neighbor_colors:
                    current_coloring[vertex] = c
                    result = backtrack(vertex + 1, current_coloring, color_usage)
                    if result is not None:
                        return result
                    current_coloring[vertex] = None
            
            return None
        
        # Try backtracking
        current = [None] * n
        result = backtrack(0, current, None)
        if result is not None:
            best_coloring = result
            best_num_colors = num_colors
            break
    
    # Verify the result
    assert all(isinstance(c, int) for c in best_coloring)
    assert all(0 <= c < 50 for c in best_coloring)
    assert len(best_coloring) == 50
    
    # Check adjacency constraint
    for u, v in edges:
        assert best_coloring[u] != best_coloring[v], f"Adjacent vertices {u},{v} have same color"
    
    # Check no bicolored cycle
    assert not has_bicolored_cycle(best_coloring, edges), "Bicolored cycle detected"
    
    return list(best_coloring)

# EVOLVE-BLOCK-END

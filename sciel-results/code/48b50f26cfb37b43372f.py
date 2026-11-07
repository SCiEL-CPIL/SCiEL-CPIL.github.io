"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolored_forest(colors, n, edges):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    # Group edges by color pair
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        if cu == cv:
            return False  # Adjacent vertices must differ
        key = (min(cu, cv), max(cu, cv))
        pair_edges[key].append((u, v))
    
    # For each pair, check if it forms a forest using union-find
    for pair, elist in pair_edges.items():
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False  # Cycle detected
            parent[rx] = ry
            return True
        for u, v in elist:
            if not union(u, v):
                return False
    return True

def _incremental_check(colors, n, edges, vertex, old_color, new_color):
    """
    Incrementally check if changing vertex's color from old_color to new_color
    maintains the bicolored forest property.
    Only checks the affected color pairs: (old_color, neighbor_colors) and (new_color, neighbor_colors)
    """
    if new_color == old_color:
        return True
    
    # Get neighbors of the vertex
    neighbors = []
    for u, v in edges:
        if u == vertex:
            neighbors.append(v)
        elif v == vertex:
            neighbors.append(u)
    
    # Check if new_color conflicts with any neighbor's color
    for nb in neighbors:
        if colors[nb] == new_color:
            return False
    
    # We need to check that the bicolored subgraphs for pairs involving old_color and new_color
    # remain forests. The tricky part is that removing an edge from one pair and adding to another
    # could create cycles.
    
    # Build the set of affected color pairs
    affected_pairs = set()
    for nb in neighbors:
        nc = colors[nb]
        if nc != old_color and nc != new_color:
            affected_pairs.add((min(old_color, nc), max(old_color, nc)))
            affected_pairs.add((min(new_color, nc), max(new_color, nc)))
    
    # For each affected pair, check if it's still a forest
    # We need to reconstruct the edge list for that pair, accounting for the change
    for pair in affected_pairs:
        c1, c2 = pair
        # Collect all edges with this color pair, adjusting for the vertex change
        pair_edges_list = []
        for u, v in edges:
            cu, cv = colors[u], colors[v]
            # If one endpoint is the vertex being changed, use its new color
            if u == vertex:
                cu = new_color
            if v == vertex:
                cv = new_color
            if cu == cv:
                continue
            pkey = (min(cu, cv), max(cu, cv))
            if pkey == pair:
                pair_edges_list.append((u, v))
        
        # Check if this forms a forest
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        for u, v in pair_edges_list:
            if not union(u, v):
                return False
    
    return True

def _backtracking_search(n, edges, num_colors, deadline, rng):
    """
    Backtracking search with constraint propagation for bicolored forest coloring.
    Uses incremental checking for speed.
    """
    colors = [-1] * n
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute edge list for quick access
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    def find(x, parent):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y, parent):
        rx, ry = find(x, parent), find(y, parent)
        if rx == ry:
            return False
        parent[rx] = ry
        return True
    
    def is_valid_assignment(vertex, color):
        """Check if assigning 'color' to 'vertex' is consistent with current colors."""
        for nb in adj[vertex]:
            if colors[nb] == color:
                return False
        return True
    
    def check_forest_after_assign(vertex, color):
        """Check that all bicolored subgraphs remain forests after assigning color to vertex."""
        # Get neighbors
        neighbors = list(adj[vertex])
        
        # For each neighbor, check the bicolored subgraph for (color, colors[nb])
        for nb in neighbors:
            nb_color = colors[nb]
            if nb_color == -1 or nb_color == color:
                continue
            
            c1, c2 = min(color, nb_color), max(color, nb_color)
            
            # Build edge list for this color pair
            pair_edges = []
            for u, v in edges:
                cu, cv = colors[u], colors[v]
                if u == vertex:
                    cu = color
                if v == vertex:
                    cv = color
                if cu == -1 or cv == -1 or cu == cv:
                    continue
                pkey = (min(cu, cv), max(cu, cv))
                if pkey == (c1, c2):
                    pair_edges.append((u, v))
            
            # Check forest
            parent = list(range(n))
            for u, v in pair_edges:
                if not union(u, v, parent):
                    return False
        
        return True
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        
        if idx == n:
            return True
        
        # Try colors in random order
        color_order = list(range(num_colors))
        rng.shuffle(color_order)
        
        for color in color_order:
            if not is_valid_assignment(idx, color):
                continue
            
            colors[idx] = color
            
            if check_forest_after_assign(idx, color):
                if backtrack(idx + 1):
                    return True
            
            colors[idx] = -1
        
        return False
    
    if backtrack(0):
        return colors
    return None

def _greedy_with_restarts(n, edges, num_colors, restarts, deadline, rng):
    """
    Greedy coloring with random restarts and backtracking.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    for _ in range(restarts):
        if time.time() > deadline:
            break
        
        # Random vertex ordering
        order = list(range(n))
        rng.shuffle(order)
        
        colors = [-1] * n
        success = True
        
        for idx in order:
            if time.time() > deadline:
                success = False
                break
            
            # Find available colors
            used_colors = set()
            for nb in adj[idx]:
                if colors[nb] != -1:
                    used_colors.add(colors[nb])
            
            # Try each color
            assigned = False
            color_order = list(range(num_colors))
            rng.shuffle(color_order)
            
            for color in color_order:
                if color in used_colors:
                    continue
                
                colors[idx] = color
                
                # Quick check: verify bicolored forest property for affected pairs
                valid = True
                for nb in adj[idx]:
                    if colors[nb] == -1:
                        continue
                    nb_color = colors[nb]
                    c1, c2 = min(color, nb_color), max(color, nb_color)
                    
                    # Check if this pair forms a cycle
                    parent = list(range(n))
                    def find(x):
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x
                    def union(x, y):
                        rx, ry = find(x), find(y)
                        if rx == ry:
                            return False
                        parent[rx] = ry
                        return True
                    
                    for u, v in edges:
                        cu, cv = colors[u], colors[v]
                        if cu == -1 or cv == -1 or cu == cv:
                            continue
                        pkey = (min(cu, cv), max(cu, cv))
                        if pkey == (c1, c2):
                            if not union(u, v):
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if not valid:
                        break
                
                if valid:
                    assigned = True
                    break
            
            if not assigned:
                success = False
                break
        
        if success:
            return colors
    
    return None

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    rng = random.Random(42)
    best = None
    best_num_colors = n  # Default: all different colors
    
    # Try from 3 colors upward
    for num_colors in range(3, n + 1):
        if best_num_colors <= num_colors:
            break
        if time.time() > deadline:
            break
        
        # First try backtracking search
        result = _backtracking_search(n, edges, num_colors, deadline, rng)
        if result is not None:
            best = result
            best_num_colors = num_colors
            break
        
        # Then try greedy with restarts
        result = _greedy_with_restarts(n, edges, num_colors, 50, deadline, rng)
        if result is not None:
            best = result
            best_num_colors = num_colors
            break
    
    if best is None:
        # Fallback: all different colors (always valid)
        best = list(range(n))
    
    # Ensure Python ints and list
    return [int(c) for c in best]
# EVOLVE-BLOCK-END

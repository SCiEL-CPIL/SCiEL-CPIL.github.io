import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Best known: 17-vertex 4-coloring without monochromatic C6
    # We'll try to find a coloring with n=17 using backtracking,
    # and if time permits, attempt n=18.
    
    # First, let's try a systematic search for n=17
    # Use a known structure or randomized backtracking
    
    best_n = 0
    best_colors = []
    
    # Try to find a valid 4-coloring for increasing n
    for target_n in [17, 18]:
        if time.time() > deadline:
            break
        
        result = search_coloring(target_n, deadline)
        if result is not None:
            best_n = target_n
            best_colors = result
            break
    
    # If we didn't find anything better, use a known 17-vertex coloring
    if best_n == 0:
        # Fallback: try to construct a valid 17-vertex coloring
        best_n, best_colors = find_known_17()
    
    if best_n == 0:
        # Absolute fallback: return n=1 (trivially valid)
        best_n = 1
        best_colors = []
    
    return {'n': best_n, 'colors': best_colors}

def find_known_17():
    """Try to construct or find a known 17-vertex 4-coloring without mono C6."""
    # Use randomized backtracking to find a valid coloring
    for attempt in range(50):
        result = search_coloring(17, time.time() + 20.0)
        if result is not None:
            return 17, result
    return 0, []

def search_coloring(n, deadline):
    """Backtracking search for a 4-coloring of K_n with no mono C6."""
    random.seed(42 + n)  # Deterministic but varied
    
    # Number of edges in K_n
    num_edges = n * (n - 1) // 2
    
    # colors[i] = color of edge (i, j) where we store in lexicographic order
    colors = [-1] * num_edges
    
    # Build adjacency info: for each color, track which edges have that color
    # adj[c][u][v] = True if edge (u,v) has color c
    adj = [[[False] * n for _ in range(n)] for _ in range(4)]
    
    # Map from (i,j) with i<j to index in colors array
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)
    
    def set_color(i, j, c):
        idx = edge_idx(i, j)
        colors[idx] = c
        adj[c][i][j] = True
        adj[c][j][i] = True
    
    def unset_color(i, j, c):
        idx = edge_idx(i, j)
        colors[idx] = -1
        adj[c][i][j] = False
        adj[c][j][i] = False
    
    def has_mono_c6(adj, n):
        """Check if any color class contains a C6 (not necessarily induced)."""
        for c in range(4):
            ac = adj[c]
            # Look for 6 distinct vertices forming a cycle
            # A cycle v0-v1-v2-v3-v4-v5-v0
            for v0 in range(n):
                for v1 in range(v0 + 1, n):
                    if not ac[v0][v1]:
                        continue
                    for v2 in range(n):
                        if v2 in (v0, v1) or not ac[v1][v2]:
                            continue
                        for v3 in range(n):
                            if v3 in (v0, v1, v2) or not ac[v2][v3]:
                                continue
                            for v4 in range(n):
                                if v4 in (v0, v1, v2, v3) or not ac[v3][v4]:
                                    continue
                                for v5 in range(n):
                                    if v5 in (v0, v1, v2, v3, v4) or not ac[v4][v5] or not ac[v5][v0]:
                                        continue
                                    # Found a C6: v0-v1-v2-v3-v4-v5-v0
                                    # Make sure all vertices are distinct
                                    if len(set([v0, v1, v2, v3, v4, v5])) == 6:
                                        return True
        return False
    
    def has_mono_c6_fast(adj, n, last_vertex):
        """Faster check: only look for C6s involving the last placed edge's endpoints."""
        # This is a heuristic; full check is expensive
        # For now, do a full check but optimized
        for c in range(4):
            ac = adj[c]
            for v0 in range(n):
                for v1 in range(v0 + 1, n):
                    if not ac[v0][v1]:
                        continue
                    for v2 in range(n):
                        if v2 in (v0, v1) or not ac[v1][v2]:
                            continue
                        for v3 in range(n):
                            if v3 in (v0, v1, v2) or not ac[v2][v3]:
                                continue
                            for v4 in range(n):
                                if v4 in (v0, v1, v2, v3) or not ac[v3][v4]:
                                    continue
                                for v5 in range(n):
                                    if v5 in (v0, v1, v2, v3, v4) or not ac[v4][v5] or not ac[v5][v0]:
                                        continue
                                    if len(set([v0, v1, v2, v3, v4, v5])) == 6:
                                        return True
        return False
    
    # Order edges for backtracking
    # Process edges in an order that builds the graph gradually
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_list.append((i, j))
    
    # Use a smarter ordering: process edges incident to new vertices first
    # Group by max endpoint
    edge_list.sort(key=lambda e: (max(e), min(e)))
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        
        if pos == len(edge_list):
            # All edges colored, verify no mono C6
            if has_mono_c6(adj, n):
                return False
            return True
        
        i, j = edge_list[pos]
        
        for c in [0, 1, 2, 3]:
            set_color(i, j, c)
            
            # Quick check: does adding this edge create a mono C6?
            # This is expensive, so do a lighter check
            # Check if any C6 in color c uses both edge (i,j) and 4 other edges
            # For efficiency, skip the full check and only check at the end
            # But this will be very slow. Let's add a incremental check.
            
            # Incremental C6 check involving edge (i,j) in color c
            if not check_c6_with_edge(adj, c, n, i, j):
                if backtrack(pos + 1):
                    return True
            else:
                # Edge (i,j) in color c creates a mono C6, skip
                pass
            
            unset_color(i, j, c)
        
        return False
    
    def check_c6_with_edge(adj, c, n, i, j):
        """Check if edge (i,j) in color c is part of a monochromatic C6."""
        ac = adj[c]
        # We need a path of length 4 from i to j in color c
        # i - v1 - v2 - v3 - v4 - j where all these edges are in color c
        # and all vertices distinct
        
        # BFS from i to find paths of length 4 to j
        # Use sets for visited states (path tracking)
        
        # State: (current_vertex, visited_set)
        # But visited_set is expensive. Use a different approach.
        
        # Find all paths of length 4 from i to j in the graph ac
        # Path: i -> v1 -> v2 -> v3 -> v4 -> j
        
        # v1 must be a neighbor of i (not j)
        # v4 must be a neighbor of j (not i)
        
        neighbors_i = [v for v in range(n) if v != j and ac[i][v]]
        neighbors_j = [v for v in range(n) if v != i and ac[j][v]]
        
        if not neighbors_i or not neighbors_j:
            return False
        
        for v1 in neighbors_i:
            # v2 must be neighbor of v1, not in {i, j, v1}
            for v2 in range(n):
                if v2 in (i, j, v1) or not ac[v1][v2]:
                    continue
                # v3 must be neighbor of v2, not in {i, j, v1, v2}
                for v3 in range(n):
                    if v3 in (i, j, v1, v2) or not ac[v2][v3]:
                        continue
                    # v4 must be neighbor of v3, not in {i, j, v1, v2, v3}
                    # and neighbor of j
                    for v4 in range(n):
                        if v4 in (i, j, v1, v2, v3) or not ac[v3][v4] or not ac[v4][j]:
                            continue
                        # Found a C6: i-v1-v2-v3-v4-j-i
                        return True
        
        return False
    
    # The full backtracking with C6 checking at each step is too slow for n=17.
    # Let's try a different approach: use a known construction or more efficient search.
    
    # For now, let's try a simpler randomized approach:
    # Assign random colors and check, with local search if needed.
    
    random.seed(12345 + n)
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Random initial assignment
        colors_try = [random.randint(0, 3) for _ in range(num_edges)]
        
        adj_try = [[[False] * n for _ in range(n)] for _ in range(4)]
        for idx, c in enumerate(colors_try):
            # Recover (i,j) from idx
            # idx = i*n - i*(i+1)//2 + (j-i-1)
            # This is tricky to reverse. Let's just rebuild from edge_list.
            pass
        
        # Rebuild adjacency from colors_try
        for pos, (i, j) in enumerate(edge_list):
            c = colors_try[pos]
            adj_try[c][i][j] = True
            adj_try[c][j][i] = True
        
        if not has_mono_c6(adj_try, n):
            return colors_try
    
    return None

# The above approach is too slow. Let's use a different strategy.

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # For n=17, we need a valid 4-coloring without mono C6.
    # Let's use a known construction based on finite geometry or group-based methods.
    
    # One approach: use the 17-cycle structure with 4 colors.
    # Another: use a probabilistic method with local search.
    
    # Let's try a local search approach:
    # Start with a random 4-coloring, then iteratively fix mono C6s by recoloring.
    
    def has_mono_c6(adj, n):
        """Check if any color class contains a C6."""
        for c in range(4):
            ac = adj[c]
            for v0 in range(n):
                for v1 in range(v0 + 1, n):
                    if not ac[v0][v1]:
                        continue
                    for v2 in range(n):
                        if v2 in (v0, v1) or not ac[v1][v2]:
                            continue
                        for v3 in range(n):
                            if v3 in (v0, v1, v2) or not ac[v2][v3]:
                                continue
                            for v4 in range(n):
                                if v4 in (v0, v1, v2, v3) or not ac[v3][v4]:
                                    continue
                                for v5 in range(n):
                                    if v5 in (v0, v1, v2, v3, v4) or not ac[v4][v5] or not ac[v5][v0]:
                                        continue
                                    if len(set([v0, v1, v2, v3, v4, v5])) == 6:
                                        return True, c, [v0, v1, v2, v3, v4, v5]
        return False, None, None
    
    def local_search(n, max_iter=100000):
        """Local search to find a valid 4-coloring."""
        num_edges = n * (n - 1) // 2
        edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Random initial coloring
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        adj = [[[False] * n for _ in range(n)] for _ in range(4)]
        for pos, (i, j) in enumerate(edge_list):
            c = colors[pos]
            adj[c][i][j] = True
            adj[c][j][i] = True
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            found, c, cycle = has_mono_c6(adj, n)
            if not found:
                return colors
            
            # Recolor a random edge in the cycle
            # Pick a random position in the cycle and change its edge color
            # The cycle is [v0, v1, v2, v3, v4, v5]
            # Edges: (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)
            edges_in_cycle = [(cycle[0], cycle[1]), (cycle[1], cycle[2]),
                             (cycle[2], cycle[3]), (cycle[3], cycle[4]),
                             (cycle[4], cycle[5]), (cycle[5], cycle[0])]
            
            # Pick a random edge to recolor
            edge_to_change = random.choice(edges_in_cycle)
            a, b = edge_to_change
            if a > b:
                a, b = b, a
            
            # Find the index of this edge
            pos = None
            for idx, (i, j) in enumerate(edge_list):
                if (i, j) == (a, b):
                    pos = idx
                    break
            
            if pos is None:
                continue
            
            old_c = colors[pos]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                new_c = (new_c + 1) % 4
            
            # Update
            adj[old_c][a][b] = False
            adj[old_c][b][a] = False
            adj[new_c][a][b] = True
            adj[new_c][b][a] = True
            colors[pos] = new_c
        
        return None
    
    # Try for n=17 first
    for target_n in [17, 18]:
        if time.time() > deadline:
            break
        
        result = local_search(target_n)
        if result is not None:
            return {'n': target_n, 'colors': result}
    
    # Fallback: return a small valid coloring
    # n=1 is trivially valid
    return {'n': 1, 'colors': []}

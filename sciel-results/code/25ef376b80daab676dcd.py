"""Rewritten program with union-find for bicolored cycle detection."""
import time
import random
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Build adjacency
adj = [[] for _ in range(N)]
for u, v in EDGES:
    adj[u].append(v)
    adj[v].append(u)

def check_bicolored_forest(colors):
    """Check that for every pair of colors, the induced subgraph is a forest using union-find."""
    # Group vertices by color
    color_to_verts = {}
    for i, c in enumerate(colors):
        if c not in color_to_verts:
            color_to_verts[c] = []
        color_to_verts[c].append(i)
    
    colors_used = sorted(color_to_verts.keys())
    
    for idx_a, a in enumerate(colors_used):
        for b in colors_used[idx_a+1:]:
            # Get vertices with color a or b
            verts = set(color_to_verts[a]) | set(color_to_verts[b])
            if len(verts) <= 1:
                continue
            
            # Union-find to detect cycles in the induced subgraph
            parent = {v: v for v in verts}
            rank = {v: 0 for v in verts}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False  # Cycle detected
                if rank[px] < rank[py]:
                    px, py = py, px
                parent[py] = px
                if rank[px] == rank[py]:
                    rank[px] += 1
                return True
            
            # Process edges in the induced subgraph
            for u in verts:
                for v in adj[u]:
                    if v in verts and u < v:
                        if not union(u, v):
                            return False  # Bicolored cycle found
    
    return True

def valid_coloring(colors):
    """Check all constraints: adjacent differ, and bicolored subgraphs are forests."""
    # Check adjacent vertices differ
    for u, v in EDGES:
        if colors[u] == colors[v]:
            return False
    # Check bicolored forests
    return check_bicolored_forest(colors)

def try_coloring_with_k_colors(k, seed, time_limit=3.0):
    """Try to find a valid coloring with at most k colors using greedy + backtracking."""
    start_time = time.time()
    rng = random.Random(seed)
    
    # Try multiple random orderings
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
        
        order = list(range(N))
        rng.shuffle(order)
        
        colors = [-1] * N
        success = True
        
        for v in order:
            forbidden = set()
            for nb in adj[v]:
                if colors[nb] != -1:
                    forbidden.add(colors[nb])
            
            assigned = False
            color_options = [c for c in range(k) if c not in forbidden]
            rng.shuffle(color_options)
            
            for c in color_options:
                colors[v] = c
                # Quick check: verify no bicolored cycle for pairs involving c
                valid = True
                for other in range(k):
                    if other == c:
                        continue
                    # Check subgraph induced by colors c and other
                    verts = set(i for i in range(N) if colors[i] in (c, other))
                    if len(verts) <= 1:
                        continue
                    parent = {x: x for x in verts}
                    rank = {x: 0 for x in verts}
                    
                    def find(x):
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x
                    
                    def union(x, y):
                        px, py = find(x), find(y)
                        if px == py:
                            return False
                        if rank[px] < rank[py]:
                            px, py = py, px
                        parent[py] = px
                        if rank[px] == rank[py]:
                            rank[px] += 1
                        return True
                    
                    for u in verts:
                        for w in adj[u]:
                            if w in verts and u < w:
                                if not union(u, w):
                                    valid = False
                                    break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    assigned = True
                    break
                else:
                    colors[v] = -1
            
            if not assigned:
                success = False
                break
        
        if success and valid_coloring(colors):
            return colors
    
    return None

def local_search_eliminate(colors, max_colors, time_limit=5.0):
    """Try to eliminate colors one by one using local search."""
    start_time = time.time()
    current = list(colors)
    current_max = max(current) + 1
    
    for target_color in range(current_max - 1, -1, -1):
        if time.time() - start_time > time_limit:
            break
        if target_color not in set(current):
            continue
        
        verts_to_recolor = [i for i in range(N) if current[i] == target_color]
        if not verts_to_recolor:
            continue
        
        # Try multiple orderings
        for _ in range(20):
            if time.time() - start_time > time_limit:
                break
            
            new_current = list(current)
            verts_copy = list(verts_to_recolor)
            random.shuffle(verts_copy)
            
            success = True
            for v in verts_copy:
                recolored = False
                for c in range(target_color):
                    new_current[v] = c
                    if valid_coloring(new_current):
                        recolored = True
                        break
                if not recolored:
                    success = False
                    break
            
            if success:
                current = new_current
                current_max = target_color
                break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    best_colors = list(range(N))  # Fallback: 50 colors
    best_count = N
    
    # Try to find colorings with few colors
    for k in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            
            colors = try_coloring_with_k_colors(k, seed=42 + attempt * 137 + k * 31)
            if colors is not None:
                count = len(set(colors))
                if count < best_count:
                    best_count = count
                    best_colors = list(colors)
                
                # Try to eliminate more colors
                if time.time() < deadline - 15:
                    improved = local_search_eliminate(colors, k, time_limit=3.0)
                    improved_count = len(set(improved))
                    if improved_count < best_count:
                        best_count = improved_count
                        best_colors = list(improved)
                
                found = True
                if best_count <= 3:
                    break
        
        if best_count <= 3:
            break
    
    # Final verification
    if valid_coloring(best_colors):
        result = [int(c) for c in best_colors]
        return result
    else:
        # Fallback: return 50 distinct colors
        return list(range(N))

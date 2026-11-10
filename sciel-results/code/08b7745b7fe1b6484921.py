"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def find_bicolored_cycle(coloring, c1, c2):
        """Find a bicolored cycle between colors c1 and c2. Returns list of edges in cycle or None."""
        # Build subgraph with only vertices colored c1 or c2
        parent = list(range(n))
        rank = [0] * n
        edge_parent = {}  # (u,v) -> parent edge info for cycle reconstruction
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return True
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return False
        
        # Track edges to reconstruct cycle
        edge_list = []
        for u, v in edges:
            if coloring[u] in (c1, c2) and coloring[v] in (c1, c2):
                if coloring[u] != coloring[v]:
                    edge_list.append((u, v))
        
        # Use DFS to find cycle
        visited = set()
        def dfs(node, prev_node):
            visited.add(node)
            for nb in adj[node]:
                if coloring[nb] not in (c1, c2) or coloring[node] == coloring[nb]:
                    continue
                if nb == prev_node:
                    continue
                if nb in visited:
                    # Found a cycle - reconstruct it
                    cycle = [node, nb]
                    curr = node
                    while curr != nb:
                        # Find next in cycle
                        for nn in adj[curr]:
                            if coloring[nn] in (c1, c2) and coloring[curr] != coloring[nn]:
                                if nn == prev_node:
                                    continue
                                # Check if this edge is in the cycle
                                break
                        # This is complex; just return the two vertices for now
                        break
                    return cycle
                else:
                    result = dfs(nb, node)
                    if result:
                        return result
            return None
        
        # Simpler approach: use union-find to detect cycle, then find it
        for u, v in edge_list:
            if union(u, v):
                # Cycle detected between u and v
                # Find the cycle by tracing back
                # For simplicity, just return the conflicting edge
                return [(u, v)]
        return None
    
    def check_valid(coloring):
        """Check if coloring is valid: proper and no bicolored cycles."""
        # Check proper coloring
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        # Check no bicolored cycles using union-find
        colors_used = set(coloring)
        for c1 in colors_used:
            for c2 in colors_used:
                if c1 >= c2:
                    continue
                parent = list(range(n))
                rank = [0] * n
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                def union(x, y):
                    px, py = find(x), find(y)
                    if px == py:
                        return True
                    if rank[px] < rank[py]:
                        px, py = py, px
                    parent[py] = px
                    if rank[px] == rank[py]:
                        rank[px] += 1
                    return False
                for u, v in edges:
                    if coloring[u] in (c1, c2) and coloring[v] in (c1, c2):
                        if coloring[u] != coloring[v]:
                            if union(u, v):
                                return False
        return True
    
    def get_cycle_vertices(coloring, c1, c2):
        """Get all vertices involved in bicolored subgraph for colors c1, c2."""
        return [i for i in range(n) if coloring[i] in (c1, c2)]
    
    def break_cycle_by_recolor(coloring, c1, c2, max_colors):
        """Break a bicolored cycle by recoloring a vertex in the cycle to a third color."""
        cycle_verts = get_cycle_vertices(coloring, c1, c2)
        if not cycle_verts:
            return coloring, False
        
        # Try to recolor each vertex in the bicolored subgraph
        random.shuffle(cycle_verts)
        for v in cycle_verts:
            neighbor_colors = set()
            for nb in adj[v]:
                neighbor_colors.add(coloring[nb])
            
            # Try existing colors first (excluding c1, c2 if they're neighbors)
            for new_c in range(max_colors):
                if new_c == coloring[v]:
                    continue
                if new_c in neighbor_colors:
                    continue
                # Check if this recoloring doesn't create new bicolored cycles
                new_coloring = coloring[:]
                new_coloring[v] = new_c
                if check_valid(new_coloring):
                    return new_coloring, True
            else:
                # Try a new color
                if max_colors < 50:
                    new_coloring = coloring[:]
                    new_coloring[v] = max_colors
                    if check_valid(new_coloring):
                        return new_coloring, True
        
        return coloring, False
    
    def iterative_local_search(coloring, max_colors, iterations=5000):
        """Iteratively find and break bicolored cycles."""
        coloring = coloring[:]
        for it in range(iterations):
            # Find any bicolored cycle
            colors_used = sorted(set(coloring))
            cycle_broken = False
            
            for c1 in colors_used:
                for c2 in colors_used:
                    if c1 >= c2:
                        continue
                    
                    # Check if there's a bicolored cycle
                    parent = list(range(n))
                    rank = [0] * n
                    def find(x):
                        while parent[x] != x:
                            parent[x] = parent[parent[x]]
                            x = parent[x]
                        return x
                    def union(x, y):
                        px, py = find(x), find(y)
                        if px == py:
                            return True
                        if rank[px] < rank[py]:
                            px, py = py, px
                        parent[py] = px
                        if rank[px] == rank[py]:
                            rank[px] += 1
                        return False
                    
                    has_cycle = False
                    for u, v in edges:
                        if coloring[u] in (c1, c2) and coloring[v] in (c1, c2):
                            if coloring[u] != coloring[v]:
                                if union(u, v):
                                    has_cycle = True
                                    break
                    
                    if has_cycle:
                        # Try to break this cycle
                        new_coloring, success = break_cycle_by_recolor(coloring, c1, c2, max_colors)
                        if success:
                            coloring = new_coloring
                            cycle_broken = True
                            break
                if cycle_broken:
                    break
            
            if not cycle_broken:
                break
        
        return coloring
    
    # Strategy: Start with a proper coloring using few colors, then iteratively break bicolored cycles
    best_coloring = None
    best_num_colors = n
    
    # Try different initial colorings
    for seed in range(10):
        random.seed(seed)
        
        # Greedy coloring with random vertex ordering
        order = list(range(n))
        random.shuffle(order)
        
        coloring = [-1] * n
        max_colors = 0
        
        for i in order:
            neighbor_colors = set()
            for nb in adj[i]:
                if coloring[nb] != -1:
                    neighbor_colors.add(coloring[nb])
            
            # Try to use existing colors
            for c in range(max_colors):
                if c not in neighbor_colors:
                    coloring[i] = c
                    break
            else:
                coloring[i] = max_colors
                max_colors += 1
        
        # Now iteratively break bicolored cycles
        coloring = iterative_local_search(coloring, max_colors, iterations=3000)
        
        if check_valid(coloring):
            num_colors = len(set(coloring))
            if best_coloring is None or num_colors < best_num_colors:
                best_num_colors = num_colors
                best_coloring = coloring[:]
    
    # Try to reduce colors further
    if best_coloring is not None:
        def try_merge_colors(coloring):
            """Try to merge two colors if it doesn't create bicolored cycles."""
            coloring = coloring[:]
            colors_used = sorted(set(coloring))
            changed = True
            while changed:
                changed = False
                for i in range(len(colors_used)):
                    for j in range(i+1, len(colors_used)):
                        c1, c2 = colors_used[i], colors_used[j]
                        # Try to merge c2 into c1
                        new_coloring = coloring[:]
                        for k in range(n):
                            if new_coloring[k] == c2:
                                new_coloring[k] = c1
                        if check_valid(new_coloring):
                            coloring = new_coloring
                            colors_used = sorted(set(coloring))
                            changed = True
                            break
                    if changed:
                        break
            return coloring
        
        best_coloring = try_merge_colors(best_coloring)
        best_num_colors = len(set(best_coloring))
    
    # Fallback if no valid coloring found
    if best_coloring is None:
        # Use a simple valid coloring: assign each vertex a unique color
        best_coloring = list(range(n))
        best_num_colors = n
    
    # Ensure proper Python ints in a list
    result = [int(c) for c in best_coloring]
    return result

# EVOLVE-BLOCK-END

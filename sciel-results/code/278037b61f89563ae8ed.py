"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolored_forest_full(colors):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    n = len(colors)
    color_to_verts = defaultdict(list)
    for i, c in enumerate(colors):
        color_to_verts[c].append(i)
    
    adj = _build_adj()
    colors_used = set(colors)
    for c1 in colors_used:
        for c2 in colors_used:
            if c1 >= c2:
                continue
            verts = set(color_to_verts[c1] + color_to_verts[c2])
            if len(verts) <= 1:
                continue
            parent = {v: v for v in verts}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False
                parent[px] = py
                return True
            for u in verts:
                for v in adj[u]:
                    if v in verts and u < v:
                        if not union(u, v):
                            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = P['n']
    adj = _build_adj()
    
    # Compute degeneracy order
    deg = {v: len(adj[v]) for v in range(n)}
    remaining = set(range(n))
    order = []
    while remaining:
        min_deg = min(deg[v] for v in remaining)
        candidates = [v for v in remaining if deg[v] == min_deg]
        v = min(candidates)
        order.append(v)
        remaining.remove(v)
        for u in adj[v]:
            if u in remaining:
                deg[u] -= 1
    
    pos_in_order = {v: i for i, v in enumerate(order)}
    
    # For each pair of colors, maintain a union-find structure
    # key: (c1, c2) with c1 < c2, value: parent dict for vertices in that subgraph
    
    def try_coloring(k):
        colors = [-1] * n
        # uf[(c1,c2)] = {vertex: parent} for the subgraph on colors c1,c2
        uf = {}
        
        def get_uf(c1, c2):
            key = (min(c1,c2), max(c1,c2))
            if key not in uf:
                uf[key] = {}
            return uf[key]
        
        def uf_find(parent, x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def uf_union(parent, x, y):
            px, py = uf_find(parent, x), uf_find(parent, y)
            if px == py:
                return False
            parent[px] = py
            return True
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            if idx == n:
                return colors[:]
            
            v = order[idx]
            forbidden = set()
            for u in adj[v]:
                if pos_in_order[u] < idx:
                    forbidden.add(colors[pos_in_order[u]])
            
            for c in range(k):
                if c in forbidden:
                    continue
                
                # Check if assigning color c to v creates a bicolored cycle
                # For each other color c' that exists among already-colored vertices,
                # check if adding v (color c) to the {c, c'} subgraph creates a cycle
                valid = True
                other_colors = set()
                for u in adj[v]:
                    if pos_in_order[u] < idx:
                        other_colors.add(colors[pos_in_order[u]])
                
                for c2 in other_colors:
                    if c2 == c:
                        continue
                    key = (min(c, c2), max(c, c2))
                    parent = uf.get(key, {})
                    
                    # v has color c. Its neighbors with color c2 are the new edges.
                    # A cycle is created if any two such neighbors are already connected
                    # in the {c, c2} subgraph (before adding v)
                    # Also, v connects to each such neighbor, so if v connects to u1 and u2,
                    # and u1 and u2 are connected, we have a cycle.
                    
                    neighbors_with_c2 = [u for u in adj[v] 
                                         if pos_in_order[u] < idx and colors[pos_in_order[u]] == c2]
                    
                    # Check if any pair of neighbors_with_c2 are connected in the existing subgraph
                    for i in range(len(neighbors_with_c2)):
                        u1 = neighbors_with_c2[i]
                        if u1 not in parent:
                            parent[u1] = u1
                        for j in range(i+1, len(neighbors_with_c2)):
                            u2 = neighbors_with_c2[j]
                            if u2 not in parent:
                                parent[u2] = u2
                            if uf_find(parent, u1) == uf_find(parent, u2):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Assign color c to v
                colors[idx] = c
                
                # Update union-find structures
                # For each neighbor u with color c2 != c, add edge (v, u) to {c, c2} subgraph
                for u in adj[v]:
                    u_pos = pos_in_order[u]
                    if u_pos < idx:
                        c2 = colors[u_pos]
                        if c2 == c:
                            continue
                        key = (min(c, c2), max(c, c2))
                        parent = uf.get(key, {})
                        if v not in parent:
                            parent[v] = v
                        if u not in parent:
                            parent[u] = u
                        # Union v and u
                        uf_union(parent, v, u)
                        uf[key] = parent
                
                result = backtrack(idx + 1)
                if result is not None:
                    return result
                
                # Backtrack: remove v from all union-find structures
                # This is tricky with union-find. Instead, let's use a different approach:
                # Save state before, restore after. But that's expensive.
                # Alternative: just recompute. Since we're backtracking, let's clear v's entries.
                # Actually, union-find doesn't support deletion easily.
                # Let's use a different strategy: for each pair, keep a list of edges and rebuild.
                # For now, let's just not support proper backtracking and restart.
                # This means we need to restart the search if we backtrack.
                # Let me restructure: instead of in-place union-find, use a check function.
                
                colors[idx] = -1
            
            return None
        
        return backtrack(0)
    
    # The above approach has issues with backtracking union-find.
    # Let me use a simpler but correct approach: for each candidate assignment,
    # do a targeted check without modifying global state.
    
    def try_coloring_v2(k):
        colors = [-1] * n
        pos_in_order = {v: i for i, v in enumerate(order)}
        
        def check_assignment(v, c, colors_so_far):
            """Check if assigning color c to vertex v is valid given colors_so_far."""
            # For each other color c' present in already-colored vertices,
            # check if the {c, c'} subgraph would have a cycle
            other_colors = set()
            for u in adj[v]:
                u_pos = pos_in_order[u]
                if u_pos < len(colors_so_far):
                    other_colors.add(colors_so_far[u_pos])
            
            for c2 in other_colors:
                if c2 == c:
                    continue
                # Build the {c, c2} subgraph including v
                # Vertices: all already-colored vertices with color c or c2, plus v
                # Edges: all edges between these vertices in the original graph
                verts = set()
                verts.add(v)
                for i in range(len(colors_so_far)):
                    if colors_so_far[i] == c or colors_so_far[i] == c2:
                        verts.add(order[i])
                
                if len(verts) <= 1:
                    continue
                
                # Union-find on this subgraph
                parent = {v: v for v in verts}
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                def union(x, y):
                    px, py = find(x), find(y)
                    if px == py:
                        return False
                    parent[px] = py
                    return True
                
                for u in verts:
                    for w in adj[u]:
                        if w in verts and u < w:
                            if not union(u, w):
                                return False
            return True
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            if idx == n:
                return colors[:]
            
            v = order[idx]
            # Get colors of already-colored neighbors
            neighbor_colors = set()
            for u in adj[v]:
                u_pos = pos_in_order[u]
                if u_pos < idx:
                    neighbor_colors.add(colors[u_pos])
            
            for c in range(k):
                if c in neighbor_colors:
                    continue
                if check_assignment(v, c, colors[:idx]):
                    colors[idx] = c
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
                    colors[idx] = -1
            return None
        
        return backtrack(0)
    
    for k in range(2, 12):
        if time.time() > deadline:
            break
        result = try_coloring_v2(k)
        if result is not None:
            final_colors = [0] * n
            for i, v in enumerate(order):
                final_colors[v] = result[i]
            # Verify
            assert _check_bicolored_forest_full(final_colors)
            for u, v in P['edges']:
                assert final_colors[u] != final_colors[v]
            color_map = {}
            remapped = []
            for c in final_colors:
                if c not in color_map:
                    color_map[c] = len(color_map)
                remapped.append(color_map[c])
            return [int(x) for x in remapped]
    
    # Fallback: greedy with many colors
    final_colors = [0] * n
    for v in order:
        neighbor_colors = set()
        for u in adj[v]:
            if final_colors[u] != 0 or u != v:
                if final_colors[u] != -1:
                    neighbor_colors.add(final_colors[u])
        c = 0
        while c in neighbor_colors:
            c += 1
        final_colors[v] = c
    
    color_map = {}
    remapped = []
    for c in final_colors:
        if c not in color_map:
            color_map[c] = len(color_map)
        remapped.append(color_map[c])
    return [int(x) for x in remapped]
# EVOLVE-BLOCK-END

import time
import random
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _count_cycles_incremental(colors, edges, adj, k):
    """Count total bicolored cycles using union-find. Returns number of cycles."""
    n = len(colors)
    total_cycles = 0
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            verts = set()
            for i in range(n):
                if colors[i] == c1 or colors[i] == c2:
                    verts.add(i)
            if len(verts) < 3:
                continue
            parent = {x: x for x in verts}
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return True
                parent[ry] = rx
                return False
            cycles = 0
            for u, v in edges:
                if u in verts and v in verts:
                    if union(u, v):
                        cycles += 1
            total_cycles += cycles
    return total_cycles

def _is_proper(colors, edges):
    """Check if coloring is proper (no adjacent vertices share a color)."""
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    return True

def _would_create_cycle(colors, v, new_c, edges, adj, k):
    """Check if assigning new_c to vertex v would create a bicolored cycle."""
    n = len(colors)
    # For each other color c2, check if adding v with new_c creates a cycle in (new_c, c2)
    for c2 in range(k):
        if c2 == new_c:
            continue
        # Collect vertices with color new_c or c2 (excluding v for now)
        verts = set()
        for i in range(n):
            if i == v:
                continue
            if colors[i] == new_c or colors[i] == c2:
                verts.add(i)
        # Add neighbors of v that have color c2 or new_c
        for nb in adj[v]:
            if colors[nb] == new_c or colors[nb] == c2:
                verts.add(nb)
        
        if len(verts) < 2:
            continue
        
        # Union-find on induced subgraph including v
        parent = {x: x for x in verts}
        parent[v] = v
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return True
            parent[ry] = rx
            return False
        
        # First, union all existing edges in the subgraph
        for u, w in edges:
            if u in verts and w in verts and u != v and w != v:
                union(u, w)
        
        # Now add edges from v to its neighbors in the subgraph
        for nb in adj[v]:
            if nb in verts:
                if union(v, nb):
                    # This edge didn't create a cycle
                    pass
                else:
                    # This edge creates a cycle!
                    return True
    
    return False

def _ac3_check(colors, v, c, edges, adj, k):
    """Arc consistency check: verify that assigning c to v doesn't immediately
    make any neighbor uncolorable or create a bicolored cycle."""
    n = len(colors)
    
    # Check proper coloring for neighbors
    for nb in adj[v]:
        if colors[nb] == c:
            return False
    
    # Check if this creates a bicolored cycle
    if _would_create_cycle(colors, v, c, edges, adj, k):
        return False
    
    # Forward checking: for each neighbor, check if it still has a valid color
    for nb in adj[v]:
        if colors[nb] == -1:
            # Neighbor is uncolored; check if it has at least one valid color
            valid = False
            for c2 in range(k):
                if c2 == c:
                    continue
                # Check if c2 is valid for nb
                proper = True
                for nb2 in adj[nb]:
                    if nb2 == v:
                        if c == c2:
                            proper = False
                            break
                    elif colors[nb2] == c2:
                        proper = False
                        break
                if proper:
                    valid = True
                    break
            if not valid:
                return False
    
    return True

def _backtrack_csp(k, edges, adj, n, deadline):
    """Backtracking CSP solver with forward checking."""
    colors = [-1] * n
    
    # Order vertices by degree (descending) for better pruning
    order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    # For each vertex, precompute which colors are initially possible
    domains = {}
    for v in order:
        domains[v] = list(range(k))
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        
        if idx == len(order):
            return True
        
        v = order[idx]
        
        # Try each color in the domain
        for c in domains[v]:
            # Check if c is still valid (considering already-assigned neighbors)
            valid = True
            for nb in adj[v]:
                if colors[nb] == c:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Tentatively assign
            colors[v] = c
            
            # Forward checking: verify all unassigned neighbors still have options
            fc_ok = True
            for nb in adj[v]:
                if colors[nb] == -1:
                    has_option = False
                    for c2 in range(k):
                        if c2 == c:
                            continue
                        # Check if c2 is valid for nb given current assignments
                        ok = True
                        for nb2 in adj[nb]:
                            if nb2 == v:
                                if c == c2:
                                    ok = False
                                    break
                            elif colors[nb2] == c2:
                                ok = False
                                break
                        if ok:
                            has_option = True
                            break
                    if not has_option:
                        fc_ok = False
                        break
            
            if not fc_ok:
                colors[v] = -1
                continue
            
            # Check for bicolored cycles involving v
            if _would_create_cycle(colors, v, c, edges, adj, k):
                colors[v] = -1
                continue
            
            # Recurse
            if backtrack(idx + 1):
                return True
            
            colors[v] = -1
        
        return False
    
    if backtrack(0):
        return [int(c) for c in colors]
    
    return None

def _local_search_polish(colors, k, edges, adj, n, deadline, max_steps=2000):
    """Polish a valid coloring by trying to reduce the number of distinct colors used."""
    # Count distinct colors
    used_colors = set(colors)
    if len(used_colors) <= k:
        return colors
    
    # Try to merge colors: recolor some vertices of one color to another
    color_list = sorted(used_colors)
    for i in range(len(color_list)):
        for j in range(i + 1, len(color_list)):
            c1, c2 = color_list[i], color_list[j]
            # Try to recolor all c1 vertices to c2
            new_colors = colors[:]
            changed = False
            for v in range(n):
                if new_colors[v] == c1:
                    # Check if c2 is valid for v
                    valid = True
                    for nb in adj[v]:
                        if new_colors[nb] == c2:
                            valid = False
                            break
                    if valid:
                        new_colors[v] = c2
                        changed = True
            
            if not changed:
                continue
            
            # Check if new coloring is valid
            if _is_proper(new_colors, edges):
                cycles = _count_cycles_incremental(new_colors, edges, adj, k)
                if cycles == 0:
                    return new_colors
    
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s margin before 90s limit
    n = P['n']
    edges = P['edges']
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Try k=3 first with CSP backtracking
    for k in [3, 4]:
        if time.time() > deadline:
            break
        
        result = _backtrack_csp(k, edges, adj, n, deadline)
        if result is not None:
            # Verify
            if _is_proper(result, edges):
                cycles = _count_cycles_incremental(result, edges, adj, k)
                if cycles == 0:
                    return result
    
    # Fallback to local search if CSP fails
    for k in [3, 4]:
        if time.time() > deadline:
            break
        
        for restart in range(20):
            if time.time() > deadline:
                break
            
            colors = [random.randint(0, k - 1) for _ in range(n)]
            
            # Local search to fix violations
            for step in range(3000):
                if time.time() > deadline:
                    break
                
                if not _is_proper(colors, edges):
                    # Fix improper edge
                    for u, v in edges:
                        if colors[u] == colors[v]:
                            valid = [c for c in range(k) if not any(colors[nb] == c for nb in adj[u])]
                            if valid:
                                colors[u] = random.choice(valid)
                            break
                    continue
                
                cycles = _count_cycles_incremental(colors, edges, adj, k)
                if cycles == 0:
                    return [int(c) for c in colors]
                
                # Random move
                v = random.randint(0, n - 1)
                valid = [c for c in range(k) if not any(colors[nb] == c for nb in adj[v])]
                if valid:
                    colors[v] = random.choice(valid)
    
    # Last resort
    return [i % 4 for i in range(n)]

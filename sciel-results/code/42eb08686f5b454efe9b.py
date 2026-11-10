import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_valid(colors, edges, k):
    """Check proper coloring and count bicolored cycles. Returns (num_cycles, is_proper)."""
    n = len(colors)
    # Check proper
    proper = True
    for u, v in edges:
        if colors[u] == colors[v]:
            proper = False
            break
    
    # Count bicolored cycles using union-find per color pair
    total_cycles = 0
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            # Collect vertices with color c1 or c2
            verts = set()
            for i in range(n):
                if colors[i] == c1 or colors[i] == c2:
                    verts.add(i)
            if len(verts) < 3:
                continue
            # Union-find on induced subgraph
            parent = {}
            for v in verts:
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
            cycles = 0
            for u, v in edges:
                if u in verts and v in verts:
                    if union(u, v):
                        cycles += 1
            total_cycles += cycles
    
    return total_cycles, proper

def _find_cycle_in_pair(colors, c1, c2, edges, adj):
    """Find a bicolored cycle between colors c1 and c2. Returns list of vertices or None."""
    n = len(colors)
    verts = set()
    for i in range(n):
        if colors[i] == c1 or colors[i] == c2:
            verts.add(i)
    if len(verts) < 3:
        return None
    
    # Build subgraph adjacency
    sub_adj = defaultdict(list)
    for u, v in edges:
        if u in verts and v in verts:
            sub_adj[u].append(v)
            sub_adj[v].append(u)
    
    # DFS to find a cycle
    visited = set()
    parent = {}
    
    def dfs(u, p):
        visited.add(u)
        for v in sub_adj[u]:
            if v == p:
                continue
            if v in visited:
                cycle = [v]
                cur = u
                while cur != v:
                    cycle.append(cur)
                    cur = parent[cur]
                return cycle
            parent[v] = u
            res = dfs(v, u)
            if res:
                return res
        return None
    
    for v in sorted(verts):
        if v not in visited:
            parent[v] = -1
            res = dfs(v, -1)
            if res:
                return res
    return None

def _greedy_color(k, edges, adj, n):
    """Greedy coloring that tries to avoid creating bicolored cycles."""
    colors = [-1] * n
    order = list(range(n))
    random.shuffle(order)
    
    for v in order:
        # Try each color, pick one that doesn't create improper edge and minimizes cycles
        best_c = -1
        best_cost = float('inf')
        for c in range(k):
            # Check proper
            if any(colors[nb] == c for nb in adj[v]):
                continue
            # Tentatively assign
            colors[v] = c
            # Quick check: count cycles involving this vertex's color pairs
            cost = 0
            for c2 in range(k):
                if c2 == c:
                    continue
                # Check if adding v creates a cycle in (c, c2) subgraph
                verts = set()
                for i in range(n):
                    if colors[i] == c or colors[i] == c2:
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
                for u, v2 in edges:
                    if u in verts and v2 in verts:
                        if union(u, v2):
                            cost += 1
            if cost < best_cost:
                best_cost = cost
                best_c = c
        if best_c == -1:
            # No valid color, pick one that's proper
            for c in range(k):
                if not any(colors[nb] == c for nb in adj[v]):
                    best_c = c
                    break
            if best_c == -1:
                best_c = random.randint(0, k - 1)
        colors[v] = best_c
    
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s limit
    n = P['n']
    edges = P['edges']
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    best_result = None
    best_k = 50
    
    for k in [3, 4]:
        if time.time() > deadline:
            break
        
        max_restarts = 30 if k == 3 else 15
        
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Use greedy initialization for first few restarts, random for rest
            if r < 5:
                colors = _greedy_color(k, edges, adj, n)
            else:
                colors = [random.randint(0, k - 1) for _ in range(n)]
            
            # Check initial validity
            cost, proper = _check_valid(colors, edges, k)
            if cost == 0 and proper:
                result = [int(c) for c in colors]
                if k < best_k:
                    best_k = k
                    best_result = result
                return result
            
            # Local search
            for step in range(5000):
                if time.time() > deadline:
                    break
                
                cost, proper = _check_valid(colors, edges, k)
                
                if cost == 0 and proper:
                    result = [int(c) for c in colors]
                    if k < best_k:
                        best_k = k
                        best_result = result
                    return result
                
                # If improper, fix improper vertices first
                if not proper:
                    # Find an improper edge
                    improper_verts = set()
                    for u, v in edges:
                        if colors[u] == colors[v]:
                            improper_verts.add(u)
                            improper_verts.add(v)
                    v = random.choice(list(improper_verts))
                    valid_colors = [c for c in range(k) if not any(colors[nb] == c for nb in adj[v])]
                    if valid_colors:
                        colors[v] = random.choice(valid_colors)
                    else:
                        # All neighbors use all k colors, just pick random
                        colors[v] = random.randint(0, k - 1)
                    continue
                
                # Find a specific bicolored cycle to break
                cycle = None
                cycle_pair = None
                for c1 in range(k):
                    for c2 in range(c1 + 1, k):
                        cycle = _find_cycle_in_pair(colors, c1, c2, edges, adj)
                        if cycle:
                            cycle_pair = (c1, c2)
                            break
                    if cycle:
                        break
                
                if not cycle:
                    # No cycle found via DFS (shouldn't happen if cost > 0), try random move
                    v = random.randint(0, n - 1)
                    valid_colors = [c for c in range(k) if not any(colors[nb] == c for nb in adj[v])]
                    if valid_colors:
                        colors[v] = random.choice(valid_colors)
                    continue
                
                # Try to break the cycle: for each vertex in cycle, try each valid color
                best_move = None
                best_new_cost = cost
                for v in cycle:
                    for new_c in range(k):
                        if new_c == colors[v]:
                            continue
                        if any(colors[nb] == new_c for nb in adj[v]):
                            continue
                        old = colors[v]
                        colors[v] = new_c
                        new_cost, new_proper = _check_valid(colors, edges, k)
                        colors[v] = old
                        if new_proper and new_cost < best_new_cost:
                            best_new_cost = new_cost
                            best_move = (v, new_c)
                
                if best_move:
                    v, new_c = best_move
                    colors[v] = new_c
                else:
                    # No improving move found, try a random perturbation
                    # Recolor a random vertex in the cycle to a valid color
                    v = random.choice(cycle)
                    valid_colors = [c for c in range(k) if not any(colors[nb] == c for nb in adj[v])]
                    if valid_colors:
                        colors[v] = random.choice(valid_colors)
                    else:
                        v = random.randint(0, n - 1)
                        valid_colors = [c for c in range(k) if not any(colors[nb] == c for nb in adj[v])]
                        if valid_colors:
                            colors[v] = random.choice(valid_colors)
    
    # Fallback
    if best_result:
        return [int(c) for c in best_result]
    
    # Last resort: 4-coloring by i%4 (may not be valid but returns something)
    return [i % 4 for i in range(n)]

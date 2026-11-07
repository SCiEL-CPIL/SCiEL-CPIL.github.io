"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    adj = [[] for _ in range(P['n'])]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _component_has_cycle(start, coloring, adj, color_a, color_b):
    """Check if the component containing 'start' in the {color_a, color_b} subgraph has a cycle.
    Returns True if there's a cycle, False otherwise."""
    visited = set()
    queue = deque([start])
    visited.add(start)
    comp_vertices = []
    while queue:
        node = queue.popleft()
        comp_vertices.append(node)
        for nb in adj[node]:
            if coloring[nb] not in (color_a, color_b) or nb in visited:
                continue
            visited.add(nb)
            queue.append(nb)
    
    edge_count = 0
    for v in comp_vertices:
        for nb in adj[v]:
            if nb in visited and coloring[nb] in (color_a, color_b) and nb > v:
                edge_count += 1
    
    return edge_count >= len(comp_vertices)

def _check_bicolored_cycles(coloring, adj, max_colors):
    """Check if any pair of colors induces a cycle. Returns True if valid (no bicolored cycles)."""
    n = P['n']
    colors_used = set(coloring[i] for i in range(n) if coloring[i] >= 0)
    for a in colors_used:
        for b in colors_used:
            if a >= b:
                continue
            visited = set()
            for start in range(n):
                if coloring[start] not in (a, b) or start in visited:
                    continue
                if _component_has_cycle(start, coloring, adj, a, b):
                    return False
                queue = deque([start])
                visited.add(start)
                while queue:
                    node = queue.popleft()
                    for nb in adj[node]:
                        if coloring[nb] not in (a, b) or nb in visited:
                            continue
                        visited.add(nb)
                        queue.append(nb)
    return True

def _check_incremental(coloring, adj, vertex, color, max_colors):
    """Check if assigning 'color' to 'vertex' creates a bicolored cycle.
    Only checks pairs involving the new color assignment."""
    n = P['n']
    for c in range(max_colors):
        if c == color:
            continue
        has_c = False
        for i in range(n):
            if coloring[i] == c:
                has_c = True
                break
        if not has_c:
            continue
        if _component_has_cycle(vertex, coloring, adj, color, c):
            return False
    return True

def _backtrack_coloring(adj, max_colors, deadline, order):
    n = P['n']
    coloring = [-1] * n
    
    def bt(idx):
        if time.time() > deadline:
            return None
        if idx == n:
            return coloring[:]
        
        v = order[idx]
        for c in range(max_colors):
            valid = True
            for nb in adj[v]:
                if coloring[nb] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            coloring[v] = c
            if _check_incremental(coloring, adj, v, c, max_colors):
                result = bt(idx + 1)
                if result is not None:
                    return result
            coloring[v] = -1
        
        return None
    
    return bt(0)

def _greedy_coloring(adj, max_colors):
    """Try a greedy approach to get an initial valid coloring."""
    n = P['n']
    coloring = [-1] * n
    order = list(range(n))
    
    for v in order:
        used_colors = set()
        for nb in adj[v]:
            if coloring[nb] >= 0:
                used_colors.add(coloring[nb])
        for c in range(max_colors):
            if c not in used_colors:
                coloring[v] = c
                break
        if coloring[v] < 0:
            return None
    
    if _check_bicolored_cycles(coloring, adj, max_colors):
        return coloring
    return None

def construct():
    deadline = time.time() + 85  # Safety margin before 90s limit
    adj = _build_adj()
    n = P['n']
    
    for mc in [3, 4, 5, 6]:
        result = _greedy_coloring(adj, mc)
        if result is not None:
            return [int(x) for x in result]
        
        if time.time() > deadline:
            break
        
        orderings = [
            list(range(n)),
            list(range(n-1, -1, -1)),
        ]
        snake = []
        for i in range(0, n, 2):
            if i + 1 < n:
                snake.extend([i, i+1])
            else:
                snake.append(i)
        orderings.append(snake)
        
        for order in orderings:
            if time.time() > deadline:
                break
            result = _backtrack_coloring(adj, mc, deadline, order)
            if result is not None:
                if _check_bicolored_cycles(result, adj, mc):
                    return [int(x) for x in result]
    
    return [int(i) for i in range(P['n'])]
# EVOLVE-BLOCK-END

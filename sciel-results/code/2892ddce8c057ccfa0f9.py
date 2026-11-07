"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolor_forest(coloring, k):
    """Check that for every pair of colors (a,b), the induced subgraph is a forest."""
    n = P['n']
    color_groups = defaultdict(list)
    for i in range(n):
        color_groups[coloring[i]].append(i)
    
    colors_used = list(color_groups.keys())
    for i in range(len(colors_used)):
        for j in range(i + 1, len(colors_used)):
            a, b = colors_used[i], colors_used[j]
            va = set(color_groups[a])
            vb = set(color_groups[b])
            vset = va | vb
            parent = {v: v for v in vset}
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
            for u in vset:
                for v in _adj[u]:
                    if v in vset and v > u:
                        if not union(u, v):
                            return False
    return True

def _is_valid(coloring, k):
    """Check all constraints: adjacent differ, and bicolored subgraphs are forests."""
    n = P['n']
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    return _check_bicolor_forest(coloring, k)

def _find_bicolored_cycle_vertex(coloring, k):
    """Find a vertex involved in a bicolored cycle, or None if none exists."""
    n = P['n']
    color_groups = defaultdict(list)
    for i in range(n):
        color_groups[coloring[i]].append(i)
    
    colors_used = list(color_groups.keys())
    for i in range(len(colors_used)):
        for j in range(i + 1, len(colors_used)):
            a, b = colors_used[i], colors_used[j]
            va = set(color_groups[a])
            vb = set(color_groups[b])
            vset = va | vb
            parent = {v: v for v in vset}
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
            for u in vset:
                for v in _adj[u]:
                    if v in vset and v > u:
                        if not union(u, v):
                            # Found a cycle; return one of the vertices
                            return u
    return None

def construct():
    deadline = time.time() + 85
    adj = _build_adj()
    n = P['n']
    adj_sets = [set(adj[i]) for i in range(n)]
    
    # Try increasing number of colors
    for k in range(2, 12):
        if time.time() > deadline:
            break
        
        # Use local search with proper bicolored-cycle detection
        max_restarts = 1000
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initial coloring with k colors
            random.seed(restart * 104729 + k * 31)
            coloring = [random.randint(0, k - 1) for _ in range(n)]
            
            # Local search: try to fix violations
            improved = True
            stall_count = 0
            max_stall = 100
            while improved and time.time() < deadline:
                improved = False
                
                # Check adjacency constraint
                adj_conflict = None
                for u, v in P['edges']:
                    if coloring[u] == coloring[v]:
                        adj_conflict = (u, v)
                        break
                
                if adj_conflict:
                    # Fix adjacency conflict
                    u, v = adj_conflict
                    # Change the one with more flexibility
                    for vertex in [u, v]:
                        for new_c in range(k):
                            if new_c == coloring[vertex]:
                                continue
                            ok = True
                            for nb in adj[vertex]:
                                if coloring[nb] == new_c:
                                    ok = False
                                    break
                            if ok:
                                coloring[vertex] = new_c
                                improved = True
                                stall_count = 0
                                break
                        if improved:
                            break
                    if not improved:
                        # Force change
                        vertex = u
                        new_c = (coloring[vertex] + 1) % k
                        coloring[vertex] = new_c
                        improved = True
                        stall_count = 0
                    continue
                
                # Check bicolored cycle
                cycle_vertex = _find_bicolored_cycle_vertex(coloring, k)
                if cycle_vertex is None:
                    # Valid coloring found!
                    if _is_valid(coloring, k):
                        return [int(c) for c in coloring]
                    break
                
                # Try to fix the bicolored cycle by changing cycle_vertex
                cv = cycle_vertex
                old_color = coloring[cv]
                found_change = False
                for new_c in range(k):
                    if new_c == old_color:
                        continue
                    # Check adjacency
                    ok = True
                    for nb in adj[cv]:
                        if coloring[nb] == new_c:
                            ok = False
                            break
                    if not ok:
                        continue
                    # Try the change and see if it helps
                    coloring[cv] = new_c
                    # Quick check: is there still a bicolored cycle?
                    still_bad = _find_bicolored_cycle_vertex(coloring, k)
                    if still_bad is None:
                        improved = True
                        stall_count = 0
                        found_change = True
                        break
                    else:
                        # Revert
                        coloring[cv] = old_color
                
                if not found_change:
                    # Try random changes near the cycle vertex
                    candidates = [cv] + list(adj[cv])
                    random.shuffle(candidates)
                    for vertex in candidates:
                        new_c = random.randint(0, k - 1)
                        if new_c == coloring[vertex]:
                            continue
                        ok = True
                        for nb in adj[vertex]:
                            if coloring[nb] == new_c:
                                ok = False
                                break
                        if ok:
                            old = coloring[vertex]
                            coloring[vertex] = new_c
                            # Check if it helped
                            still_bad = _find_bicolored_cycle_vertex(coloring, k)
                            if still_bad is None:
                                improved = True
                                stall_count = 0
                                found_change = True
                                break
                            else:
                                coloring[vertex] = old
                    
                    if not found_change:
                        # Random perturbation
                        if stall_count < max_stall:
                            vertex = random.randint(0, n - 1)
                            new_c = random.randint(0, k - 1)
                            if new_c != coloring[vertex]:
                                ok = True
                                for nb in adj[vertex]:
                                    if coloring[nb] == new_c:
                                        ok = False
                                        break
                                if ok:
                                    coloring[vertex] = new_c
                                    improved = True
                        else:
                            break
                
                if not improved:
                    stall_count += 1
                    if stall_count > max_stall:
                        break
    
    # Fallback: return a valid 50-color solution
    return [int(i) for i in range(n)]

# Precompute adjacency for module-level use
_adj = _build_adj()
# EVOLVE-BLOCK-END

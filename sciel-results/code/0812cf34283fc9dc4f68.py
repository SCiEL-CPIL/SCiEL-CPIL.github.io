"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _is_valid_assignment(colors, adj, n, deadline):
    """
    Check if the current partial or full coloring is valid.
    1. Proper coloring: adjacent vertices have different colors.
    2. Bicolored forest: for any pair of colors (c1, c2), the induced subgraph is a forest.
    """
    if time.time() > deadline:
        return False
        
    # 1. Proper coloring check
    for u, v in P['edges']:
        if colors[u] != -1 and colors[v] != -1:
            if colors[u] == colors[v]:
                return False

    # 2. Bicolored forest check
    # We only need to check pairs of colors that actually exist in the assignment
    colors_present = set(c for c in colors if c != -1)
    
    for c1 in colors_present:
        for c2 in colors_present:
            if c1 >= c2:
                continue
            
            # Check if subgraph induced by c1 and c2 is a forest
            # A graph is a forest iff it has no cycles.
            # We can check this by counting edges and vertices in components, 
            # or by using Union-Find.
            
            vertices_in_subgraph = []
            for i in range(n):
                if colors[i] == c1 or colors[i] == c2:
                    vertices_in_subgraph.append(i)
            
            if not vertices_in_subgraph:
                continue
                
            # Union-Find to detect cycle
            parent = {v: v for v in vertices_in_subgraph}
            rank = {v: 0 for v in vertices_in_subgraph}
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False # Cycle found
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            # Iterate through edges
            for u, v in P['edges']:
                # Check if edge is within the subgraph
                if (colors[u] == c1 or colors[u] == c2) and (colors[v] == c1 or colors[v] == c2):
                    if not union(u, v):
                        return False
    return True

def _backtrack_color(k, adj, n, deadline, order):
    """
    Backtracking search for k-coloring.
    Uses dynamic ordering based on degree/constraints if possible, 
    but here we use the provided static order for speed and predictability.
    """
    colors = [-1] * n
    
    # Precompute neighbors for faster access
    # adj is already a list of lists
    
    def is_feasible(v, c):
        """
        Check if assigning color c to vertex v is feasible given current colors.
        """
        if time.time() > deadline:
            return False
            
        # Check proper coloring
        for u in adj[v]:
            if colors[u] == c:
                return False
        
        # Check bicolored cycle constraint
        # Only need to check against colors of neighbors
        neighbor_colors = set()
        for u in adj[v]:
            if colors[u] != -1:
                neighbor_colors.add(colors[u])
        
        for c2 in neighbor_colors:
            if c2 == c:
                continue
            
            # Check if adding v with color c creates a cycle in subgraph {c, c2}
            # This happens if v has at least 2 neighbors in color c2 
            # that are connected in the subgraph induced by {c, c2} (excluding v).
            
            neighbors_in_c2 = [u for u in adj[v] if colors[u] == c2]
            if len(neighbors_in_c2) < 2:
                continue
            
            # BFS/DFS from the first neighbor in c2 to see if we reach another neighbor in c2
            # Restricted to vertices with colors c or c2.
            start = neighbors_in_c2[0]
            target_set = set(neighbors_in_c2[1:])
            
            visited = set()
            queue = deque([start])
            visited.add(start)
            
            while queue:
                curr = queue.popleft()
                for nxt in adj[curr]:
                    # Only traverse if color is c or c2
                    # Note: colors[curr] is either c (if curr==v, but we start from neighbor) 
                    # or c2. Wait, we are checking the subgraph *excluding* v?
                    # The cycle would be: v -> u1 -> ... -> u2 -> v.
                    # The path u1 -> ... -> u2 must exist in the graph induced by {c, c2} \ {v}.
                    # So we should NOT traverse through v.
                    # Also, we should only traverse through vertices with color c or c2.
                    
                    if nxt == v:
                        continue
                    if colors[nxt] != c and colors[nxt] != c2:
                        continue
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    if nxt in target_set:
                        return False # Cycle found
                    queue.append(nxt)
                        
        return True

    def bt(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        v = order[idx]
        
        # Try colors 0 to k-1
        # Optimization: try colors that appear in neighbors first? Or random?
        # Let's stick to 0..k-1 for determinism, or shuffle if we want randomness.
        # Deterministic is safer for reproducibility.
        for c in range(k):
            if is_feasible(v, c):
                colors[v] = c
                if bt(idx + 1):
                    return True
                colors[v] = -1
        return False

    if bt(0):
        return colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin
    n = P['n']
    adj = _build_adjacency()
    
    # Determine a good vertex ordering.
    # Since all degrees are 4, we can use a BFS order from vertex 0 to group connected parts.
    # Or simply 0..49.
    # Let's try to order by "centrality" or just use the natural order.
    # A good heuristic is to order vertices such that they have many neighbors already colored.
    # However, dynamic ordering in backtracking is complex. 
    # Static ordering: sort by index is fine for this small graph.
    # Let's try sorting by degree (all 4) then by index.
    order = list(range(n))
    
    best_result = None
    best_score = n + 1
    
    # Try increasing k
    for k in range(3, 11):
        if time.time() > deadline - 2.0:
            break
            
        # Try a few different orderings to avoid getting stuck in bad local minima of the search tree
        # Ordering 1: Natural
        res = _backtrack_color(k, adj, n, deadline, order)
        if res is not None:
            # Verify
            if _is_valid_assignment(res, adj, n, deadline):
                distinct = len(set(res))
                if distinct < best_score:
                    best_score = distinct
                    best_result = res[:]
                break
        
        # If natural order failed, try reversed
        if best_result is None or best_score > k:
             res = _backtrack_color(k, adj, n, deadline, list(reversed(order)))
             if res is not None:
                 if _is_valid_assignment(res, adj, n, deadline):
                     distinct = len(set(res))
                     if distinct < best_score:
                         best_score = distinct
                         best_result = res[:]
                     break

    # If we found a solution, try to improve (reduce colors) via local search
    if best_result is not None:
        # Simple local search: try to recolor vertices to use fewer colors
        # This is tricky with the bicolored constraint. 
        # Given the small size, backtracking usually finds the optimal or near-optimal quickly.
        # We can try k-1 if we found k.
        if best_score > 3:
            k_try = best_score - 1
            if time.time() < deadline - 5.0:
                res = _backtrack_color(k_try, adj, n, deadline, order)
                if res is not None:
                    if _is_valid_assignment(res, adj, n, deadline):
                        distinct = len(set(res))
                        if distinct < best_score:
                            best_score = distinct
                            best_result = res[:]

    # Fallback
    if best_result is None:
        # Should not happen for k up to 10 on this graph, but just in case
        best_result = list(range(n))
        best_score = n

    return [int(c) for c in best_result]
# EVOLVE-BLOCK-END

import time
import random
from collections import defaultdict
import sys
sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_validity(colors, edges, n):
    """Full validity check."""
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    color_groups = defaultdict(list)
    for i, c in enumerate(colors):
        color_groups[c].append(i)
    distinct_colors = sorted(color_groups.keys())
    for i in range(len(distinct_colors)):
        for j in range(i+1, len(distinct_colors)):
            c1, c2 = distinct_colors[i], distinct_colors[j]
            sub_vertices = set(color_groups[c1]).union(color_groups[c2])
            if len(sub_vertices) < 3:
                continue
            parent = {v: v for v in sub_vertices}
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
                if u in sub_vertices and v in sub_vertices:
                    if not union(u, v):
                        return False
    return True

def _count_bicolored_cycles(colors, edges, n):
    """Count total number of bicolored cycles (excess edges in each pair subgraph)."""
    color_groups = defaultdict(list)
    for i, c in enumerate(colors):
        color_groups[c].append(i)
    distinct_colors = sorted(color_groups.keys())
    total_cycles = 0
    for i in range(len(distinct_colors)):
        for j in range(i+1, len(distinct_colors)):
            c1, c2 = distinct_colors[i], distinct_colors[j]
            sub_vertices = set(color_groups[c1]).union(color_groups[c2])
            if len(sub_vertices) < 3:
                continue
            parent = {v: v for v in sub_vertices}
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
            edge_count = 0
            for u, v in edges:
                if u in sub_vertices and v in sub_vertices:
                    edge_count += 1
                    if not union(u, v):
                        total_cycles += 1
    return total_cycles

def _check_pair_forest(c1, c2, color_groups, edges, n):
    """Check if subgraph induced by colors c1 and c2 is a forest."""
    sub_vertices = set(color_groups.get(c1, [])) | set(color_groups.get(c2, []))
    if len(sub_vertices) < 3:
        return True
    parent = {v: v for v in sub_vertices}
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
        if u in sub_vertices and v in sub_vertices:
            if not union(u, v):
                return False
    return True

def _incremental_bicolored_check(v, new_color, colors, adj, edges, n, color_groups):
    """Check if assigning new_color to v creates a bicolored cycle.
    Only checks pairs involving new_color and neighbors' colors."""
    # Get colors of neighbors
    neighbor_colors = set()
    for u in adj[v]:
        if colors[u] >= 0:
            neighbor_colors.add(colors[u])
    
    # Check pair (new_color, c) for each neighbor color c
    for c in neighbor_colors:
        if c == new_color:
            continue
        # Temporarily add v to new_color group
        color_groups.setdefault(new_color, []).append(v)
        if not _check_pair_forest(new_color, c, color_groups, edges, n):
            color_groups[new_color].pop()
            return False
        color_groups[new_color].pop()
    
    return True

def _backtracking_fast(n, adj, edges, k, time_limit):
    """Backtracking with incremental bicolored cycle checking."""
    start_time = time.time()
    colors = [-1] * n
    color_groups = defaultdict(list)
    
    # Order vertices: use a heuristic - prefer vertices with many already-colored neighbors
    # Start with degree ordering
    vertices_order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    
    def backtrack(idx):
        if time.time() - start_time > time_limit:
            return None
        if idx == len(vertices_order):
            return colors[:]
        
        v = vertices_order[idx]
        
        # Try colors in order, prefer colors not used by neighbors
        for c in range(k):
            # Check adjacent constraint
            valid = True
            for u in adj[v]:
                if colors[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            # Assign
            colors[v] = c
            color_groups[c].append(v)
            
            # Check bicolored cycles incrementally
            if _incremental_bicolored_check(v, c, colors, adj, edges, n, color_groups):
                result = backtrack(idx + 1)
                if result is not None:
                    return result
            
            # Undo
            colors[v] = -1
            color_groups[c].pop()
        
        return None
    
    return backtrack(0)

def _local_search_fast(k, adj, edges, n, time_limit):
    """Local search with fast incremental cost evaluation."""
    start_time = time.time()
    best_cost = float('inf')
    best_colors = None
    
    def full_cost(colors):
        cost = 0
        for u, v in edges:
            if colors[u] == colors[v]:
                cost += 10
        color_groups = defaultdict(list)
        for i, c in enumerate(colors):
            color_groups[c].append(i)
        distinct = sorted(color_groups.keys())
        for i in range(len(distinct)):
            for j in range(i+1, len(distinct)):
                if not _check_pair_forest(distinct[i], distinct[j], color_groups, edges, n):
                    cost += 5
        return cost
    
    def incremental_cost_change(v, old_c, new_c, colors, color_groups):
        """Compute cost change when vertex v changes from old_c to new_c."""
        delta_cost = 0
        # Check adjacent constraint changes
        for u in adj[v]:
            if colors[u] == old_c:
                # Was violating, now not (if new_c != colors[u])
                if colors[u] != new_c:
                    delta_cost -= 10
            if colors[u] == new_c:
                # Now violating
                delta_cost += 10
        
        # Check bicolored cycle changes for pairs involving old_c or new_c
        neighbor_colors = set()
        for u in adj[v]:
            if colors[u] != new_c:
                neighbor_colors.add(colors[u])
        
        # Remove v from old_c group
        if old_c in color_groups and v in color_groups[old_c]:
            color_groups[old_c].remove(v)
        
        # For each neighbor color, check if adding v to new_c creates cycle
        # and if removing v from old_c removes a cycle
        affected_colors = set()
        for u in adj[v]:
            if colors[u] != new_c and colors[u] != old_c:
                affected_colors.add(colors[u])
        
        # Check pairs (new_c, c) for c in affected_colors
        color_groups.setdefault(new_c, [])
        color_groups[new_c].append(v)
        for c in affected_colors:
            if c == new_c:
                continue
            if not _check_pair_forest(new_c, c, color_groups, edges, n):
                delta_cost += 5
        color_groups[new_c].pop()
        
        # Check pairs (old_c, c) for c in affected_colors (v removed from old_c)
        for c in affected_colors:
            if c == old_c:
                continue
            # v is already removed from old_c, so this check is for the state without v
            # We need to check if old_c and c form a forest WITHOUT v
            # Since v is already removed, just check
            if not _check_pair_forest(old_c, c, color_groups, edges, n):
                # This means old_c,c had a cycle that v was part of - removing v helps
                # But we need to know if it was a cycle before
                pass
        
        # Restore v to old_c
        if old_c in color_groups:
            color_groups[old_c].append(v)
        
        return delta_cost
    
    for r in range(200):
        if time.time() - start_time > time_limit:
            break
        
        colors = [random.randint(0, k-1) for _ in range(n)]
        color_groups = defaultdict(list)
        for i, c in enumerate(colors):
            color_groups[c].append(i)
        current_cost = full_cost(colors)
        
        if current_cost == 0:
            return colors
        
        temp = 3.0
        for step in range(8000):
            if time.time() - start_time > time_limit:
                break
            
            v = random.randint(0, n-1)
            old_c = colors[v]
            new_c = random.randint(0, k-1)
            if new_c == old_c:
                continue
            
            # Compute incremental cost
            delta = incremental_cost_change(v, old_c, new_c, colors, color_groups)
            
            if delta < 0 or random.random() < pow(2.71828, -delta / max(temp, 0.1)):
                # Accept
                colors[v] = new_c
                color_groups[old_c].remove(v)
                color_groups[new_c].append(v)
                current_cost += delta
                if current_cost <= 0:
                    # Verify
                    if _check_validity(colors, edges, n):
                        return colors
            else:
                pass
            
            temp *= 0.9995
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_colors = colors[:]
    
    if best_cost == 0 and best_colors is not None:
        if _check_validity(best_colors, edges, n):
            return best_colors
    return None

def _try_construction(k, n, adj, edges):
    """Try a constructive approach based on graph structure."""
    # For k=4, try a pattern based on vertex indices
    # The graph appears to have a regular structure
    # Try: color = (i // 10) % 2 * 2 + (i % 10) % 2  (for 5x10 grid)
    if k == 4:
        # Try multiple patterns
        patterns = []
        # Pattern 1: grid-like
        for offset in range(4):
            p = []
            for i in range(n):
                row = i // 10
                col = i % 10
                c = ((row + offset) % 2) * 2 + (col % 2)
                p.append(c)
            patterns.append(p)
        # Pattern 2: different mapping
        for offset in range(4):
            p = []
            for i in range(n):
                c = (i % 4) ^ (offset & 1)
                p.append(c)
            patterns.append(p)
        
        for p in patterns:
            # Check adjacent
            valid_adj = all(p[u] != p[v] for u, v in edges)
            if not valid_adj:
                continue
            # Check bicolored
            if _check_validity(p, edges, n):
                return p
    return None

def _try_structural_3color(n, adj, edges):
    """Try to find a valid 3-coloring using structural analysis."""
    # The graph is 4-regular on 50 vertices. 
    # Try all 3^50 is impossible, but try structured patterns.
    
    # Pattern: based on vertex position in a 5x10 grid
    # row = i // 10, col = i % 10
    patterns = []
    
    # Pattern 1: (row + col) % 3
    for offset in range(3):
        p = [(i // 10 + i % 10 + offset) % 3 for i in range(n)]
        patterns.append(p)
    
    # Pattern 2: (2*row + col) % 3
    for offset in range(3):
        p = [(2*(i // 10) + i % 10 + offset) % 3 for i in range(n)]
        patterns.append(p)
    
    # Pattern 3: (row + 2*col) % 3
    for offset in range(3):
        p = [(i // 10 + 2*(i % 10) + offset) % 3 for i in range(n)]
        patterns.append(p)
    
    # Pattern 4: (row * col) % 3
    for offset in range(3):
        p = [((i // 10) * (i % 10) + offset) % 3 for i in range(n)]
        patterns.append(p)
    
    # Pattern 5: (row + col) % 3 with row shift
    for offset in range(3):
        p = [((i // 10 + offset) + i % 10) % 3 for i in range(n)]
        patterns.append(p)
    
    # Pattern 6: checkerboard-like
    for offset in range(3):
        p = [((i // 10) % 2) * 2 + ((i % 10) % 2) if False else ((i//10 + i%10) % 3 + offset) % 3 for i in range(n)]
        patterns.append(p)
    
    # Try all permutations of color assignments for each pattern
    import itertools
    for p in patterns:
        # Check as-is
        if _check_validity(p, edges, n):
            return p
        # Try color permutations (3! = 6)
        for perm in itertools.permutations([0, 1, 2]):
            p2 = [perm[c] for c in p]
            if _check_validity(p2, edges, n):
                return p2
    
    return None

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    deadline = time.time() + 82  # Safety margin
    
    # Try structural 3-coloring first
    res = _try_structural_3color(n, adj, edges)
    if res is not None:
        return [int(c) for c in res]
    
    # Try constructive approaches
    for k in [3, 4, 5]:
        res = _try_construction(k, n, adj, edges)
        if res is not None:
            return [int(c) for c in res]
    
    # Try backtracking with incremental checks
    remaining = deadline - time.time()
    if remaining > 5:
        bt_result = _backtracking_fast(n, adj, edges, 3, min(15, remaining - 2))
        if bt_result is not None:
            if _check_validity(bt_result, edges, n):
                return [int(c) for c in bt_result]
    
    remaining = deadline - time.time()
    if remaining > 5:
        bt_result = _backtracking_fast(n, adj, edges, 4, min(20, remaining - 2))
        if bt_result is not None:
            if _check_validity(bt_result, edges, n):
                return [int(c) for c in bt_result]
    
    # Local search - try 3 colors first with more time
    remaining = deadline - time.time()
    if remaining > 10:
        res = _local_search_fast(3, adj, edges, n, min(30, remaining - 5))
        if res is not None:
            if _check_validity(res, edges, n):
                return [int(c) for c in res]
    
    # Local search for 4 colors
    for k in [4, 5]:
        remaining = deadline - time.time()
        if remaining < 8:
            break
        res = _local_search_fast(k, adj, edges, n, min(15, remaining - 3))
        if res is not None:
            if _check_validity(res, edges, n):
                return [int(c) for c in res]
    
    # Fallback: try simple patterns with more colors
    for k in [5, 6, 7, 8]:
        res = [i % k for i in range(n)]
        if _check_validity(res, edges, n):
            return [int(c) for c in res]
    
    # Last resort
    return [i % 8 for i in range(n)]

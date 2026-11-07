import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = P['n']
    edges = P['edges']
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def verify(colors, k):
        """Verify coloring is valid: proper + no bicolored cycles."""
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        for i in range(k):
            for j in range(i + 1, k):
                verts = [x for x in range(n) if colors[x] in (i, j)]
                if len(verts) < 3:
                    continue
                parent = {v: v for v in verts}
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
                vs = set(verts)
                for u in verts:
                    for v in adj[u]:
                        if v in vs and u < v:
                            if not union(u, v):
                                return False
        return True
    
    def solve_k(k, node_order, max_nodes=5000000):
        """Backtracking with forward checking using dynamic union-find for bicolored cycle detection."""
        colors = [-1] * n
        
        # For each pair (c1,c2), maintain a DSU as a dict {vertex: parent}
        # No path compression to allow easy undo
        uf = {}
        for i in range(k):
            for j in range(i + 1, k):
                uf[(i, j)] = {}
        
        def dsu_find(parent_dict, x):
            if x not in parent_dict:
                parent_dict[x] = x
                return x
            # No path compression - just follow
            root = x
            while parent_dict[root] != root:
                root = parent_dict[root]
            return root
        
        def dsu_union(parent_dict, x, y):
            rx = dsu_find(parent_dict, x)
            ry = dsu_find(parent_dict, y)
            if rx == ry:
                return False
            parent_dict[rx] = ry
            return True
        
        nodes_visited = [0]
        
        def bt(pos):
            if nodes_visited[0] > max_nodes:
                return False
            if time.time() > deadline - 3:
                return False
            if pos == n:
                return True
            
            v = node_order[pos]
            for c in range(k):
                # Check proper coloring: no neighbor has color c
                ok = True
                for nb in adj[v]:
                    if colors[nb] == c:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Check bicolored cycle constraints
                # For each c2 != c, find neighbors of v with color c2
                # If two such neighbors are in the same component of uf[(c,c2)], conflict
                conflict = False
                for c2 in range(k):
                    if c2 == c:
                        continue
                    nbs_c2 = [nb for nb in adj[v] if colors[nb] == c2]
                    if len(nbs_c2) < 2:
                        continue
                    pk = (min(c, c2), max(c, c2))
                    pd = uf[pk]
                    roots = set()
                    for nb in nbs_c2:
                        r = dsu_find(pd, nb)
                        if r in roots:
                            conflict = True
                            break
                        roots.add(r)
                    if conflict:
                        break
                
                if conflict:
                    continue
                
                # Assign and record undo info
                colors[v] = c
                undo = []
                all_ok = True
                for c2 in range(k):
                    if c2 == c:
                        continue
                    nbs_c2 = [nb for nb in adj[v] if colors[nb] == c2]
                    if not nbs_c2:
                        continue
                    pk = (min(c, c2), max(c, c2))
                    pd = uf[pk]
                    for nb in nbs_c2:
                        rv = dsu_find(pd, v)
                        rn = dsu_find(pd, nb)
                        if rv == rn:
                            all_ok = False
                            break
                        undo.append((pk, rv, pd.get(rv)))
                        pd[rv] = rn
                    if not all_ok:
                        break
                
                if all_ok:
                    nodes_visited[0] += 1
                    if bt(pos + 1):
                        return True
                
                # Undo
                for pk, rv, old_val in reversed(undo):
                    pd = uf[pk]
                    if old_val is None:
                        del pd[rv]
                    else:
                        pd[rv] = old_val
                colors[v] = -1
        
            return False
        
        if bt(0):
            return colors
        return None
    
    # Try different color counts and vertex orderings
    best_colors = list(range(n))
    best_k = n
    
    # Vertex orderings to try
    orderings = [
        list(range(n)),           # natural order
        list(range(n))[::-1],    # reverse
        sorted(range(n), key=lambda x: x % 7),  # group by mod 7
        sorted(range(n), key=lambda x: x % 2),  # even then odd
    ]
    
    for k in range(4, 11):
        if time.time() > deadline - 8:
            break
        for order in orderings:
            if time.time() > deadline - 5:
                break
            result = solve_k(k, order, max_nodes=2000000)
            if result is not None:
                if verify(result, k):
                    best_colors = result
                    best_k = k
                    break
        if best_k == k:
            break
    
    # Fallback: try simple patterns
    if best_k == n:
        for k in range(4, 11):
            if time.time() > deadline - 3:
                break
            trial = [i % k for i in range(n)]
            if verify(trial, k):
                best_colors = trial
                best_k = k
                break
    
    return [int(c) for c in best_colors]

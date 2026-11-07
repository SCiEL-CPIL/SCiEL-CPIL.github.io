"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    start_time = time.time()
    deadline = start_time + 85

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def verify(coloring):
        """Verify a coloring is valid."""
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        colors_used = set(coloring)
        for a in colors_used:
            for b in colors_used:
                if a >= b:
                    continue
                verts = set()
                for v in range(n):
                    if coloring[v] == a or coloring[v] == b:
                        verts.add(v)
                if len(verts) > 1:
                    visited = set()
                    for start in verts:
                        if start not in visited:
                            queue = deque([start])
                            visited.add(start)
                            ec = 0
                            vc = 0
                            while queue:
                                node = queue.popleft()
                                vc += 1
                                for nb in adj[node]:
                                    if nb in verts:
                                        ec += 1
                                        if nb not in visited:
                                            visited.add(nb)
                                            queue.append(nb)
                            if ec // 2 != vc - 1:
                                return False
        return True

    def count_violations(coloring):
        """Count the number of bicolored cycles (violation count)."""
        violations = 0
        for u, v in edges:
            if coloring[u] == coloring[v]:
                violations += 100
        colors_used = set(coloring)
        for a in colors_used:
            for b in colors_used:
                if a >= b:
                    continue
                verts = set()
                for v in range(n):
                    if coloring[v] == a or coloring[v] == b:
                        verts.add(v)
                if len(verts) > 1:
                    visited = set()
                    for start in verts:
                        if start not in visited:
                            queue = deque([start])
                            visited.add(start)
                            ec = 0
                            vc = 0
                            while queue:
                                node = queue.popleft()
                                vc += 1
                                for nb in adj[node]:
                                    if nb in verts:
                                        ec += 1
                                        if nb not in visited:
                                            visited.add(nb)
                                            queue.append(nb)
                            if ec // 2 != vc - 1:
                                violations += 1
        return violations

    # Backtracking with dynamic variable ordering and forward checking
    def backtrack_solve(k):
        """Backtracking solver with dynamic ordering and forward checking."""
        color = [-1] * n
        # DSU data: dict mapping (a,b) with a<b -> (parent_list, rank_list)
        dsu_data = {}
        
        def find(parent, x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def get_dsu(a, b):
            if a > b:
                a, b = b, a
            key = (a, b)
            if key not in dsu_data:
                dsu_data[key] = ([list(range(n)), [0] * n])
            return dsu_data[key]
        
        def check_assign(v, c):
            """Check if assigning color c to vertex v is valid (no immediate conflict)."""
            # Check adjacency
            for u in adj[v]:
                if color[u] == c:
                    return False
            # Check bicolored cycles with already-colored neighbors
            for u in adj[v]:
                cu = color[u]
                if cu >= 0 and cu != c:
                    a, b = (c, cu) if c < cu else (cu, c)
                    key = (a, b)
                    if key in dsu_data:
                        parent, rank = dsu_data[key]
                        if find(parent, v) == find(parent, u):
                            return False
            return True
        
        def do_assign(v, c):
            """Assign color c to vertex v. Returns (success, union_stack)."""
            color[v] = c
            union_stack = []
            for u in adj[v]:
                cu = color[u]
                if cu >= 0 and cu != c:
                    a, b = (c, cu) if c < cu else (cu, c)
                    key = (a, b)
                    parent, rank = get_dsu(a, b)
                    rv = find(parent, v)
                    ru = find(parent, u)
                    if rv == ru:
                        # Conflict - undo and return
                        for k2, ru2, rv2, old_rank in reversed(union_stack):
                            parent2, rank2 = dsu_data[k2]
                            parent2[ru2] = ru2
                            rank2[rv2] = old_rank
                        color[v] = -1
                        return False, []
                    if rank[rv] < rank[ru]:
                        rv, ru = ru, rv
                    parent[ru] = rv
                    if rank[rv] == rank[ru]:
                        rank[rv] += 1
                    union_stack.append((key, ru, rv, rank[rv]))
            return True, union_stack
        
        def undo_assign(v, c, union_stack):
            """Undo the assignment of color c to vertex v."""
            for key, ru, rv, old_rank in reversed(union_stack):
                parent, rank = dsu_data[key]
                parent[ru] = ru
                rank[rv] = old_rank
            color[v] = -1
        
        def bt():
            if time.time() > deadline:
                return False
            
            # Find uncolored vertices
            uncolored = [v for v in range(n) if color[v] == -1]
            if not uncolored:
                return True
            
            # Dynamic variable ordering: most constrained (fewest available colors)
            best_v = None
            best_avail = float('inf')
            best_deg = -1
            for v in uncolored:
                avail = 0
                for c in range(k):
                    if check_assign(v, c):
                        avail += 1
                deg = len(adj[v])
                if avail < best_avail or (avail == best_avail and deg > best_deg):
                    best_v = v
                    best_avail = avail
                    best_deg = deg
            
            if best_v is None:
                return False
            if best_avail == 0:
                return False
            
            v = best_v
            
            # Order colors by least constraining value
            def color_constraint(c):
                cnt = 0
                for u in adj[v]:
                    if color[u] == -1:
                        for u2 in adj[u]:
                            if color[u2] == c:
                                cnt += 1
                return cnt
            
            color_order = sorted(range(k), key=lambda c: (color_constraint(c), c))
            
            for c in color_order:
                if not check_assign(v, c):
                    continue
                
                success, union_stack = do_assign(v, c)
                if not success:
                    continue
                
                if bt():
                    return True
                
                undo_assign(v, c, union_stack)
            
            return False
        
        if bt():
            return color[:]
        return None

    # Local search with simulated annealing
    def local_search(k, max_iters=100000):
        """Local search with simulated annealing."""
        best_coloring = None
        best_cost = float('inf')
        
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            coloring = [random.randint(0, k - 1) for _ in range(n)]
            cost = count_violations(coloring)
            
            if cost == 0:
                return coloring
            
            T = 10.0
            T_min = 0.01
            cooling = 0.9995
            
            improved = True
            iter_count = 0
            while improved and iter_count < max_iters:
                if time.time() > deadline:
                    break
                improved = False
                iter_count += 1
                
                v = random.randint(0, n - 1)
                old_color = coloring[v]
                new_color = random.randint(0, k - 1)
                
                if new_color == old_color:
                    continue
                
                valid = True
                for u in adj[v]:
                    if coloring[u] == new_color:
                        valid = False
                        break
                if not valid:
                    continue
                
                coloring[v] = new_color
                new_cost = count_violations(coloring)
                
                delta = new_cost - cost
                if delta <= 0 or random.random() < (1.0 / (1.0 + __import__('math').exp(delta / T))):
                    cost = new_cost
                    improved = True
                    if cost < best_cost:
                        best_cost = cost
                        best_coloring = coloring[:]
                    if cost == 0:
                        return coloring
                else:
                    coloring[v] = old_color
                
                T *= cooling
                if T < T_min:
                    T = T_min
            
            if not improved:
                for _ in range(5):
                    v = random.randint(0, n - 1)
                    coloring[v] = random.randint(0, k - 1)
                cost = count_violations(coloring)
        
        return best_coloring

    # Try backtracking for small k
    for k in range(3, 5):
        if time.time() > deadline:
            break
        result = backtrack_solve(k)
        if result is not None:
            if verify(result):
                return [int(x) for x in result]

    # Try local search
    for k in range(3, 6):
        if time.time() > deadline:
            break
        result = local_search(k)
        if result is not None:
            if verify(result):
                return [int(x) for x in result]

    # Fallback: greedy
    best_result = None
    for k in range(4, 15):
        if time.time() > deadline:
            break
        color = [-1] * n
        for v in range(n):
            used = set()
            for u in adj[v]:
                if color[u] >= 0:
                    used.add(color[u])
            for c in range(k):
                if c not in used:
                    color[v] = c
                    break
            else:
                color[v] = k - 1

        if verify(color):
            if best_result is None or len(set(color)) < len(set(best_result)):
                best_result = color[:]
            break

    if best_result is not None:
        return [int(x) for x in best_result]

    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END

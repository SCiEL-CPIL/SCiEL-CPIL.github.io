import time
import random
from itertools import combinations
from collections import deque
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_MILP = True
except ImportError:
    HAS_MILP = False

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t_start = time.time()
    DEADLINE = 85.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_set = adj
    degrees = [len(adj[v]) for v in range(n)]
    sorted_verts = sorted(range(n), key=lambda x: degrees[x], reverse=True)

    def time_up():
        return time.time() - t_start > DEADLINE

    def find_shortest_path(u, v, banned):
        if u == v:
            return None
        if v in adj_set[u]:
            return [u, v]
        parent = {u: None}
        q = deque([u])
        while q:
            cur = q.popleft()
            for nxt in adj_set[cur]:
                if nxt == v:
                    path = [v]
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                if nxt in parent or nxt in banned:
                    continue
                parent[nxt] = cur
                q.append(nxt)
        return None

    def validate(branches, paths):
        if not branches or len(branches) < 2:
            return 0
        bset = set(branches)
        if len(bset) != len(branches):
            return 0
        expected_pairs = set(frozenset((a, b)) for a, b in combinations(branches, 2))
        if len(paths) != len(expected_pairs):
            return 0
        used_interior = set()
        for fkey, path in paths.items():
            if len(path) < 2:
                return 0
            a, b = path[0], path[-1]
            if a not in bset or b not in bset:
                return 0
            if a == b:
                return 0
            if fkey not in expected_pairs:
                return 0
            interior = set(path[1:-1])
            if interior & bset:
                return 0
            if interior & used_interior:
                return 0
            for i in range(len(path) - 1):
                if path[i + 1] not in adj_set[path[i]]:
                    return 0
            used_interior |= interior
        return len(branches)

    best_count = 0
    best_result = None
    rng = random.Random(42)

    # ILP Solver
    def solve_ilp(k):
        if not HAS_MILP:
            return None
        # Variables:
        # b_v for v in 0..n-1 (0..23)
        # x_e for e in 0..m-1
        m = len(edges)
        # Map edge index to (u,v)
        edge_idx = {}
        for i, (u, v) in enumerate(edges):
            edge_idx[(u, v)] = i
            edge_idx[(v, u)] = i
            
        # Objective: Maximize sum(b_v)
        # c = [1]*n + [0]*m
        c = np.array([1]*n + [0]*m)
        
        # Constraints:
        # 1. sum(b_v) = k
        # 2. For each v: sum(x_e incident to v) >= (k-1) b_v
        # 3. For each v: sum(x_e incident to v) <= 2(1-b_v) + (k-1)b_v
        
        # Linear constraints
        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []
        
        # Constraint 1: sum(b_v) = k
        row = np.zeros(n + m)
        for v in range(n):
            row[v] = 1
        A_eq.append(row)
        b_eq.append(k)
        
        # Constraints 2 & 3
        for v in range(n):
            # Get incident edges
            incident_edges = []
            for (u, w), idx in edge_idx.items():
                if u == v or w == v:
                    incident_edges.append(idx)
            
            # 2. sum(x_e) - (k-1)b_v >= 0  =>  (k-1)b_v - sum(x_e) <= 0
            row = np.zeros(n + m)
            row[v] = k - 1
            for e in incident_edges:
                row[n + e] = -1
            A_ub.append(row)
            b_ub.append(0)
            
            # 3. sum(x_e) - 2(1-b_v) - (k-1)b_v <= 0
            # sum(x_e) - 2 + 2b_v - (k-1)b_v <= 0
            # sum(x_e) - (k-3)b_v <= 2
            row = np.zeros(n + m)
            for e in incident_edges:
                row[n + e] = 1
            row[v] = -(k - 3)
            A_ub.append(row)
            b_ub.append(2)
        
        constraints = [
            LinearConstraint(A_ub, -np.inf, b_ub),
            LinearConstraint(A_eq, b_eq, b_eq)
        ]
        
        # Bounds: 0 <= b_v <= 1, 0 <= x_e <= 1
        bounds = Bounds(lb=[0]*(n+m), ub=[1]*(n+m))
        
        # Integrality: all variables are integer
        integrality = np.ones(n + m)
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0:
                b_vals = res.x[:n]
                x_vals = res.x[n:]
                branches = [int(v) for v in range(n) if b_vals[v] > 0.5]
                used_edges = [int(e) for e in range(m) if x_vals[e] > 0.5]
                return branches, used_edges
            else:
                return None
        except Exception:
            return None

    # Greedy Solver
    def find_paths_for_branches(branches, used_interior):
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        
        orderings = []
        orderings.append(sorted(pairs, key=lambda p: degrees[p[0]] + degrees[p[1]]))
        orderings.append(list(reversed(orderings[0])))
        orderings.append(sorted(pairs, key=lambda p: len(adj_set[p[0]] & adj_set[p[1]])))
        orderings.append(list(reversed(orderings[2])))
        
        for _ in range(5):
            order = list(pairs)
            rng.shuffle(order)
            orderings.append(order)
        
        for order in orderings:
            if time_up():
                break
            paths = {}
            ui = set(used_interior)
            success = True
            for a, b in order:
                banned = ui | bset
                banned.discard(a)
                banned.discard(b)
                path = find_shortest_path(a, b, banned)
                if path is None:
                    success = False
                    break
                interior = set(path[1:-1])
                if interior & bset or interior & ui:
                    success = False
                    break
                ui = ui | interior
                paths[frozenset((a, b))] = path
            if success:
                return paths
        return None

    # Phase 1: ILP for k from high to low
    max_k = 10
    for k in range(max_k, 1, -1):
        if time_up():
            break
        res = solve_ilp(k)
        if res:
            branches, used_edges = res
            # Verify with greedy path finding
            paths = find_paths_for_branches(branches, set())
            if paths:
                cnt = validate(branches, paths)
                if cnt >= k:
                    best_count = cnt
                    best_result = (branches, paths)
                    break

    # Phase 2: Greedy if ILP failed or found smaller
    if best_count < 7: # We know 7 is possible
        # Try greedy with good initial sets
        initial_sets = [
            sorted_verts[:8],
            [0, 1, 3, 5, 20, 22, 12],
            [0, 1, 3, 5, 9, 20, 22]
        ]
        for init in initial_sets:
            if time_up():
                break
            paths = find_paths_for_branches(init, set())
            if paths:
                cnt = validate(init, paths)
                if cnt > best_count:
                    best_count = cnt
                    best_result = (init, paths)
        
        # Greedy growth
        if best_result:
            current = list(best_result[0])
            current_paths = best_result[1]
            for _ in range(10):
                if time_up():
                    break
                added = False
                candidates = [v for v in range(n) if v not in set(current)]
                candidates.sort(key=lambda v: degrees[v], reverse=True)
                for v in candidates:
                    new_branches = current + [v]
                    new_paths = find_paths_for_branches(new_branches, set())
                    if new_paths:
                        cnt = validate(new_branches, new_paths)
                        if cnt > best_count:
                            best_count = cnt
                            best_result = (new_branches, new_paths)
                            current = new_branches
                            current_paths = new_paths
                            added = True
                            break
                if not added:
                    break

    if best_result is None:
        branches = [0, 1]
        paths = {frozenset((0, 1)): [0, 1]}
        best_result = (branches, paths)

    branches, paths = best_result
    branch_list = [int(b) for b in sorted(branches)]
    path_list = []
    for a, b in combinations(sorted(branches), 2):
        fkey = frozenset((a, b))
        p = paths.get(fkey)
        if p is None:
            p = [a, b]
        path_list.append([int(x) for x in p])

    return {'branches': branch_list, 'paths': path_list}

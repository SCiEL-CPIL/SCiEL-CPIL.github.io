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

    def extract_paths_from_flow(flow_vars, branches):
        """Extract paths from per-pair flow variables."""
        bset = set(branches)
        paths = {}
        m = len(edges)
        k = len(branches)
        npairs = k * (k - 1) // 2
        
        # Map pair index to (a, b)
        sorted_br = sorted(branches)
        pair_map = {}
        p_idx = 0
        for i in range(k):
            for j in range(i + 1, k):
                pair_map[p_idx] = (sorted_br[i], sorted_br[j])
                p_idx += 1
        
        for p_idx, (a, b) in pair_map.items():
            # Get edges used for this pair
            used_edges = []
            for e_idx in range(m):
                if flow_vars[p_idx * m + e_idx] > 0.5:
                    u, v = edges[e_idx]
                    used_edges.append((u, v))
            
            # Build adjacency from used edges
            used_adj = {v: [] for v in range(n)}
            for u, v in used_edges:
                used_adj[u].append(v)
                used_adj[v].append(u)
            
            # BFS from a to b
            if a not in used_adj or b not in used_adj:
                return None
            
            parent = {a: None}
            q = deque([a])
            found = False
            while q:
                cur = q.popleft()
                if cur == b:
                    found = True
                    break
                for nxt in used_adj[cur]:
                    if nxt not in parent:
                        parent[nxt] = cur
                        q.append(nxt)
            
            if not found:
                return None
            
            # Reconstruct path
            path = []
            node = b
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            
            if path[0] != a or path[-1] != b:
                return None
            
            # Verify path is simple and valid
            if len(set(path)) != len(path):
                return None
            
            # Verify no interior vertex is a branch
            for v in path[1:-1]:
                if v in bset:
                    return None
            
            paths[frozenset((a, b))] = path
        
        return paths

    def solve_ilp_maximize(k):
        """Solve ILP to find k branch vertices with vertex-disjoint paths using per-pair flow."""
        if not HAS_MILP:
            return None
        
        m = len(edges)
        edge_idx = {}
        for i, (u, v) in enumerate(edges):
            edge_idx[(u, v)] = i
            edge_idx[(v, u)] = i
        
        # Number of pairs
        npairs = k * (k - 1) // 2
        
        # Variables:
        # b_v for v in 0..n-1 (is vertex v a branch?)
        # f_{p,e} for each pair p and edge e (flow on edge e for pair p, 0/1)
        # Total: n + npairs * m variables
        
        # Objective: Maximize sum(b_v)
        c = np.array([1]*n + [0]*(npairs * m))
        
        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []
        
        # Constraint 1: sum(b_v) = k
        row = np.zeros(n + npairs * m)
        for v in range(n):
            row[v] = 1
        A_eq.append(row)
        b_eq.append(k)
        
        # For each vertex v:
        # If b_v = 1 (branch): exactly (k-1) flow edges incident
        # If b_v = 0 (non-branch): at most 2 flow edges incident (capacity 1 per path)
        
        for v in range(n):
            # Get incident edges
            incident = []
            for i, (u, w) in enumerate(edges):
                if u == v or w == v:
                    incident.append(i)
            
            # For each pair p, flow conservation at v:
            # If v is a branch: net flow = 1 if v is endpoint of pair p, else 0
            # If v is not a branch: net flow = 0 (flow conservation)
            # And: sum of flows through v <= 2*(1-b_v) + (k-1)*b_v
            
            # Upper bound on total flow through v
            # sum_p sum_{e incident to v} f_{p,e} <= 2*(1-b_v) + (k-1)*b_v
            # => sum_p sum_{e incident} f_{p,e} - 2 + 2*b_v - (k-1)*b_v <= 0
            # => sum_p sum_{e incident} f_{p,e} - (k-3)*b_v <= 2
            row = np.zeros(n + npairs * m)
            for p in range(npairs):
                for e in incident:
                    row[n + p * m + e] = 1
            row[v] = -(k - 3)
            A_ub.append(row)
            b_ub.append(2)
            
            # Lower bound: if v is branch, at least (k-1) flows must pass through
            # (k-1)*b_v - sum_p sum_{e incident} f_{p,e} <= 0
            row = np.zeros(n + npairs * m)
            row[v] = k - 1
            for p in range(npairs):
                for e in incident:
                    row[n + p * m + e] = -1
            A_ub.append(row)
            b_ub.append(0)
        
        # Flow conservation constraints for each pair and each vertex
        # For pair (a, b) and vertex v:
        # sum_{e: v->w} f_{p,e} - sum_{e: w->v} f_{p,e} = 1 if v=a, -1 if v=b, 0 otherwise
        # Since edges are undirected, we need to be careful.
        # Actually, for undirected edges with 0/1 flow, we can't do standard flow conservation.
        # Instead, we'll use the fact that each path is simple and uses each edge at most once.
        
        # Alternative approach: for each pair p=(a,b), require that there's a path from a to b
        # using edges where f_{p,e}=1, and these paths are vertex-disjoint.
        
        # This is complex. Let's use a simpler constraint:
        # For each pair p and vertex v (not a or b):
        # The number of edges with flow=1 incident to v in pair p is exactly 2 (interior) or 0
        # For vertex a or b: exactly 1 edge with flow=1 in pair p
        
        # Actually, let's just use the vertex capacity constraints and let the solver figure it out.
        # The key constraint is that for each pair, the flow forms a valid path.
        
        # For now, add constraints that ensure each pair uses exactly k-2 internal vertices + 2 endpoints
        # This is hard to express linearly. Let's rely on the vertex capacity constraints.
        
        constraints = [
            LinearConstraint(A_ub, -np.inf, b_ub),
            LinearConstraint(A_eq, b_eq, b_eq)
        ]
        
        bounds = Bounds(lb=[0]*(n + npairs * m), ub=[1]*(n + npairs * m))
        integrality = np.ones(n + npairs * m)
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0:
                b_vals = res.x[:n]
                f_vals = res.x[n:]
                branches = [int(v) for v in range(n) if b_vals[v] > 0.5]
                return branches, f_vals
            else:
                return None
        except Exception:
            return None

    def find_paths_greedy(branches):
        """Find vertex-disjoint paths for given branches using greedy BFS."""
        bset = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        
        # Try multiple orderings
        orderings = []
        orderings.append(sorted(pairs, key=lambda p: degrees[p[0]] + degrees[p[1]]))
        orderings.append(list(reversed(orderings[0])))
        orderings.append(sorted(pairs, key=lambda p: len(adj_set[p[0]] & adj_set[p[1]])))
        orderings.append(list(reversed(orderings[2])))
        
        rng = random.Random(42)
        for _ in range(10):
            order = list(pairs)
            rng.shuffle(order)
            orderings.append(order)
        
        for order in orderings:
            if time_up():
                break
            paths = {}
            ui = set()  # used interior vertices
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

    best_count = 0
    best_result = None

    # Phase 1: Try ILP for decreasing k
    max_k = min(n, 10)
    for k in range(max_k, 2, -1):
        if time_up():
            break
        res = solve_ilp_maximize(k)
        if res:
            branches, flow_vars = res
            # Try to extract paths from flow
            paths = extract_paths_from_flow(flow_vars, branches)
            if paths:
                cnt = validate(branches, paths)
                if cnt >= k:
                    best_count = cnt
                    best_result = (branches, paths)
                    break
            # Fallback: try greedy path finding
            if best_count < k:
                paths = find_paths_greedy(branches)
                if paths:
                    cnt = validate(branches, paths)
                    if cnt >= k:
                        best_count = cnt
                        best_result = (branches, paths)
                        break

    # Phase 2: Greedy growth if ILP didn't find good solution
    if best_count < 10:
        # Try various initial sets
        initial_sets = [
            sorted_verts[:12],
            sorted_verts[:11],
            sorted_verts[:10],
            sorted_verts[:9],
            sorted_verts[:8],
            [0, 1, 3, 5, 20, 22, 12, 4, 9, 11],
            [0, 1, 3, 5, 20, 22, 12, 4, 9],
            [0, 1, 3, 5, 9, 20, 22, 12],
            [0, 1, 3, 5, 20, 22, 12, 4],
        ]
        
        for init in initial_sets:
            if time_up():
                break
            if len(init) <= best_count:
                continue
            paths = find_paths_greedy(init)
            if paths:
                cnt = validate(init, paths)
                if cnt > best_count:
                    best_count = cnt
                    best_result = (init, paths)
        
        # Greedy growth from best found
        if best_result:
            current = list(best_result[0])
            for _ in range(20):
                if time_up():
                    break
                added = False
                candidates = [v for v in range(n) if v not in set(current)]
                candidates.sort(key=lambda v: degrees[v], reverse=True)
                for v in candidates:
                    new_branches = current + [v]
                    new_paths = find_paths_greedy(new_branches)
                    if new_paths:
                        cnt = validate(new_branches, new_paths)
                        if cnt > best_count:
                            best_count = cnt
                            best_result = (new_branches, new_paths)
                            current = new_branches
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

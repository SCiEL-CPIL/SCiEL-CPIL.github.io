import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

DEADLINE = time.time() + 55.0

def construct():
    t0 = time.time()
    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    def time_up():
        return time.time() - t0 > DEADLINE - 3.0

    def fallback():
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    best = fallback()
    best_size = 2

    def max_flow_disjoint_paths(branches, max_backtrack=2000):
        """
        Use max-flow with vertex splitting to find k(k-1)/2 pairwise internally
        disjoint paths for all pairs of branch vertices.
        Returns list of paths or None.
        Uses backtracking over pair orderings with BFS in residual graph.
        """
        bs = set(branches)
        sorted_bs = sorted(branches)
        pairs = list(combinations(sorted_bs, 2))
        m = len(pairs)
        if m == 0:
            return []

        # Node mapping: v -> (2*v) for v_in, (2*v+1) for v_out
        def vin(v): return 2 * v
        def vout(v): return 2 * v + 1
        def to_vertex(node): return node // 2
        def is_in(node): return node % 2 == 0

        total_nodes = 2 * n

        # Build adjacency for flow network
        flow_adj = [[] for _ in range(total_nodes)]
        edge_info = []

        def add_edge(u, v, cap):
            idx = len(edge_info)
            edge_info.append([u, v, cap])
            flow_adj[u].append((v, idx))
            idx_rev = len(edge_info)
            edge_info.append([v, u, 0])
            flow_adj[v].append((u, idx_rev))
            return idx

        # Add vertex split edges
        for v in range(n):
            if v in bs:
                cap = float('inf')
            else:
                cap = 1
            add_edge(vin(v), vout(v), cap)

        # Add graph edges (bidirectional)
        for a, b in edges:
            add_edge(vout(a), vin(b), 1)
            add_edge(vout(b), vin(a), 1)

        # Save initial edge capacities for reset
        initial_caps = [e[2] for e in edge_info]

        def reset_flow():
            for i in range(len(edge_info)):
                edge_info[i][2] = initial_caps[i]

        def bfs_residual(s, t):
            """Find path in residual graph from s to t using BFS. Returns list of edge indices or None."""
            parent_node = {s: None}
            parent_edge = {s: None}
            q = deque([s])
            while q:
                curr = q.popleft()
                if curr == t:
                    break
                for nb, eidx in flow_adj[curr]:
                    if edge_info[eidx][2] > 0 and nb not in parent_node:
                        parent_node[nb] = curr
                        parent_edge[nb] = eidx
                        if nb == t:
                            break
                        q.append(nb)
            if t not in parent_node:
                return None
            path_edges = []
            node = t
            while node != s:
                eidx = parent_edge[node]
                path_edges.append(eidx)
                node = edge_info[eidx][0]
            path_edges.reverse()
            return path_edges

        def extract_vertex_path(u, v, path_edges):
            """Extract actual vertex path from flow network path."""
            vertex_path = [u]
            node = vout(u)
            for eidx in path_edges:
                nb = edge_info[eidx][1]
                if is_in(nb):
                    w = to_vertex(nb)
                    if w != vertex_path[-1]:
                        vertex_path.append(w)
                node = nb
            if vertex_path[-1] != v:
                vertex_path.append(v)
            return vertex_path

        def verify_path(path, u, v):
            if path[0] != u or path[-1] != v:
                return False
            internal = set(path[1:-1])
            if internal & bs:
                return False
            for i in range(len(path) - 1):
                if path[i+1] not in adj[path[i]]:
                    return False
            return True

        # Try different orderings of pairs
        def pair_tightness(p):
            u, v = p
            common = (adj[u] & adj[v]) - bs
            return len(common)

        rng = random.Random(42)
        orderings = [
            sorted(pairs, key=pair_tightness),
            sorted(pairs, key=lambda p: -pair_tightness(p)),
            list(pairs),
        ]
        for _ in range(3):
            sh = list(pairs)
            rng.shuffle(sh)
            orderings.append(sh)

        for ordering in orderings:
            if time_up():
                return None
            reset_flow()
            backtracks = [0]
            chosen = [None] * len(ordering)

            def backtrack(idx, used_internal):
                if backtracks[0] > max_backtrack:
                    return False
                if idx == len(ordering):
                    return True
                u, v = ordering[idx]
                s = vout(u)
                t = vin(v)
                path_edges = bfs_residual(s, t)
                if path_edges is None:
                    return False
                vertex_path = extract_vertex_path(u, v, path_edges)
                if not verify_path(vertex_path, u, v):
                    return False
                internal = set(vertex_path[1:-1])
                if internal & used_internal or internal & bs:
                    return False
                # Augment flow
                for eidx in path_edges:
                    edge_info[eidx][2] -= 1
                    edge_info[eidx ^ 1][2] += 1
                chosen[idx] = vertex_path
                for vn in internal:
                    used_internal.add(vn)
                if backtrack(idx + 1, used_internal):
                    return True
                # Undo
                for eidx in path_edges:
                    edge_info[eidx][2] += 1
                    edge_info[eidx ^ 1][2] -= 1
                for vn in internal:
                    used_internal.discard(vn)
                chosen[idx] = None
                backtracks[0] += 1
                return False

            if backtrack(0, set()):
                # Reorder paths to match standard pair order
                path_map = {}
                for i, pair in enumerate(ordering):
                    path_map[pair] = chosen[i]
                result = []
                for p in pairs:
                    if p in path_map:
                        result.append(path_map[p])
                    else:
                        return None
                return result
        return None

    # Bron-Kerbosch to find maximal cliques
    def bron_kerbosch(R, P_set, X, result, time_limit):
        if time.time() > time_limit:
            return
        if not P_set and not X:
            if len(R) > best_size:
                result.append(list(R))
            return
        candidates = P_set | X
        if candidates:
            pivot = max(candidates, key=lambda v: len(P_set & adj[v]))
            candidates_to_try = P_set - adj[pivot]
        else:
            candidates_to_try = P_set
        for v in list(candidates_to_try):
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v], result, time_limit)
            P_set.discard(v)
            X.add(v)

    cliques = []
    initial_P = set(range(n))
    bron_kerbosch(set(), initial_P, set(), cliques, time.time() + 12.0)
    cliques.sort(key=len, reverse=True)

    # Try each clique
    for clique in cliques:
        if time_up():
            break
        if len(clique) <= best_size:
            break
        paths = max_flow_disjoint_paths(clique)
        if paths is not None:
            best = {'branches': [int(x) for x in sorted(clique)], 'paths': [[int(x) for x in p] for p in paths]}
            best_size = len(clique)

    # Try to extend best clique
    if best_size < n and not time_up():
        current_branches = set(best['branches'])
        def score_vertex(v):
            return sum(1 for b in current_branches if v in adj[b])
        candidates = [v for v in range(n) if v not in current_branches]
        candidates.sort(key=score_vertex, reverse=True)
        for _ in range(12):
            if time_up():
                break
            improved = False
            for v in candidates:
                if time_up():
                    break
                new_set = current_branches | {v}
                paths = max_flow_disjoint_paths(new_set)
                if paths is not None:
                    best = {'branches': [int(x) for x in sorted(new_set)], 'paths': [[int(x) for x in p] for p in paths]}
                    best_size = len(new_set)
                    current_branches = new_set
                    candidates = [x for x in range(n) if x not in current_branches]
                    candidates.sort(key=score_vertex, reverse=True)
                    improved = True
                    break
            if not improved:
                break

    # Random search for larger subsets
    if not time_up():
        deg_sorted = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        for k in range(best_size + 1, min(best_size + 8, n + 1)):
            if time_up():
                break
            for _ in range(100):
                if time_up():
                    break
                subset = random.sample(deg_sorted[:min(22, n)], k)
                paths = max_flow_disjoint_paths(subset)
                if paths is not None:
                    best = {'branches': [int(x) for x in sorted(subset)], 'paths': [[int(x) for x in p] for p in paths]}
                    best_size = k
                    break

    return best

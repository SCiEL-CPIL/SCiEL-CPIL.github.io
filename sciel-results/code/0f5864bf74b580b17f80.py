import time
import random
from itertools import combinations
from collections import deque
import networkx as nx

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t_start = time.time()
    DEADLINE = 82.0
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[v]) for v in range(n)]
    sorted_verts = sorted(range(n), key=lambda x: degrees[x], reverse=True)

    def time_up():
        return time.time() - t_start > DEADLINE

    def find_shortest_path(u, v, banned):
        if u == v:
            return None
        if v in adj[u]:
            return [u, v]
        parent = {u: None}
        q = deque([u])
        while q:
            level_size = len(q)
            for _ in range(level_size):
                cur = q.popleft()
                for nxt in adj[cur]:
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

    def find_path_any(u, v, banned):
        if u == v:
            return None
        if v in adj[u]:
            return [u, v]
        parent = {u: None}
        q = deque([u])
        while q:
            cur = q.popleft()
            for nxt in adj[cur]:
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

    def max_flow_check(branches):
        """Check if vertex-disjoint paths exist for all pairs using max-flow on split-vertex network.
        Returns True if feasible (necessary condition, not sufficient for extraction)."""
        bset = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        num_pairs = len(pairs)
        if num_pairs == 0:
            return True
        
        # Build flow network with vertex splitting
        # Each vertex v -> v_in (2*v) and v_out (2*v+1)
        # Branch vertices: capacity INF on v_in -> v_out
        # Non-branch: capacity 1 on v_in -> v_out
        # Edges: v_out -> w_in with capacity 1 for each undirected edge (v,w)
        
        INF = 10**9
        total_nodes = 2 * n
        # Flow network: adjacency list with capacities
        # Use dict-of-dicts for residual graph
        cap = [[0] * total_nodes for _ in range(total_nodes)]
        
        for v in range(n):
            if v in bset:
                cap[2*v][2*v+1] = INF
            else:
                cap[2*v][2*v+1] = 1
        
        for u, v in edges:
            cap[2*u+1][2*v] = 1
            cap[2*v+1][2*u] = 1
        
        # For each pair, we need 1 unit of flow from a_out to b_in
        # Use a super-source and super-sink approach:
        # Actually, we need simultaneous flows. Let's use multi-commodity which is hard.
        # Instead, use a simpler necessary condition: for each pair, check if a path exists
        # in the residual after accounting for other pairs. This is complex.
        # 
        # Simpler approach: just check connectivity in the graph with branch vertices removed
        # from interiors. For each pair (a,b), there must be a path from a to b that doesn't
        # go through any other branch vertex.
        for a, b in pairs:
            # BFS from a to b avoiding all other branches
            banned = bset - {a, b}
            if not find_shortest_path(a, b, banned):
                return False
        return True

    def try_find(branches, rng, attempts=80):
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        def pair_key(p):
            a, b = p
            if b in adj[a]:
                return (0, degrees[a] + degrees[b])
            else:
                return (1, -(degrees[a] + degrees[b]))
        deg_order = sorted(pairs, key=pair_key)

        for attempt in range(attempts):
            if time_up():
                break
            if attempt == 0:
                order = list(deg_order)
            elif attempt % 4 == 1:
                order = list(deg_order)
                order.reverse()
            elif attempt % 4 == 2:
                order = sorted(pairs, key=lambda p: degrees[p[0]] + degrees[p[1]])
            else:
                order = list(pairs)
                rng.shuffle(order)

            used = set()
            paths = {}
            ok = True
            for a, b in order:
                banned = used | bset
                banned.discard(a)
                banned.discard(b)
                p = find_shortest_path(a, b, banned)
                if p is None:
                    ok = False
                    break
                interior = set(p[1:-1])
                if interior & bset or interior & used:
                    ok = False
                    break
                used |= interior
                paths[frozenset((a, b))] = p
            if ok:
                return paths
        return None

    def try_find_mixed(branches, rng, attempts=60):
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        def pair_key(p):
            a, b = p
            if b in adj[a]:
                return (0, degrees[a] + degrees[b])
            else:
                return (1, -(degrees[a] + degrees[b]))
        deg_order = sorted(pairs, key=pair_key)

        for attempt in range(attempts):
            if time_up():
                break
            if attempt % 3 == 0:
                order = list(deg_order)
            elif attempt % 3 == 1:
                order = list(pairs)
                rng.shuffle(order)
            else:
                order = sorted(pairs, key=lambda p: degrees[p[0]] + degrees[p[1]])

            used = set()
            paths = {}
            ok = True
            for a, b in order:
                banned = used | bset
                banned.discard(a)
                banned.discard(b)
                if attempt % 2 == 0:
                    p = find_shortest_path(a, b, banned)
                else:
                    p = find_path_any(a, b, banned)
                if p is None:
                    ok = False
                    break
                interior = set(p[1:-1])
                if interior & bset or interior & used:
                    ok = False
                    break
                used |= interior
                paths[frozenset((a, b))] = p
            if ok:
                return paths
        return None

    def validate(branches, paths):
        if not branches or len(branches) < 2:
            return 0
        bset = set(branches)
        if len(bset) != len(branches):
            return 0
        expected = set(frozenset((a, b)) for a, b in combinations(branches, 2))
        if len(paths) != len(expected):
            return 0
        used = set()
        for fkey, p in paths.items():
            if len(p) < 2:
                return 0
            a, b = p[0], p[-1]
            if a not in bset or b not in bset or a == b:
                return 0
            if fkey not in expected:
                return 0
            interior = set(p[1:-1])
            if interior & bset or interior & used:
                return 0
            for i in range(len(p) - 1):
                if p[i+1] not in adj[p[i]]:
                    return 0
            used |= interior
        return len(branches)

    def final_result(branches, paths):
        branch_list = [int(b) for b in sorted(branches)]
        path_list = []
        for a, b in combinations(sorted(branches), 2):
            fkey = frozenset((a, b))
            p = paths.get(fkey)
            if p is None:
                p = [a, b]
            path_list.append([int(x) for x in p])
        return {'branches': branch_list, 'paths': path_list}

    rng = random.Random(42)
    best_count = 0
    best_result = None

    # Phase 1: Top-down k search with flow pre-filtering
    for k in range(12, 1, -1):
        if time_up():
            break
        candidates = []
        seen_sets = set()

        # Top k by degree
        c = tuple(sorted(sorted_verts[:k]))
        candidates.append(sorted_verts[:k])
        seen_sets.add(frozenset(c))

        # Top k-1 + random (more samples)
        for _ in range(10):
            top_k1 = sorted_verts[:k-1]
            rest = [v for v in range(n) if v not in top_k1]
            if rest:
                c = sorted(top_k1 + [rng.choice(rest)])
                fs = frozenset(c)
                if fs not in seen_sets:
                    candidates.append(c)
                    seen_sets.add(fs)

        # Random (more samples)
        for _ in range(50):
            c = sorted(rng.sample(range(n), k))
            fs = frozenset(c)
            if fs not in seen_sets:
                candidates.append(c)
                seen_sets.add(fs)

        # Medium-degree mix
        for _ in range(6):
            half = k // 2
            c1 = sorted_verts[:half]
            c2 = sorted_verts[n//2:n//2 + (k - half)]
            c = sorted(c1 + c2)
            fs = frozenset(c)
            if fs not in seen_sets:
                candidates.append(c)
                seen_sets.add(fs)

        # Near-clique: pick k vertices that are mutually well-connected
        for _ in range(8):
            start = rng.choice(range(n))
            # Greedily add most-connected-to-current-set vertices
            current = [start]
            cset = {start}
            for _ in range(k - 1):
                best_v = None
                best_score = -1
                for v in range(n):
                    if v in cset:
                        continue
                    score = sum(1 for u in cset if v in adj[u])
                    if score > best_score:
                        best_score = score
                        best_v = v
                if best_v is not None and best_score > 0:
                    current.append(best_v)
                    cset.add(best_v)
                else:
                    # Fill randomly
                    remaining = [v for v in range(n) if v not in cset]
                    if remaining:
                        current.append(rng.choice(remaining))
                        cset.add(current[-1])
            c = sorted(current)
            fs = frozenset(c)
            if fs not in seen_sets:
                candidates.append(c)
                seen_sets.add(fs)

        for cand in candidates:
            if time_up():
                break
            # Pre-filter with connectivity check
            if not max_flow_check(cand):
                continue
            # Try path extraction
            paths = try_find(cand, rng, attempts=80)
            if paths is None:
                paths = try_find_mixed(cand, rng, attempts=40)
            if paths is not None:
                cnt = validate(cand, paths)
                if cnt > best_count:
                    best_count = cnt
                    best_result = (cand, paths)
                    break
        if best_result is not None and best_count >= k:
            break

    # Phase 2: Greedy growth
    if not time_up():
        if best_result is not None:
            current = list(best_result[0])
        else:
            current = sorted_verts[:2]
        current_paths = try_find(current, rng, attempts=80)
        for _ in range(15):
            if time_up():
                break
            if current_paths is None:
                break
            cset = set(current)
            cands = [v for v in range(n) if v not in cset]
            cands.sort(key=lambda v: degrees[v], reverse=True)
            added = False
            for v in cands[:20]:
                if time_up():
                    break
                new_br = current + [v]
                if not max_flow_check(new_br):
                    continue
                new_paths = try_find(new_br, rng, attempts=40)
                if new_paths is None:
                    new_paths = try_find_mixed(new_br, rng, attempts=30)
                if new_paths is not None:
                    cnt = validate(new_br, new_paths)
                    if cnt > best_count:
                        best_count = cnt
                        best_result = (new_br, new_paths)
                        current = new_br
                        current_paths = new_paths
                        added = True
                        break
            if not added:
                break

    # Fallback
    if best_result is None:
        found = False
        for u, v in edges:
            paths = {frozenset((u, v)): [u, v]}
            if validate([u, v], paths) > 0:
                best_result = ([u, v], paths)
                found = True
                break
        if not found:
            u, v = edges[0]
            best_result = ([u, v], {frozenset((u, v)): [u, v]})

    branches, paths = best_result
    return final_result(branches, paths)

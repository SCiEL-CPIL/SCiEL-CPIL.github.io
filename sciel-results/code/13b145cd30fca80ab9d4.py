"""Combined: incremental greedy growth + max-flow verification for vertex-disjoint paths."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t_start = time.time()
    DEADLINE = 80.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[v]) for v in range(n)]
    sorted_verts = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    rng = random.Random(12345)

    def time_up():
        return time.time() - t_start > DEADLINE

    # ---- Max-flow verification (Ford-Fulkerson BFS) ----
    # Vertex splitting: v_in = 2*v, v_out = 2*v+1
    # Non-branch: v_in -> v_out cap 1
    # Branch: v_in -> v_out cap k (can be endpoint of up to k-1 paths, but cap k is safe)
    # Original edge (u,v): u_out -> v_in cap 1, v_out -> u_in cap 1
    # Super source S, super sink T.
    # For each pair (a,b) of branches: S -> a_in cap 1, b_out -> T cap 1
    # If max flow == number of pairs, we can extract vertex-disjoint paths.

    def max_flow_verify(branches):
        """Return dict mapping frozenset({a,b}) -> path, or None if infeasible."""
        bset = set(branches)
        k = len(branches)
        if k < 2:
            return None
        pairs = list(combinations(sorted(branches), 2))
        num_pairs = len(pairs)

        # Node indices: 2*v = v_in, 2*v+1 = v_out
        # S = 2*n, T = 2*n+1
        S = 2 * n
        T = 2 * n + 1
        total_nodes = 2 * n + 2

        # Adjacency list for flow network: adj_flow[u] = list of (v, cap)
        # We'll use a residual graph with parallel edges handled by index pairs.
        # Use list of [to, cap, rev_index]
        flow_adj = [[] for _ in range(total_nodes)]

        def add_edge(u, v, cap):
            flow_adj[u].append([v, cap, len(flow_adj[v])])
            flow_adj[v].append([u, 0, len(flow_adj[u]) - 1])

        # Vertex capacity edges
        for v in range(n):
            if v in bset:
                add_edge(2 * v, 2 * v + 1, k)  # branches can be endpoints of many paths
            else:
                add_edge(2 * v, 2 * v + 1, 1)  # non-branches: capacity 1

        # Original graph edges (directed both ways, cap 1)
        for u, v in edges:
            add_edge(2 * u + 1, 2 * v, 1)
            add_edge(2 * v + 1, 2 * u, 1)

        # Source/sink edges for each pair
        # We need to track which flow unit corresponds to which pair.
        # Strategy: add edges S->a_in and b_out->T for each pair, and track
        # which pair each source edge belongs to.
        pair_source_edge = []  # (pair_idx, from_node, to_node)
        for pi, (a, b) in enumerate(pairs):
            add_edge(S, 2 * a, 1)
            add_edge(2 * b + 1, T, 1)
            pair_source_edge.append((pi, S, 2 * a))

        # Run Ford-Fulkerson with BFS (Edmonds-Karp)
        total_flow = 0
        while True:
            if time_up():
                return None
            # BFS to find augmenting path
            parent = [-1] * total_nodes
            parent_edge = [-1] * total_nodes  # index into flow_adj[parent[node]]
            parent[S] = S
            queue = deque([S])
            found = False
            while queue:
                u = queue.popleft()
                if u == T:
                    found = True
                    break
                for ei, (v, cap, rev) in enumerate(flow_adj[u]):
                    if cap > 0 and parent[v] == -1:
                        parent[v] = u
                        parent_edge[v] = ei
                        queue.append(v)
            if not found:
                break
            # Augment
            path_nodes = []
            node = T
            while node != S:
                path_nodes.append(node)
                node = parent[node]
            path_nodes.append(S)
            path_nodes.reverse()

            # Find bottleneck
            bottleneck = float('inf')
            for i in range(len(path_nodes) - 1):
                u = path_nodes[i]
                v = path_nodes[i + 1]
                # Find edge u->v
                for ei, (vv, cap, rev) in enumerate(flow_adj[u]):
                    if vv == v and cap > 0:
                        bottleneck = min(bottleneck, cap)
                        break
            # Apply augmentation
            for i in range(len(path_nodes) - 1):
                u = path_nodes[i]
                v = path_nodes[i + 1]
                for ei, (vv, cap, rev) in enumerate(flow_adj[u]):
                    if vv == v and cap > 0:
                        flow_adj[u][ei][1] -= bottleneck
                        flow_adj[v][rev][1] += bottleneck
                        break
            total_flow += bottleneck

        if total_flow != num_pairs:
            return None

        # Extract paths from the flow
        # For each pair, find the path in the residual graph that carries flow
        # We need to trace: S -> a_in -> ... -> b_out -> T
        # The flow is 1 on each S->a_in edge and b_out->T edge.
        # We trace through the vertex-splitting network.

        paths = {}
        for pi, (a, b) in enumerate(pairs):
            # Trace from a_in (2*a) to b_out (2*b+1) through the network
            # Start: a_in = 2*a, we know flow went S->2*a (cap was 1, now 0)
            # From 2*a, flow went to 2*a+1 (vertex capacity edge)
            # Then through original edges...
            # End: 2*b+1 -> T

            # BFS/DFS in the residual graph following edges with flow
            # (i.e., the reverse edge has cap > 0, meaning original had flow)
            # Actually, we trace forward: an edge (u,v) has flow if flow_adj[v][rev][1] > 0
            # where rev is the index of the reverse edge in flow_adj[v]

            # Let's trace: start at 2*a (a_in)
            # The path in the split graph: 2*a -> 2*a+1 -> ... -> 2*b -> 2*b+1
            # Each transition is either a vertex-capacity edge or an original-edge

            # Trace using the flow
            path_split = [2 * a]
            cur = 2 * a
            visited_split = {2 * a}
            while cur != 2 * b + 1:
                # Find outgoing edge from cur with flow
                found_next = False
                for ei, (v, cap, rev) in enumerate(flow_adj[cur]):
                    # Edge has flow if reverse edge has residual > 0
                    if flow_adj[v][rev][1] > 0:
                        if v not in visited_split:
                            path_split.append(v)
                            visited_split.add(v)
                            cur = v
                            found_next = True
                            break
                if not found_next:
                    return None  # shouldn't happen if flow is valid
            # Convert split path to original path
            # 2*v = v_in, 2*v+1 = v_out
            # The path alternates: v_in -> v_out (vertex edge), v_out -> w_in (original edge)
            # Original vertices: for each node x in path_split, vertex = x // 2
            # But we need to deduplicate consecutive same vertices
            orig_path = []
            for x in path_split:
                vert = x // 2
                if not orig_path or orig_path[-1] != vert:
                    orig_path.append(vert)
            # Verify endpoints
            if orig_path[0] != a or orig_path[-1] != b:
                return None
            paths[frozenset((a, b))] = orig_path

        return paths

    # ---- Greedy BFS fallback (for small cases or as backup) ----
    def find_path_bfs(s, t, blocked):
        if s == t:
            return None
        if t in adj[s]:
            return [s, t]
        parent = {s: None}
        q = deque([s])
        while q:
            cur = q.popleft()
            for nxt in sorted(adj[cur]):
                if nxt == t:
                    path = [t]
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                if nxt in parent or nxt in blocked:
                    continue
                parent[nxt] = cur
                q.append(nxt)
        return None

    def greedy_verify(branches):
        """Greedy BFS path packing with multiple orderings."""
        bset = set(branches)
        k = len(branches)
        if k < 2:
            return None
        pairs = list(combinations(sorted(branches), 2))

        # Multiple orderings
        orderings = []
        orderings.append(sorted(pairs, key=lambda p: degrees[p[0]] + degrees[p[1]]))
        orderings.append(sorted(pairs, key=lambda p: -(degrees[p[0]] + degrees[p[1]])))
        orderings.append(list(pairs))
        orderings.append(list(reversed(pairs)))

        for order in orderings:
            if time_up():
                break
            used_interior = set()
            paths = {}
            success = True
            for a, b in order:
                banned = used_interior | bset
                banned.discard(a)
                banned.discard(b)
                path = find_path_bfs(a, b, banned)
                if path is None:
                    success = False
                    break
                interior = set(path[1:-1])
                if interior & bset or interior & used_interior:
                    success = False
                    break
                used_interior |= interior
                paths[frozenset((a, b))] = path
            if success:
                return paths
        return None

    # ---- Verification (both methods) ----
    def verify(branches):
        """Try max-flow first, then greedy BFS as fallback."""
        if len(branches) < 2:
            return None
        # Max-flow is exact; use it first
        paths = max_flow_verify(branches)
        if paths is not None:
            return paths
        # Fallback to greedy (sometimes greedy finds solutions flow misses due to
        # the specific flow decomposition, though this should be rare)
        paths = greedy_verify(branches)
        return paths

    def validate(branches, paths):
        """Validate a solution. Return count or 0."""
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
                if path[i + 1] not in adj[path[i]]:
                    return 0
            used_interior |= interior
        return len(branches)

    # ---- Main search ----
    best_count = 0
    best_result = None

    # Phase 1: Incremental greedy growth from high-degree seeds
    # Start with top-2 by degree, try to grow
    for seed_k in range(2, 5):
        if time_up():
            break
        seed = sorted_verts[:seed_k]
        paths = verify(seed)
        if paths is not None:
            cnt = validate(seed, paths)
            if cnt > best_count:
                best_count = cnt
                best_result = (list(seed), paths)

    # Phase 2: Grow the best solution incrementally
    if best_result is not None:
        current = list(best_result[0])
        current_paths = best_result[1]
    else:
        current = sorted_verts[:2]
        current_paths = verify(current)
        if current_paths is not None:
            best_result = (current, current_paths)
            best_count = len(current)

    if best_result is not None:
        improved = True
        while improved and not time_up():
            improved = False
            current_set = set(current)
            # Try adding each non-branch vertex, sorted by degree
            candidates = [v for v in sorted_verts if v not in current_set]
            for v in candidates:
                if time_up():
                    break
                new_branches = current + [v]
                new_paths = verify(new_branches)
                if new_paths is not None:
                    cnt = validate(new_branches, new_paths)
                    if cnt > best_count:
                        best_count = cnt
                        best_result = (list(new_branches), new_paths)
                        current = new_branches
                        current_paths = new_paths
                        improved = True
                        break  # restart with new current

    # Phase 3: Try to swap in vertices for improvement
    if best_result is not None and not time_up():
        current = list(best_result[0])
        for _ in range(3):  # a few rounds of swap
            if time_up():
                break
            current_set = set(current)
            swapped = False
            # Try removing one branch and adding two
            for remove_v in list(current):
                if time_up():
                    break
                reduced = [v for v in current if v != remove_v]
                if len(reduced) < 2:
                    continue
                reduced_set = set(reduced)
                for add_v1 in [v for v in sorted_verts if v not in reduced_set]:
                    if time_up():
                        break
                    for add_v2 in [v for v in sorted_verts if v not in reduced_set and v != add_v1]:
                        if time_up():
                            break
                        new_branches = reduced + [add_v1, add_v2]
                        if len(new_branches) <= best_count:
                            continue
                        new_paths = verify(new_branches)
                        if new_paths is not None:
                            cnt = validate(new_branches, new_paths)
                            if cnt > best_count:
                                best_count = cnt
                                best_result = (list(new_branches), new_paths)
                                current = new_branches
                                swapped = True
                                break
                    if swapped:
                        break
                if swapped:
                    break
            if not swapped:
                break

    # Phase 4: Random restarts for diversity
    if not time_up():
        for restart in range(20):
            if time_up():
                break
            # Random seed of size 3-5
            seed_size = rng.randint(3, 5)
            seed = rng.sample(range(n), seed_size)
            paths = verify(seed)
            if paths is not None:
                cnt = validate(seed, paths)
                if cnt > best_count:
                    best_count = cnt
                    best_result = (list(seed), paths)
                    # Try to grow this
                    current = list(seed)
                    improved = True
                    while improved and not time_up():
                        improved = False
                        current_set = set(current)
                        for v in sorted_verts:
                            if time_up():
                                break
                            if v in current_set:
                                continue
                            new_branches = current + [v]
                            new_paths = verify(new_branches)
                            if new_paths is not None:
                                cnt2 = validate(new_branches, new_paths)
                                if cnt2 > best_count:
                                    best_count = cnt2
                                    best_result = (list(new_branches), new_paths)
                                    current = new_branches
                                    improved = True
                                    break

    # Final: if no result, use a trivial 2-branch solution
    if best_result is None:
        # Find any edge
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    branches, paths = best_result
    branches_list = sorted([int(b) for b in branches])
    # Order paths by (min_branch, max_branch)
    paths_ordered = []
    for pair in combinations(branches_list, 2):
        key = frozenset(pair)
        if key in paths:
            p = [int(v) for v in paths[key]]
            # Ensure path starts with min and ends with max for consistency
            if p[0] > p[-1]:
                p = p[::-1]
            paths_ordered.append(p)
        else:
            paths_ordered.append([int(pair[0]), int(pair[1])])

    return {'branches': branches_list, 'paths': paths_ordered}

# EVOLVE-BLOCK-END

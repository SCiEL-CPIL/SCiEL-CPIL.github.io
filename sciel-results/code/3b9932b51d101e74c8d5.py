"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
from itertools import combinations
from collections import deque
import random as _random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [sorted(adj[i]) for i in range(n)]

    deadline = _time.time() + 55

    # --- Max-flow based feasibility check using vertex splitting ---
    # For a set of k branches, we need k-1 internally vertex-disjoint paths
    # from each branch to a "hub". We model this as a flow problem.
    #
    # Key idea: A set of branches B is feasible iff there exists a set of
    # internally-disjoint paths connecting every pair. By Menger's theorem,
    # this is equivalent to: for any partition of B into two non-empty parts,
    # there are at least |B1|*|B2| internally-disjoint paths crossing the cut.
    #
    # Practical approach: Use max-flow to test if we can route k-1 disjoint
    # paths from branch[0] to each other branch (star topology), then verify
    # the full pairwise requirement.

    def max_flow_vertex_disjoint(sources, target, forbidden_vertices, num_paths):
        """
        Find up to num_paths internally vertex-disjoint paths from any source
        to target, avoiding forbidden_vertices (except sources and target).
        Uses vertex-splitting: each vertex v becomes v_in -> v_out with capacity 1.
        Returns list of paths or None if not enough disjoint paths.
        """
        if not sources:
            return []
        sources = list(sources)
        # Check if target is reachable at all
        # Build flow network
        # Vertex splitting: v_in = 2*v, v_out = 2*v+1
        # Edges: v_in -> v_out (cap 1), for each edge (u,v): u_out -> v_in (cap 1)
        # Source: connect super_source to each s_in (cap 1)
        # Sink: t_out

        # Nodes: 2*n + 2 (super_source, super_sink)
        SS = 2 * n
        TT = 2 * n + 1
        total_nodes = 2 * n + 2

        # Build adjacency for flow network
        # flow_adj[node] = list of (neighbor, capacity, edge_index)
        flow_adj = [[] for _ in range(total_nodes)]
        flow_rev = [0] * total_nodes  # not used, we'll use edge pairs

        def add_edge(u, v, cap):
            flow_adj[u].append([v, cap, len(flow_adj[v])])
            flow_adj[v].append([u, 0, len(flow_adj[u]) - 1])

        # Add vertex split edges
        for v in range(n):
            if v in forbidden_vertices:
                continue  # skip forbidden vertices entirely
            v_in = 2 * v
            v_out = 2 * v + 1
            # Capacity 1 for non-source, non-target vertices
            # Sources and target get higher capacity
            cap = 1
            if v == target:
                cap = num_paths
            elif v in sources:
                cap = num_paths
            add_edge(v_in, v_out, cap)

        # Add graph edges
        for u, v in edges:
            if u in forbidden_vertices or v in forbidden_vertices:
                continue
            u_out = 2 * u + 1
            v_in = 2 * v
            v_out = 2 * v + 1
            u_in = 2 * u
            # u_out -> v_in
            add_edge(u_out, v_in, 1)
            # v_out -> u_in
            add_edge(v_out, u_in, 1)

        # Connect super source to sources
        for s in sources:
            if s in forbidden_vertices:
                continue
            s_in = 2 * s
            add_edge(SS, s_in, 1)

        # Connect target to super sink
        t_out = 2 * target + 1
        add_edge(t_out, TT, num_paths)

        # Run Ford-Fulkerson with BFS (Edmonds-Karp)
        def bfs():
            parent = [-1] * total_nodes
            parent[SS] = SS
            q = deque([SS])
            while q:
                u = q.popleft()
                for i, (v, cap, _) in enumerate(flow_adj[u]):
                    if cap > 0 and parent[v] == -1:
                        parent[v] = u
                        if v == TT:
                            return parent
                        q.append(v)
            return None

        paths_found = []
        flow = 0
        while flow < num_paths:
            parent = bfs()
            if parent is None:
                break
            # Find bottleneck
            path_flow = num_paths - flow
            v = TT
            while v != SS:
                u = parent[v]
                for (nv, cap, rev_idx) in flow_adj[u]:
                    if nv == v:
                        path_flow = min(path_flow, cap)
                        break
                v = u
            # Augment
            v = TT
            while v != SS:
                u = parent[v]
                for i, (nv, cap, rev_idx) in enumerate(flow_adj[u]):
                    if nv == v:
                        flow_adj[u][i][1] -= path_flow
                        flow_adj[v][rev_idx][1] += path_flow
                        break
                v = u
            flow += path_flow

        # Extract paths from residual graph
        # For each unit of flow, trace back from SS
        # Actually, let's trace paths through the original flow
        # We need to reconstruct paths from the flow

        # Reconstruct: for each source, find path to target
        # Use the flow values to trace
        result_paths = []
        for s in sources:
            if s in forbidden_vertices:
                continue
            # Trace from s to target using positive flow
            path = [s]
            cur = 2 * s + 1  # s_out
            visited_nodes = {s}
            while cur != 2 * target + 1:
                found = False
                for (nv, cap, rev_idx) in flow_adj[cur]:
                    # Check if there's flow on edge cur -> nv
                    # The reverse edge nv -> cur should have positive capacity
                    # Actually, we need to check original edge flow
                    # Let's check: flow on (cur, nv) = original_cap - current_cap
                    # We stored [v, cap, rev_idx] where cap is residual
                    # The actual flow = original_cap - residual_cap
                    # But we don't store original_cap separately...
                    # Let me use a different approach: check if residual on reverse is > 0
                    pass
                if not found:
                    break
                break  # This approach is getting complex

        # Simpler: just check if flow == num_paths
        if flow >= num_paths:
            return True
        return False

    # --- Simpler feasibility check: iterative path removal ---
    def check_feasibility_flow(branches):
        """
        Check feasibility by iteratively finding and removing paths.
        For each pair of branches, find a path, remove its interior vertices,
        repeat. If we can't find a path, the set is infeasible.
        Uses BFS for path finding.
        """
        bset = set(branches)
        bsorted = sorted(branches)
        pairs = list(combinations(bsorted, 2))
        used_interiors = set()

        # Order pairs by difficulty (shortest path first)
        def path_len(u, v):
            forbidden = bset - {u, v} | used_interiors
            # BFS
            visited = {u}
            parent = {u: None}
            q = deque([u])
            while q:
                cur = q.popleft()
                if cur == v:
                    # Count path length
                    length = 0
                    c = v
                    while c != u:
                        c = parent[c]
                        length += 1
                    return length
                for nb in adj_list[cur]:
                    if nb not in visited and (nb == v or nb not in forbidden):
                        visited.add(nb)
                        parent[nb] = cur
                        q.append(nb)
            return float('inf')

        # Sort pairs by path length (hardest last)
        pairs_sorted = sorted(pairs, key=lambda p: path_len(p[0], p[1]))

        def find_path(src, dst, forbidden):
            """Find a simple path from src to dst avoiding forbidden vertices."""
            if src == dst:
                return [src]
            visited = {src}
            parent = {src: None}
            q = deque([src])
            while q:
                u = q.popleft()
                if u == dst:
                    path = []
                    cur = dst
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    return list(reversed(path))
                for v in adj_list[u]:
                    if v not in visited and (v == dst or v not in forbidden):
                        visited.add(v)
                        parent[v] = u
                        q.append(v)
            return None

        assigned = {}
        for (u, v) in pairs_sorted:
            forbidden = bset - {u, v} | used_interiors
            path = find_path(u, v, forbidden)
            if path is None:
                return None
            interior = set(path[1:-1])
            if interior & bset:
                return None
            if interior & used_interiors:
                return None
            used_interiors |= interior
            assigned[(u, v)] = path

        return assigned

    # --- Greedy growth with flow-based feasibility ---
    def try_grow(initial_branches):
        current = list(initial_branches)
        improved = True
        while improved and _time.time() < deadline:
            improved = False
            candidates = [v for v in range(n) if v not in current]
            # Score: prefer vertices adjacent to many existing branches
            def cand_score(v):
                score = sum(1 for u in current if v in adj[u]) * 10
                score += len(adj[v])
                return score
            candidates.sort(key=cand_score, reverse=True)
            for candidate in candidates:
                if _time.time() > deadline:
                    return current
                new_set = current + [candidate]
                paths = check_feasibility_flow(new_set)
                if paths is not None:
                    current = new_set
                    improved = True
                    break
        return current

    # --- Local search: add/swap moves ---
    def local_search(branches):
        """Try to improve by adding or swapping vertices."""
        current = list(branches)
        improved = True
        while improved and _time.time() < deadline:
            improved = False
            # Try adding
            candidates = [v for v in range(n) if v not in current]
            candidates.sort(key=lambda v: sum(1 for u in current if v in adj[u]), reverse=True)
            for candidate in candidates:
                if _time.time() > deadline:
                    return current
                new_set = current + [candidate]
                paths = check_feasibility_flow(new_set)
                if paths is not None:
                    current = new_set
                    improved = True
                    break
            # Try swapping
            for i in range(len(current)):
                if _time.time() > deadline:
                    return current
                for candidate in [v for v in range(n) if v not in current]:
                    if _time.time() > deadline:
                        return current
                    new_set = current[:i] + [candidate] + current[i+1:]
                    paths = check_feasibility_flow(new_set)
                    if paths is not None:
                        current = new_set
                        improved = True
                        break
                if improved:
                    break
        return current

    # --- Core-based pruning: use k-core as candidate pool ---
    def compute_k_core(k):
        """Compute k-core of the graph."""
        deg = [len(adj[i]) for i in range(n)]
        in_core = [True] * n
        # Iteratively remove vertices with degree < k
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if in_core[v] and deg[v] < k:
                    in_core[v] = False
                    changed = True
                    for nb in adj[v]:
                        if in_core[nb]:
                            deg[nb] -= 1
        return [v for v in range(n) if in_core[v]]

    # --- Maximum clique via branch and bound with coloring ---
    def max_clique_bnb():
        """Find maximum clique using branch and bound with greedy coloring."""
        best = []

        # Order vertices by degree
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        pos = {v: i for i, v in enumerate(order)}

        def color_bound(vertices):
            """Greedy coloring to get upper bound on clique size in vertices."""
            colors = {}
            for v in vertices:
                # Try to assign smallest available color
                used_colors = set()
                for nb in adj[v]:
                    if nb in colors and nb in vertices:
                        used_colors.add(colors[nb])
                c = 0
                while c in used_colors:
                    c += 1
                colors[v] = c
            return max(colors.values()) + 1 if colors else 0

        def expand(R, P):
            nonlocal best
            if not P:
                if len(R) > len(best):
                    best = list(R)
                return
            # Color bound
            P_list = sorted(P, key=lambda v: pos[v])
            bound = color_bound(P_list)
            if len(R) + bound <= len(best):
                return
            # Process vertices in reverse color order
            for i in range(len(P_list) - 1, -1, -1):
                v = P_list[i]
                if len(R) + i <= len(best):
                    break
                expand(R | {v}, P & adj[v])
                P.discard(v)

        expand(set(), set(range(n)))
        return best

    # --- Main search ---
    best_branches = None
    best_paths = None

    def update_best(b, p):
        nonlocal best_branches, best_paths
        if b is not None and p is not None and len(b) >= 2:
            if best_branches is None or len(b) > len(best_branches):
                best_branches = list(b)
                best_paths = p

    # Strategy 1: Start from max clique
    if _time.time() < deadline:
        mc = max_clique_bnb()
        if len(mc) >= 2:
            # Try to grow from clique
            grown = try_grow(mc)
            paths = check_feasibility_flow(grown)
            update_best(grown, paths)
            # Local search
            ls = local_search(grown)
            paths_ls = check_feasibility_flow(ls)
            update_best(ls, paths_ls)

    # Strategy 2: Core-based candidates
    if _time.time() < deadline:
        for k in [6, 5, 4]:
            core = compute_k_core(k)
            if len(core) >= 2:
                # Try growing from top-degree vertices in core
                core_sorted = sorted(core, key=lambda v: -len(adj[v]))
                for start_size in [2, 3, 4]:
                    if _time.time() > deadline:
                        break
                    start = core_sorted[:start_size]
                    grown = try_grow(start)
                    paths = check_feasibility_flow(grown)
                    update_best(grown, paths)

    # Strategy 3: Top-degree starts
    if _time.time() < deadline:
        deg = [len(adj[i]) for i in range(n)]
        top = sorted(range(n), key=lambda x: -deg[x])
        for start_size in [2, 3, 4, 5]:
            if _time.time() > deadline:
                break
            start = top[:start_size]
            grown = try_grow(start)
            paths = check_feasibility_flow(grown)
            update_best(grown, paths)

    # Strategy 4: Random restarts
    if _time.time() < deadline:
        rng = _random.Random(123)
        for _ in range(30):
            if _time.time() > deadline:
                break
            start_size = rng.randint(2, 5)
            start = rng.sample(range(n), start_size)
            grown = try_grow(start)
            paths = check_feasibility_flow(grown)
            update_best(grown, paths)

    # Strategy 5: All pairs of high-degree
    if _time.time() < deadline:
        high_deg = [v for v in range(n) if len(adj[v]) >= 5]
        for i in range(len(high_deg)):
            if _time.time() > deadline:
                break
            for j in range(i + 1, len(high_deg)):
                if _time.time() > deadline:
                    break
                seed = [high_deg[i], high_deg[j]]
                grown = try_grow(seed)
                paths = check_feasibility_flow(grown)
                update_best(grown, paths)

    # Fallback
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = {(u, v): [u, v]}

    # Format output
    branches = [int(b) for b in sorted(best_branches)]
    paths_list = []
    bset = set(branches)
    for (u, v) in combinations(branches, 2):
        path = best_paths.get((u, v)) or best_paths.get((v, u))
        if path is None:
            # Find path avoiding other branches and already-used interiors
            path = find_path(u, v, bset - {u, v})
            if path is None:
                path = [u, v]
        paths_list.append([int(x) for x in path])

    return {'branches': branches, 'paths': paths_list}
# EVOLVE-BLOCK-END

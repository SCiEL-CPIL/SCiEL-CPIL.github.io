"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    deadline = time.time() + 80  # 80s safety margin

    # --- Max-flow feasibility oracle using vertex-splitting ---
    # For a set of branches B, we check if vertex-disjoint paths exist
    # between every pair of branches.
    #
    # Construction:
    #   - Each original vertex v becomes v_in -> v_out (capacity 1)
    #   - Each edge (u,v) becomes u_out -> v_in (capacity 1) and v_out -> u_in (capacity 1)
    #   - For branch vertices, capacity is k-1 (they can be endpoints of k-1 paths)
    #   - For non-branch vertices, capacity is 1 (interior of at most one path)
    #   - Add a super-source S connected to each branch_in with capacity k-1
    #   - Add a super-sink T connected from each branch_out with capacity k-1
    #   - If max flow = k*(k-1), then vertex-disjoint paths exist for all pairs.
    #
    # To recover actual paths, we run max-flow and trace the flow.

    def max_flow_feasible(branches):
        """Check if vertex-disjoint paths exist for all pairs of branches.
        Returns (feasible, flow_network) where flow_network is a dict of edges with residual capacity."""
        k = len(branches)
        if k < 2:
            return False, None

        branch_set = set(branches)

        # Node indexing: v_in = 2*v, v_out = 2*v+1
        # Super source = 2*n, super sink = 2*n+1
        S = 2 * n
        T = 2 * n + 1
        total_nodes = 2 * n + 2

        # Build adjacency list for residual graph
        # adj_flow[u] = list of (v, capacity)
        adj_flow = defaultdict(list)
        # We also need reverse edges for residual updates
        # Use a dict: (u, v) -> index in adj_flow[u] for the reverse edge
        # Actually, let's use a simpler approach: store edges as list of [to, cap, rev_idx]

        def add_edge(u, v, cap):
            adj_flow[u].append([v, cap, len(adj_flow[v])])
            adj_flow[v].append([u, 0, len(adj_flow[u]) - 1])

        # Add vertex-split edges
        for v in range(n):
            if v in branch_set:
                cap = k - 1
            else:
                cap = 1
            add_edge(2 * v, 2 * v + 1, cap)

        # Add graph edges (bidirectional, capacity 1 each direction)
        for u, v in edges:
            add_edge(2 * u + 1, 2 * v, 1)
            add_edge(2 * v + 1, 2 * u, 1)

        # Add super source and sink connections
        for b in branches:
            add_edge(S, 2 * b, k - 1)
            add_edge(2 * b + 1, T, k - 1)

        # Run max-flow (Edmonds-Karp for simplicity, graph is small)
        def edmonds_karp():
            flow = 0
            while True:
                # BFS to find augmenting path
                parent = [-1] * total_nodes
                parent_edge = [-1] * total_nodes
                parent[S] = S
                queue = deque([S])
                found = False
                while queue:
                    u = queue.popleft()
                    if u == T:
                        found = True
                        break
                    for idx, (v, cap, _) in enumerate(adj_flow[u]):
                        if cap > 0 and parent[v] == -1:
                            parent[v] = u
                            parent_edge[v] = idx
                            queue.append(v)
                if not found:
                    break
                # Find bottleneck
                bottleneck = float('inf')
                v = T
                while v != S:
                    u = parent[v]
                    idx = parent_edge[v]
                    bottleneck = min(bottleneck, adj_flow[u][idx][1])
                    v = u
                # Update residual
                v = T
                while v != S:
                    u = parent[v]
                    idx = parent_edge[v]
                    adj_flow[u][idx][1] -= bottleneck
                    rev_idx = adj_flow[u][idx][2]
                    adj_flow[v][rev_idx][1] += bottleneck
                    v = u
                flow += bottleneck

            return flow

        maxf = edmonds_karp()
        required = k * (k - 1)

        return maxf >= required, adj_flow

    def extract_paths(branches, adj_flow):
        """Extract vertex-disjoint paths from the flow network.
        Returns dict mapping (u,v) -> path, or None if extraction fails."""
        k = len(branches)
        branch_set = set(branches)
        S = 2 * n
        T = 2 * n + 1

        # For each branch b, there are k-1 units of flow from S -> b_in.
        # We need to trace each unit of flow to its destination branch.
        #
        # Strategy: For each branch b, look at the k-1 units of flow leaving b_out.
        # Each unit goes through some path to another branch c's c_in.
        # We trace the path in the original graph.

        # First, identify which edges carry flow.
        # An edge (u, v) in the flow network carries flow if adj_flow[v][rev][1] > 0
        # where rev is the reverse edge index. Actually, the residual capacity of
        # the reverse edge tells us the flow on the forward edge.

        # Let's collect all edges with positive flow.
        # For each node u, for each edge (u, v, cap, rev_idx) in adj_flow[u]:
        #   flow = adj_flow[v][rev_idx][1]  (this is the reverse edge's capacity = flow sent)
        #   Wait, that's not right. Let me think again.
        #
        # When we add_edge(u, v, cap):
        #   adj_flow[u].append([v, cap, len(adj_flow[v])])
        #   adj_flow[v].append([u, 0, len(adj_flow[u]) - 1])
        #
        # So the forward edge is adj_flow[u][i] = [v, cap, j]
        # and the reverse edge is adj_flow[v][j] = [u, 0, i]
        #
        # After flow, the forward edge's capacity decreases and the reverse edge's capacity increases.
        # So flow on forward edge = original_cap - adj_flow[u][i][1]
        # Or equivalently, flow = adj_flow[v][j][1] (the reverse edge's current capacity)

        # Collect flow edges
        flow_edges = defaultdict(int)  # (u, v) -> flow
        for u in range(2 * n + 2):
            for idx, (v, cap, rev_idx) in enumerate(adj_flow[u]):
                rev_cap = adj_flow[v][rev_idx][1]
                if rev_cap > 0:
                    flow_edges[(u, v)] = rev_cap

        # Now trace paths. For each branch b, there are k-1 units of flow from S to b_in (2*b).
        # From b_out (2*b+1), k-1 units of flow go out.
        # We need to trace each unit to its destination.

        # Let's trace from each branch's out-node.
        # For branch b, look at edges from 2*b+1 with positive flow.
        # Each such edge leads to some vertex's in-node, and we follow the path.

        # Build a mapping: for each node, list of (next_node, flow) for flow edges
        out_flow = defaultdict(list)
        for (u, v), f in flow_edges.items():
            for _ in range(f):
                out_flow[u].append(v)

        # Trace paths from each branch
        paths_dict = {}
        for b in branches:
            b_out = 2 * b + 1
            # Get all flow destinations from b_out
            destinations = out_flow[b_out]
            for dest_in in destinations:
                # dest_in should be 2*c for some vertex c
                c = dest_in // 2
                # Trace the path from b to c
                path = [b]
                cur_in = dest_in  # we're at c_in
                # We need to trace backwards? No, let's trace forward from b_out.
                # Actually, the flow goes: b_out -> some_in -> some_out -> ... -> c_in
                # Let's trace from b_out following flow edges.

                # Reset: trace from b_out
                path = [b]
                cur = b_out
                while cur != 2 * c:  # 2*c is c_in
                    # Find next node with flow
                    nexts = out_flow[cur]
                    if not nexts:
                        return None
                    # Pick the first one (should be deterministic if we're careful)
                    nxt = nexts.pop(0)
                    # Determine which vertex this corresponds to
                    if nxt % 2 == 0:  # in-node
                        vertex = nxt // 2
                        path.append(vertex)
                        cur = nxt  # now at vertex_in, next should be vertex_out
                    else:  # out-node (shouldn't happen directly)
                        return None
                    # From in-node, go to out-node
                    cur = cur + 1  # vertex_in -> vertex_out
                    # Check if cur is 2*c (c_in)
                    if cur == 2 * c:
                        break

                # cur should now be 2*c (c_in)
                if cur != 2 * c:
                    return None

                # path should start with b and end with c
                if path[0] != b or path[-1] != c:
                    return None

                # Normalize: store as (min(b,c), max(b,c))
                key = (min(b, c), max(b, c))
                if key in paths_dict:
                    return None  # duplicate
                paths_dict[key] = list(path)

        # Verify all pairs are covered
        for u, v in combinations(branches, 2):
            key = (min(u, v), max(u, v))
            if key not in paths_dict:
                return None

        return paths_dict

    def verify_paths(branches, paths_dict):
        """Verify that the paths are valid: correct endpoints, interiors disjoint, avoid branches."""
        branch_set = set(branches)
        used_interiors = set()
        for (u, v), path in paths_dict.items():
            if path[0] != u and path[0] != v:
                return False
            if path[-1] != u and path[-1] != v:
                return False
            if path[0] == path[-1]:
                return False
            # Check interiors
            for w in path[1:-1]:
                if w in branch_set:
                    return False
                if w in used_interiors:
                    return False
                used_interiors.add(w)
            # Check adjacency
            for i in range(len(path) - 1):
                if path[i + 1] not in adj[path[i]]:
                    return False
        return True

    def try_clique_subdivision(branches):
        """Try to find vertex-disjoint paths for all pairs in branches using max-flow.
        Returns dict with 'branches' and 'paths' if successful, else None."""
        branch_set = set(branches)
        k = len(branches)
        if k < 2:
            return None

        feasible, adj_flow = max_flow_feasible(branches)
        if not feasible:
            return None

        paths_dict = extract_paths(branches, adj_flow)
        if paths_dict is None:
            return None

        if not verify_paths(branches, paths_dict):
            return None

        result_paths = []
        for u, v in combinations(branches, 2):
            key = (min(u, v), max(u, v))
            result_paths.append(list(paths_dict[key]))

        return {'branches': list(branches), 'paths': result_paths}

    # --- Bron-Kerbosch to find maximal cliques ---
    def bron_kerbosch(R, P, X, cliques):
        if time.time() > deadline:
            return
        if not P and not X:
            cliques.append(list(R))
            return
        candidates = list(P | X)
        if not candidates:
            return
        pivot = max(candidates, key=lambda v: len(adj[v] & P))
        for v in list(P - adj[pivot]):
            bron_kerbosch(R | {v}, P & adj[v], X & adj[v], cliques)
            P.remove(v)
            X.add(v)

    # Find all maximal cliques
    all_cliques = []
    bron_kerbosch(set(), set(range(n)), set(), all_cliques)
    all_cliques.sort(key=len, reverse=True)

    best_result = None
    best_size = 0

    # Strategy 1: Try cliques from largest to smallest
    for clique in all_cliques:
        if len(clique) <= best_size:
            break
        if time.time() > deadline:
            break
        result = try_clique_subdivision(clique)
        if result is not None:
            best_result = result
            best_size = len(clique)

    # Strategy 2: Greedy extension from a small feasible set
    # Start with a pair, then try to add vertices one by one.
    if best_size < n:
        # Find a good starting pair
        # Try all pairs, keep the best feasible one
        best_start = None
        for u, v in combinations(range(n), 2):
            if time.time() > deadline:
                break
            feasible, _ = max_flow_feasible([u, v])
            if feasible:
                if best_start is None:
                    best_start = [u, v]
                break  # Any pair with an edge works; just take the first

        if best_start is not None:
            current = list(best_start)
            # Try to extend
            for _ in range(n - len(current)):
                if time.time() > deadline:
                    break
                extended = False
                # Try adding each non-branch vertex
                for candidate in range(n):
                    if candidate in current:
                        continue
                    new_set = current + [candidate]
                    feasible, _ = max_flow_feasible(new_set)
                    if feasible:
                        current = new_set
                        extended = True
                        # Try to get the actual paths
                        result = try_clique_subdivision(current)
                        if result is not None and len(current) > best_size:
                            best_result = result
                            best_size = len(current)
                        break
                if not extended:
                    break

    # Strategy 3: If still small, try random restarts of greedy extension
    if best_size < n and time.time() < deadline - 5:
        import random
        rng = random.Random(42)
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            # Pick a random starting edge
            u, v = edges[rng.randrange(len(edges))]
            current = [u, v]
            for _ in range(n - len(current)):
                if time.time() > deadline - 5:
                    break
                candidates = [x for x in range(n) if x not in current]
                rng.shuffle(candidates)
                extended = False
                for candidate in candidates:
                    new_set = current + [candidate]
                    feasible, _ = max_flow_feasible(new_set)
                    if feasible:
                        current = new_set
                        extended = True
                        result = try_clique_subdivision(current)
                        if result is not None and len(current) > best_size:
                            best_result = result
                            best_size = len(current)
                        break
                if not extended:
                    break

    # Fallback
    if best_result is None:
        u, v = edges[0]
        best_result = {'branches': [u, v], 'paths': [[u, v]]}

    # Ensure types are correct
    best_result['branches'] = list(best_result['branches'])
    best_result['paths'] = [list(p) for p in best_result['paths']]

    return best_result
# EVOLVE-BLOCK-END

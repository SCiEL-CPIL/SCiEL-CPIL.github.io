import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (90s kill)

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # ---------- Max-flow oracle for vertex-disjoint paths ----------
    # For a candidate branch set B, we check if there exist pairwise
    # vertex-disjoint paths (disjoint in interior vertices, avoiding all
    # branches except endpoints) connecting every pair in B.
    #
    # We model this as a flow problem on a split graph:
    #   Each vertex v becomes v_in -> v_out with capacity 1.
    #   For each edge (u,v) in G, add u_out -> v_in and v_out -> u_in with cap 1.
    #   For each branch b: b_in has infinite supply, b_out has infinite demand.
    #   For a pair (b1,b2): we need 1 unit of flow from b1 to b2.
    #
    # Since we need paths for ALL pairs simultaneously, we use a super-source
    # connected to each branch's in-node and a super-sink from each branch's
    # out-node. But that doesn't enforce pair-specific routing.
    #
    # Better approach: For each pair (u,v), we need a path. The paths must be
    # vertex-disjoint in interiors. We can model this as: create a flow network
    # where each pair demands 1 unit, and interior vertices have capacity 1.
    #
    # Actually, the cleanest model: 
    #   - Split every vertex v into v_in and v_out with edge v_in->v_out cap 1.
    #   - For graph edge (u,v): u_out -> v_in (cap 1) and v_out -> u_in (cap 1).
    #   - For branch b: bypass the capacity on b by making b_in->b_out cap INF.
    #   - Super source S connects to each branch b_in with cap = (k-1) where k=|B|.
    #   - Each branch b_out connects to super sink T with cap = (k-1).
    #   - But this allows any branch to route to any branch, not specific pairs.
    #
    # The correct model for "all pairs have disjoint interior paths":
    #   This is equivalent to: the subgraph induced by B plus interior vertices
    #   contains a subdivision of K_|B|. 
    #
    # A practical check: use max flow where each pair (b_i, b_j) requires
    # one unit of flow, and all non-branch vertices have capacity 1.
    # We can do this by: for each pair, add a demand. But multi-commodity flow
    # is NP-hard in general.
    #
    # However, for our purposes, we can use a simpler sufficient check:
    #   Build the split graph. For each branch b, give b_in->b_out capacity INF.
    #   Add super source S -> b_in (cap = |B|-1) for each branch b.
    #   Add b_out -> T (cap = |B|-1) for each branch b.
    #   Compute max flow. If max flow >= |B|*(|B|-1)/2, then we have enough
    #   "throughput" but not necessarily correct pairing.
    #
    # A better practical approach: since the graph is small (n=24), we can
    # use a direct approach: for each pair, find a shortest path avoiding
    # used interiors, but use backtracking/flow to ensure global feasibility.
    #
    # Let me use a different structural approach: 
    #   Model as a single-commodity flow problem per "layer" won't work.
    #
    # Practical approach for small graphs: 
    #   Use the fact that we need |B| choose 2 paths. 
    #   Create a flow network where:
    #     - Each non-branch vertex v has v_in -> v_out with cap 1.
    #     - Each branch b has b_in -> b_out with cap INF (unlimited).
    #     - Graph edges become directed arcs between out and in nodes.
    #     - For each pair (b1, b2), we need 1 unit from b1 to b2.
    #   This is a multi-commodity flow. For small instances, we can approximate
    #   by checking if a single max flow from a super-source (connected to all
    #   branch in-nodes) to super-sink (from all branch out-nodes) achieves
    #   |B|*(|B|-1)/2 units. This is a necessary condition and often sufficient
    #   for dense graphs.
    #
    # Let me implement the single-commodity flow check as a feasibility test,
    # then verify by actually constructing paths.

    def flow_feasible(branches):
        """Check if vertex-disjoint interior paths exist for all pairs in branches.
        Returns True if feasible (necessary condition via max flow)."""
        if len(branches) < 2:
            return True
        b_set = set(branches)
        k = len(branches)
        required_flow = k * (k - 1) // 2

        # Build split graph for max flow
        # Node indexing: v_in = 2*v, v_out = 2*v+1
        # Super source = 2*n, super sink = 2*n+1
        S = 2 * n
        T = 2 * n + 1
        total_nodes = 2 * n + 2

        # Adjacency list for flow network: (to, cap)
        flow_adj = [[] for _ in range(total_nodes)]

        def add_flow_edge(u, v, cap):
            flow_adj[u].append([v, cap, len(flow_adj[v])])
            flow_adj[v].append([u, 0, len(flow_adj[u]) - 1])

        # Vertex split edges
        for v in range(n):
            if v in b_set:
                add_flow_edge(2 * v, 2 * v + 1, 10**9)  # branches: unlimited
            else:
                add_flow_edge(2 * v, 2 * v + 1, 1)  # interior: capacity 1

        # Graph edges
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    add_flow_edge(2 * u + 1, 2 * v, 10**9)
                    add_flow_edge(2 * v + 1, 2 * u, 10**9)

        # Super source/sink connections
        for b in branches:
            add_flow_edge(S, 2 * b, 10**9)
            add_flow_edge(2 * b + 1, T, 10**9)

        # Edmonds-Karp (BFS-based max flow) - good for small graphs
        def bfs_max_flow():
            total_flow = 0
            while True:
                # BFS to find augmenting path
                parent = [-1] * total_nodes
                parent_edge = [None] * total_nodes
                queue = deque([S])
                parent[S] = S
                found = False
                while queue and not found:
                    cur = queue.popleft()
                    for i, (to, cap, rev) in enumerate(flow_adj[cur]):
                        if cap > 0 and parent[to] == -1:
                            parent[to] = cur
                            parent_edge[to] = (i, rev)
                            if to == T:
                                found = True
                                break
                            queue.append(to)
                if not found:
                    break
                # Find bottleneck
                bottleneck = 10**18
                node = T
                while node != S:
                    i, rev = parent_edge[node]
                    bottleneck = min(bottleneck, flow_adj[parent[node]][i][1])
                    node = parent[node]
                # Augment
                node = T
                while node != S:
                    i, rev = parent_edge[node]
                    flow_adj[parent[node]][i][1] -= bottleneck
                    flow_adj[node][rev][1] += bottleneck
                    node = parent[node]
                total_flow += bottleneck
            return total_flow

        maxflow = bfs_max_flow()
        return maxflow >= required_flow

    def find_paths(branches):
        """Given a feasible branch set, find actual paths using BFS with
        careful ordering. Returns dict of (u,v) -> path, or None if fails."""
        b_set = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        paths = {}
        used_interior = set()

        # Sort pairs: direct edges first, then by shortest path length
        def path_length_est(u, v):
            if v in adj[u]:
                return 0
            return 10  # placeholder; will refine
        pairs.sort(key=lambda uv: (0 if uv[1] in adj[uv[0]] else 1))

        for u, v in pairs:
            if v in adj[u]:
                paths[(u, v)] = [u, v]
                continue
            # BFS avoiding branches (except endpoints) and used interiors
            forbidden = b_set - {u, v}
            forbidden |= used_interior
            visited = {u}
            parent = {u: None}
            queue = deque([u])
            found = False
            while queue:
                cur = queue.popleft()
                if cur == v:
                    path = []
                    node = v
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    paths[(u, v)] = path
                    for mid in path[1:-1]:
                        used_interior.add(mid)
                    found = True
                    break
                for nb in sorted(adj[cur]):
                    if nb not in visited and nb not in forbidden:
                        visited.add(nb)
                        parent[nb] = cur
                        queue.append(nb)
            if not found:
                return None
        return paths

    # ---------- Find maximum clique (Bron-Kerbosch with pivot) ----------
    best_clique = []

    def bronk(R, P_set, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P_set and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        candidates = list(P_set | X)
        if not candidates:
            return
        pivot = max(candidates, key=lambda u: len(adj[u] & P_set))
        for v in list(P_set - adj[pivot]):
            bronk(R | {v}, P_set & adj[v], X & adj[v])
            P_set = P_set - {v}
            X = X | {v}
            if time.time() > deadline:
                return

    bronk(set(), set(range(n)), set())

    # ---------- Local search over branch sets ----------
    # Start from max clique, try to add/swap vertices using flow oracle.
    current_branches = list(best_clique)
    current_paths = find_paths(current_branches)
    if current_paths is None:
        # Fallback: use clique with direct edges only
        current_paths = {}
        for u, v in combinations(sorted(current_branches), 2):
            current_paths[(u, v)] = [u, v]

    best_size = len(current_branches)
    best_branches = list(current_branches)

    # Try to add vertices
    remaining = [v for v in range(n) if v not in set(current_branches)]
    improved = True
    while improved:
        improved = False
        for v in remaining[:]:
            if time.time() > deadline:
                break
            test_branches = current_branches + [v]
            if flow_feasible(test_branches):
                test_paths = find_paths(test_branches)
                if test_paths is not None:
                    current_branches = test_branches
                    current_paths = test_paths
                    remaining.remove(v)
                    improved = True
                    best_size = len(current_branches)
                    best_branches = list(current_branches)

    # Try swap moves: replace one branch with another to potentially allow more
    if time.time() < deadline - 10:
        for _ in range(3):  # a few rounds of swaps
            swap_improved = False
            for i in range(len(current_branches)):
                if time.time() > deadline:
                    break
                for v in remaining:
                    if time.time() > deadline:
                        break
                    test_branches = current_branches[:i] + current_branches[i+1:] + [v]
                    if flow_feasible(test_branches):
                        test_paths = find_paths(test_branches)
                        if test_paths is not None:
                            current_branches = test_branches
                            current_paths = test_paths
                            remaining.remove(v)
                            remaining.append(current_branches[i])
                            swap_improved = True
                            best_size = len(current_branches)
                            best_branches = list(current_branches)
                            break
                if swap_improved:
                    break
            if not swap_improved:
                break

    # Use the best found
    if len(best_branches) > len(current_branches):
        current_branches = best_branches
        current_paths = find_paths(current_branches)

    # Build result
    if current_paths is None:
        # Fallback: just direct edges for the clique
        current_branches = list(best_clique)
        current_paths = {}
        for u, v in combinations(sorted(current_branches), 2):
            current_paths[(u, v)] = [u, v]

    branches = [int(b) for b in sorted(current_branches)]
    paths_list = []
    for u, v in combinations(sorted(current_branches), 2):
        p = current_paths.get((u, v), [u, v])
        paths_list.append([int(x) for x in p])

    return {'branches': branches, 'paths': paths_list}

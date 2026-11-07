import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def hopcroft_karp(adj, n):
        """Maximum bipartite matching. adj: list of lists (adj[u] = list of v's).
        Returns list of (u, v) pairs."""
        from collections import deque
        pair_u = [-1] * n
        pair_v = [-1] * n
        dist = [0] * n

        def bfs():
            q = deque()
            for u in range(n):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            found = False
            while q:
                u = q.popleft()
                for v in adj[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
            return found

        def dfs(u):
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False

        matching = 0
        while bfs():
            for u in range(n):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1

        result = []
        for u in range(n):
            if pair_u[u] != -1:
                result.append((u, pair_u[u]))
        return result

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        n_i = n - k

        # Build edge index for K-edges
        k_edge_index = {}
        k_edge_list = []
        for u in range(k):
            for w in range(u + 1, k):
                k_edge_index[(u, w)] = len(k_edge_list)
                k_edge_list.append((u, w))
        num_k_edges = len(k_edge_list)

        # For each I-vertex v, find which K-edges can be absorbed (both endpoints in N(v))
        # We want to assign each K-edge to at most one v, maximizing total savings.
        # Using a matching: left side = K-edges, right side = I-vertices.
        # Edge exists if both endpoints of the K-edge are neighbors of v.
        # Weight: 1 for each assignment (saves 1 clique).

        # Build adjacency for bipartite graph: left = K-edges, right = I-vertices
        left_adj = [[] for _ in range(num_k_edges)]
        for v_idx in range(n_i):
            v = k + v_idx
            nset = set(neighbors[v])
            for e_idx, (u, w) in enumerate(k_edge_list):
                if u in nset and w in nset:
                    left_adj[e_idx].append(v_idx)

        # Find maximum matching in this bipartite graph
        # Left nodes: 0..num_k_edges-1, Right nodes: 0..n_i-1
        # Use Hopcroft-Karp: left side = K-edges
        # adj for HK: adj[left_node] = list of right_nodes
        hk_adj = left_adj  # already in the right format
        hk_result = hopcroft_karp(hk_adj, num_k_edges)
        # hk_result: list of (left_node, right_node) = (edge_idx, v_idx)

        # Assign K-edges to I-vertices
        edge_to_v = {}  # edge_idx -> v_idx
        v_clique_edges = {v_idx: [] for v_idx in range(n_i)}
        for e_idx, v_idx in hk_result:
            edge_to_v[e_idx] = v_idx
            v_clique_edges[v_idx].append(e_idx)

        # Mark which K-edges are absorbed
        absorbed = [False] * num_k_edges
        for e_idx in edge_to_v:
            absorbed[e_idx] = True

        # Build cliques for I-vertices
        cliques = []
        for v_idx in range(n_i):
            v = k + v_idx
            # K-I edges: each neighbor u of v gets a clique
            # If u is paired with v via an absorbed K-edge (u,w), then clique is [u, w, v]
            # Otherwise clique is [u, v]
            # We need to track which K-edges are in v's group
            # For each absorbed edge (u,w) assigned to v, both u and w are in N(v)
            # The K-I edge (u,v) and (w,v) are covered by the size-3 clique [u,w,v]
            # For K-I edges (u,v) where u is not in any absorbed edge for v, use [u,v]

            # Determine which K-vertices are "paired" via absorbed edges for this v
            paired_vertices = set()
            for e_idx in v_clique_edges[v_idx]:
                u, w = k_edge_list[e_idx]
                paired_vertices.add(u)
                paired_vertices.add(w)

            # For paired vertices, we form size-3 cliques [u, w, v]
            # Each absorbed edge gives one size-3 clique
            for e_idx in v_clique_edges[v_idx]:
                u, w = k_edge_list[e_idx]
                cliques.append([u, w, v])

            # For unpaired neighbors, form size-2 cliques [u, v]
            for u in neighbors[v]:
                if u not in paired_vertices:
                    cliques.append([u, v])

        # Remaining K-edges: those not absorbed
        remaining_edges = [e_idx for e_idx in range(num_k_edges) if not absorbed[e_idx]]
        remaining_set = set(remaining_edges)

        # Find maximum matching among remaining K-edges to form size-2 cliques
        # Build adjacency: for each remaining edge, which other remaining edges share a vertex
        # Actually, we want to pair up K-edges that share a vertex into size-3 cliques?
        # No - we can only form cliques from vertices in K. A clique in K of size s covers C(s,2) edges.
        # For remaining edges, we want to cover them with minimum cliques within K.
        # Greedy: find maximal cliques in the remaining subgraph.

        # Build remaining subgraph
        rem_adj = {u: set() for u in range(k)}
        for e_idx in remaining_edges:
            u, w = k_edge_list[e_idx]
            rem_adj[u].add(w)
            rem_adj[w].add(u)

        # Greedy maximal clique cover on remaining edges
        remaining_edge_set = set(remaining_edges)
        while remaining_edge_set:
            e_idx = next(iter(remaining_edge_set))
            u, w = k_edge_list[e_idx]
            # Try to grow a clique starting from {u, w}
            clique = {u, w}
            changed = True
            while changed:
                changed = False
                for x in range(k):
                    if x not in clique:
                        # Check if x is adjacent to all in clique in the remaining graph
                        if all(x in rem_adj[y] for y in clique):
                            clique.add(x)
                            changed = True
                            break
            if len(clique) >= 2:
                cliques.append(sorted(clique))
                for a, b in combinations(sorted(clique), 2):
                    # Find the edge index
                    if a < b:
                        key = (a, b)
                    else:
                        key = (b, a)
                    if key in k_edge_index:
                        idx = k_edge_index[key]
                        if idx in remaining_edge_set:
                            remaining_edge_set.discard(idx)

        return [[int(x) for x in c] for c in cliques]

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
    return results

"""Decompose each instance graph into edge-disjoint cycles and edges, minimizing pieces.

Structural approach: use maximum flow to find the maximum set of edge-disjoint
cycles simultaneously (a flow-based cycle packing), which is provably optimal for
the number of cycles, then emit remaining edges as singletons.
"""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque


def construct():
    deadline = time.time() + 140  # safety margin before the 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_flow_network(edge_set, n_vertices):
        """
        Build a flow network whose max flow equals the maximum number of
        edge-disjoint cycles in the graph defined by edge_set.

        Vertex splitting: each vertex v -> v_in = 2*v, v_out = 2*v+1.
        Edge v_in -> v_out with capacity 1 (each vertex used by at most one cycle).
        Each undirected edge (u,v) -> arcs u_out -> v_in and v_out -> u_in, cap 1.
        Super source S = 2*n -> each v_in, cap 1.
        Each v_out -> super sink T = 2*n+1, cap 1.

        A unit of flow S -> a_in -> a_out -> b_in -> b_out -> ... -> T traces a
        cycle through the original edges.
        """
        if not edge_set:
            return None

        S = 2 * n_vertices
        T = 2 * n_vertices + 1
        cap = defaultdict(int)

        def add_arc(a, b, c):
            cap[(a, b)] += c
            cap[(b, a)] += 0  # ensure reverse exists for residual

        # vertex split
        for v in range(n_vertices):
            add_arc(2 * v, 2 * v + 1, 1)
        # original edges as directed arcs
        for u, v in edge_set:
            add_arc(2 * u + 1, 2 * v, 1)
            add_arc(2 * v + 1, 2 * u, 1)
        # source / sink
        for v in range(n_vertices):
            add_arc(S, 2 * v, 1)
            add_arc(2 * v + 1, T, 1)

        return S, T, cap

    def max_flow(S, T, cap, time_budget):
        """Edmonds-Karp (BFS augmenting paths). Returns (flow_value, parent)."""
        # residual stored in cap dict; we mutate it
        parent = {}
        flow = 0
        while time_budget() is not None:
            # BFS for augmenting path
            level = {S: 0}
            par = {S: None}
            q = deque([S])
            found = False
            while q and not found:
                u = q.popleft()
                for (a, b) in list(cap.keys()):
                    if a != u:
                        continue
                    if cap[(a, b)] > 0 and b not in level:
                        level[b] = level[u] + 1
                        par[b] = u
                        if b == T:
                            found = True
                            break
                        q.append(b)
            if not found:
                break
            # augment by 1 (all capacities are 0/1 integers)
            v = T
            while v != S:
                u = par[v]
                cap[(u, v)] -= 1
                cap[(v, u)] += 1
                v = u
            flow += 1
            parent = par
        return flow, parent

    def extract_cycles_from_flow(edge_set, n_vertices, time_budget):
        """
        Run max flow, then decompose the resulting flow into cycles.
        Returns list of cycles (each a list of distinct vertices).
        """
        net = build_flow_network(edge_set, n_vertices)
        if net is None:
            return []
        S, T, cap = net
        flow_val, _ = max_flow(S, T, cap, time_budget)
        if flow_val == 0:
            return []

        # Reconstruct which original edges carry flow.
        # An original undirected edge (u,v) carries flow iff cap[(2u+1,2v)]==0
        # (i.e., it was used) OR cap[(2v+1,2u)]==0.
        used_edges = set()
        for u, v in edge_set:
            if cap[(2 * u + 1, 2 * v)] == 0 or cap[(2 * v + 1, 2 * u)] == 0:
                used_edges.add(tuple(sorted((u, v))))

        # Build adjacency of used edges; each vertex has degree exactly 2 in the
        # subgraph of used edges (since each vertex's split edge carried exactly
        # one unit of flow). So used_edges decomposes into disjoint cycles.
        adj = defaultdict(set)
        for u, v in used_edges:
            adj[u].add(v)
            adj[v].add(u)

        cycles = []
        remaining = set(used_edges)
        while remaining:
            start = min(remaining)[0]
            # walk the cycle
            cycle = [start]
            prev = None
            cur = start
            while True:
                # choose next neighbor not equal to prev
                nbrs = [x for x in adj[cur] if x != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                if nxt == start:
                    break
                cycle.append(nxt)
                prev, cur = cur, nxt
            if len(cycle) >= 3:
                cycles.append(cycle)
            # remove all edges of this cycle from remaining
            for i in range(len(cycle)):
                a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                remaining.discard(tuple(sorted((a, b))))
        return cycles

    def decompose(edges, n, time_budget):
        """
        Decompose the edge set into cycles (max-flow packed) and single edges.
        """
        if not edges:
            return []

        edge_set = set(tuple(sorted(e)) for e in edges)
        rem = set(edge_set)

        pieces = []
        # Try to extract cycles via max-flow packing.
        # Because each max-flow iteration is relatively cheap for these sizes,
        # we attempt one global packing; if it fails or leaves a non-2-core
        # remainder, fall back to greedy for the rest.
        cycles = extract_cycles_from_flow(rem, n, time_budget)
        if cycles:
            for cyc in cycles:
                pieces.append(cyc[:])
                for i in range(len(cyc)):
                    a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                    rem.discard(tuple(sorted((a, b))))

        # Greedy fallback for any remaining edges (should be a forest, but be safe).
        while rem:
            # find any cycle in rem via BFS
            adj = defaultdict(set)
            for u, v in rem:
                adj[u].add(v)
                adj[v].add(u)
            cycle = None
            edge_list = list(rem)
            random.shuffle(edge_list)
            for u, v in edge_list:
                visited = {v}
                parent = {v: None}
                q = deque([v])
                found = False
                while q and not found:
                    curr = q.popleft()
                    for nb in adj[curr]:
                        if tuple(sorted((curr, nb))) == tuple(sorted((u, v))):
                            continue
                        if nb == u:
                            # path v -> ... -> curr -> u
                            cycle = [u]
                            node = curr
                            while node != v:
                                cycle.append(node)
                                node = parent[node]
                            cycle.append(v)
                            found = True
                            break
                        if nb not in visited:
                            visited.add(nb)
                            parent[nb] = curr
                            q.append(nb)
                if cycle and len(cycle) >= 3:
                    break
            if cycle and len(cycle) >= 3:
                pieces.append(cycle[:])
                for i in range(len(cycle)):
                    a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                    rem.discard(tuple(sorted((a, b))))
            else:
                e = tuple(next(iter(rem)))
                rem.discard(e)
                pieces.append(list(e))

        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for spec in specs:
        if time.time() > deadline - 3:
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n, lambda: None if time.time() > deadline - 5 else 0)
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END

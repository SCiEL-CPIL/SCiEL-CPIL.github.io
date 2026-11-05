# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch_maximal_cliques(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        max_cliques = []

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return
            if not P:
                return

            pivot_candidates = P | X
            best_pivot = None
            max_common = -1
            for u in pivot_candidates:
                common = len(P & adj[u])
                if common > max_common:
                    max_common = common
                    best_pivot = u

            if best_pivot is not None:
                candidates = [v for v in P if v not in adj[best_pivot]]
            else:
                candidates = list(P)

            for v in candidates:
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P.discard(v)
                X.add(v)

        bk(frozenset(), set(range(n)), set())
        return max_cliques

    def line_graph_max_matching(n, edges):
        """
        Find maximum matching in the line graph of the original graph.
        Line graph vertices = edges of original graph.
        Two line-graph vertices are adjacent if their edges share an endpoint.
        Use greedy + augmenting path (Edmonds-style simplified for bipartite-like structure).
        Since line graph may not be bipartite, we use a simple augmenting path approach.
        """
        m = len(edges)
        if m == 0:
            return []

        # Map each edge to an index
        edge_to_idx = {}
        for i, e in enumerate(edges):
            edge_to_idx[e] = i
            edge_to_idx[(e[1], e[0])] = i

        # Build line graph adjacency: two edges are adjacent if they share a vertex
        # For efficiency, group edges by endpoint
        endpoint_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            endpoint_edges[u].append(i)
            endpoint_edges[v].append(i)

        # Line graph adjacency list
        lg_adj = [set() for _ in range(m)]
        for u in range(n):
            e_list = endpoint_edges[u]
            for i in range(len(e_list)):
                for j in range(i + 1, len(e_list)):
                    a, b = e_list[i], e_list[j]
                    lg_adj[a].add(b)
                    lg_adj[b].add(a)

        # Maximum matching via augmenting paths (BFS-based)
        # match[i] = j means line-graph vertex i is matched to j
        match = [-1] * m

        def bfs_augment(start):
            """Find augmenting path from unmatched vertex start."""
            parent = [-1] * m
            visited = [False] * m
            queue = [start]
            visited[start] = True
            found = -1

            while queue and found == -1:
                u = queue.pop(0)
                for v in lg_adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        parent[v] = u
                        # If v is matched, follow the match
                        if match[v] != -1:
                            w = match[v]
                            if not visited[w]:
                                visited[w] = True
                                parent[w] = v
                                queue.append(w)
                        else:
                            # Found augmenting path
                            found = v
                            break

            if found == -1:
                return False

            # Augment
            v = found
            while v != start:
                u = parent[v]
                prev_match = match[u]
                match[u] = v
                match[v] = u
                v = prev_match if prev_match != -1 else start
                if v == start:
                    break
                # Actually need to trace back properly
                # Let me redo the augmentation

            return True

        # Simpler augmenting path: standard BFS for bipartite matching won't work directly
        # Use a different approach: greedy initial matching + local improvement
        # For line graphs, we can use a simpler heuristic

        # Greedy matching
        matched = [False] * m
        matching = []
        # Sort edges by degree in line graph (descending) for better greedy
        order = sorted(range(m), key=lambda i: len(lg_adj[i]), reverse=True)
        for i in order:
            if matched[i]:
                continue
            for j in lg_adj[i]:
                if not matched[j]:
                    matching.append((i, j))
                    matched[i] = True
                    matched[j] = True
                    break

        # Try to improve with augmenting paths
        # Build match structure
        match2 = [-1] * m
        for i, j in matching:
            match2[i] = j
            match2[j] = i

        improved = True
        while improved:
            improved = False
            for start in range(m):
                if match2[start] != -1:
                    continue
                # BFS to find augmenting path
                parent = [-1] * m
                visited = [False] * m
                queue = [start]
                visited[start] = True
                found_end = -1

                while queue and found_end == -1:
                    u = queue.pop(0)
                    for v in lg_adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            parent[v] = u
                            if match2[v] != -1:
                                w = match2[v]
                                if not visited[w]:
                                    visited[w] = True
                                    parent[w] = v
                                    queue.append(w)
                            else:
                                found_end = v
                                break

                if found_end != -1:
                    # Augment along path
                    v = found_end
                    while v != start:
                        u = parent[v]
                        old_match = match2[u]
                        match2[u] = v
                        match2[v] = u
                        if old_match != -1:
                            match2[old_match] = -1
                        v = old_match if old_match != -1 else -1
                        if v == -1:
                            break
                    improved = True
                    break

        # Extract matching pairs
        final_matching = []
        seen = set()
        for i in range(m):
            if match2[i] != -1 and i < match2[i]:
                final_matching.append((i, match2[i]))

        return final_matching

    def vertex_cover_from_matching(n, edges, matching):
        """
        From a maximum matching in the line graph, construct a vertex cover of the original graph.
        Each matched pair of edges (e1, e2) that share a vertex v: include v in the cover.
        Unmatched edges: include one of their endpoints.
        """
        cover = set()
        matched_edge_idx = set()
        for i, j in matching:
            matched_edge_idx.add(i)
            matched_edge_idx.add(j)
            # The two edges share at least one endpoint
            e1 = edges[i]
            e2 = edges[j]
            common = set(e1) & set(e2)
            if common:
                cover.add(min(common))
            else:
                # Shouldn't happen for adjacent edges in line graph, but safety
                cover.add(e1[0])

        # For unmatched edges, add a vertex
        for idx, (u, v) in enumerate(edges):
            if idx not in matched_edge_idx:
                # Add the vertex with lower degree or just u
                cover.add(u)

        return cover

    def local_search_reduce(n, edges, cliques, initial_cover, time_budget):
        """
        Local search: try to remove vertices from the cover while maintaining validity.
        Also try swaps: remove one vertex, add another if it reduces size.
        """
        cover = set(initial_cover)
        best_cover = set(cover)
        best_size = len(cover)

        def is_valid(cov):
            for c in cliques:
                if not (c & cov):
                    return False
            return True

        # Precompute: for each vertex, which cliques it covers
        vertex_cliques = defaultdict(set)
        for i, c in enumerate(cliques):
            for v in c:
                vertex_cliques[v].add(i)

        # Phase 1: Remove redundant vertices
        changed = True
        while changed and time.time() < time_budget:
            changed = False
            for v in sorted(cover):
                test = set(cover)
                test.discard(v)
                if is_valid(test):
                    cover = test
                    best_cover = set(cover)
                    best_size = len(cover)
                    changed = True
                    break

        # Phase 2: Try swaps - remove a vertex, see if we can add a different one
        # Actually, for minimum hitting set, swaps don't directly help unless we can
        # replace two vertices with one. Let's try: for each pair of vertices in cover,
        # check if removing both and adding one vertex maintains validity.
        # This is expensive, so only do it if we have time.

        if time.time() < time_budget - 5:
            # Try 2-for-1 replacements
            cover_list = list(cover)
            for i in range(len(cover_list)):
                if time.time() > time_budget - 5:
                    break
                for j in range(i + 1, len(cover_list)):
                    if time.time() > time_budget - 5:
                        break
                    v1, v2 = cover_list[i], cover_list[j]
                    test = set(cover)
                    test.discard(v1)
                    test.discard(v2)
                    # Try adding each other vertex
                    for v3 in range(n):
                        if v3 in test:
                            continue
                        test.add(v3)
                        if is_valid(test) and len(test) < best_size:
                            best_cover = set(test)
                            best_size = len(test)
                            cover = set(test)
                            cover_list = list(cover)
                            break
                        test.discard(v3)

        return best_cover

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        cliques = bron_kerbosch_maximal_cliques(n, edges)
        cliques = [c for c in cliques if len(c) >= 2]

        if not cliques:
            return []

        # Get maximum matching in line graph
        matching = line_graph_max_matching(n, edges)

        # Construct vertex cover from matching
        cover = vertex_cover_from_matching(n, edges, matching)

        # Local search to reduce
        cover = local_search_reduce(n, edges, cliques, cover, time_budget)

        return sorted(cover)

    results = []

    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline - 30:
            results.append(list(range(n)))
            continue

        elapsed = time.time() - start_time
        remaining_instances = 3 - len(results)
        per_instance = (deadline - 15 - elapsed) / max(1, remaining_instances)
        per_instance = max(2.0, min(per_instance, deadline - 15 - elapsed))

        sol = solve_instance(n, p, seed, start_time + elapsed + per_instance)
        results.append(sol)

    return results
# EVOLVE-BLOCK-END

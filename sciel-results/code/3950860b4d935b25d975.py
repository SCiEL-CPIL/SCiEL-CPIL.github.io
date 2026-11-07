import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def bron_kerbosch_max_clique(adj, vertices):
        """
        Find a maximum clique in the graph defined by adj (list of sets) on vertices.
        Uses Bron-Kerbosch with pivot for efficiency.
        Returns the largest clique found as a list.
        """
        if not vertices:
            return []

        best = []
        vertices_set = set(vertices)

        def bk(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return

            pivot_candidates = list(P | X)
            if pivot_candidates:
                pivot = max(pivot_candidates, key=lambda u: len(P & adj[u]))
                candidates = [v for v in P if v not in adj[pivot]]
            else:
                candidates = list(P)

            for v in candidates:
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                P.remove(v)
                X.add(v)
                bk(new_R, new_P, new_X)
                P.add(v)
                X.remove(v)

        bk(set(), vertices_set, set())
        return best

    def dsatur_coloring(adj, vertices):
        """
        Greedy DSATUR coloring.
        adj: list of sets, where adj[u] contains neighbors of u.
        vertices: list of vertex indices.
        Returns a list of color classes (lists of vertices).
        """
        if not vertices:
            return []
        
        v_set = set(vertices)
        color = {v: -1 for v in v_set}
        colors_used = 0
        color_classes = {} # color -> list of vertices
        
        # Order vertices by degree descending initially
        uncolored = sorted(v_set, key=lambda v: -len(adj[v] & v_set))
        
        while uncolored:
            # Pick vertex with highest saturation degree (number of different colors in neighbors)
            # Tie-break by degree in uncolored graph
            def sat_degree(v):
                neighbor_colors = set()
                for u in adj[v]:
                    if u in v_set and color[u] != -1:
                        neighbor_colors.add(color[u])
                return len(neighbor_colors)
            
            v = max(uncolored, key=lambda x: (sat_degree(x), len(adj[x] & set(uncolored))))
            uncolored.remove(v)
            
            # Find smallest available color
            used_colors = set()
            for u in adj[v]:
                if u in v_set and color[u] != -1:
                    used_colors.add(color[u])
            
            c = 0
            while c in used_colors:
                c += 1
            
            color[v] = c
            if c not in color_classes:
                color_classes[c] = []
                colors_used += 1
            color_classes[c].append(v)
            
        return [color_classes[c] for c in range(colors_used)]

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # Build adjacency for K
        adj_k = [set() for _ in range(k)]
        neighbors_in_k = {v: set() for v in range(k, n)}

        for u, v in edges:
            if u < k and v < k:
                adj_k[u].add(v)
                adj_k[v].add(u)
            elif u >= k and v < k:
                neighbors_in_k[u].add(v)
            elif v >= k and u < k:
                neighbors_in_k[v].add(u)

        used_edges = set()
        cliques = []

        def norm_edge(u, v):
            return (min(u, v), max(u, v))

        # Process I vertices
        # Sort by degree in K descending to allow larger cliques to form first
        i_vertices = sorted(range(k, n), key=lambda v: -len(neighbors_in_k[v]))

        for v in i_vertices:
            if time.time() > deadline:
                # Fallback: cover remaining cross edges with pairs
                for u in neighbors_in_k[v]:
                    if norm_edge(u, v) not in used_edges:
                        cliques.append([v, u])
                        used_edges.add(norm_edge(u, v))
                continue

            N_v = set(neighbors_in_k[v])
            if not N_v:
                continue

            # Build graph H on N_v where edges exist if (u, w) is in E(K) and not used
            # Vertices are the actual vertex IDs in N_v
            # We need to map them to 0..m-1 for the BK algorithm
            rem_list = list(N_v)
            m = len(rem_list)
            if m == 0:
                continue
            idx_map = {v_idx: i for i, v_idx in enumerate(rem_list)}
            
            adj_avail = [set() for _ in range(m)]
            for i in range(m):
                for j in range(i + 1, m):
                    u, w = rem_list[i], rem_list[j]
                    e = norm_edge(u, w)
                    if e not in used_edges:
                        adj_avail[i].add(j)
                        adj_avail[j].add(i)

            best_local = bron_kerbosch_max_clique(adj_avail, list(range(m)))
            
            if best_local:
                S = [rem_list[idx] for idx in best_local]
                cliques.append([v] + S)
                # Mark edges
                for u in S:
                    used_edges.add(norm_edge(v, u))
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        used_edges.add(norm_edge(S[i], S[j]))
            else:
                # No clique of size >= 2 in H? Or H is empty?
                # If H has no edges, we must use pairs.
                # But BK returns max clique. If max clique size is 1, best_local will have 1 element?
                # BK returns [] if no vertices? No, if P is not empty, it finds cliques.
                # If adj_avail is empty graph, max clique is 1.
                # So best_local should have at least 1 element if rem_list is not empty.
                if not best_local and rem_list:
                    # This case shouldn't happen if rem_list is not empty
                    u = rem_list[0]
                    cliques.append([v, u])
                    used_edges.add(norm_edge(v, u))

        # Cover remaining K-edges
        E_rem = set()
        for u in range(k):
            for w in range(u + 1, k):
                if norm_edge(u, w) not in used_edges:
                    E_rem.add(norm_edge(u, w))

        if E_rem:
            if time.time() > deadline:
                # Fallback: cover with pairs
                for u, w in E_rem:
                    cliques.append([u, w])
                return cliques

            # Find active vertices
            active_vertices = set()
            for u, w in E_rem:
                active_vertices.add(u)
                active_vertices.add(w)
            
            v_list = list(active_vertices)
            m = len(v_list)
            if m == 0:
                return cliques
            idx_map = {v: i for i, v in enumerate(v_list)}

            # Build complement graph of G_rem
            # G_rem has edges E_rem.
            # Complement has edges (u, w) if u, w in active_vertices and (u, w) NOT in E_rem.
            adj_comp = [set() for _ in range(m)]
            for i in range(m):
                for j in range(i + 1, m):
                    u, w = v_list[i], v_list[j]
                    e = norm_edge(u, w)
                    if e not in E_rem:
                        adj_comp[i].add(j)
                        adj_comp[j].add(i)

            # Color the complement graph
            color_classes = dsatur_coloring(adj_comp, list(range(m)))
            
            for cc in color_classes:
                if len(cc) >= 2:
                    S = [v_list[idx] for idx in cc]
                    cliques.append(S)
                # If len(cc) == 1, it's an isolated vertex in G_rem, so no edges to cover.
                # We don't need to add it as a clique.

        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))

    for spec in specs:
        res = solve_one(*spec)
        results.append(res)

    return results

# EVOLVE-BLOCK-END

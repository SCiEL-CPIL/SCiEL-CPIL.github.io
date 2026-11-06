import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve(edges, time_budget=70):
        if not edges:
            return []

        n = max(max(e) for e in edges) + 1
        edge_set = set()
        for u, v in edges:
            edge_set.add((u, v) if u < v else (v, u))

        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        # Find all maximal cliques using Bron-Kerbosch with pivot
        all_maximal_cliques = []

        def bronk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    all_maximal_cliques.append(frozenset(R))
                return
            # Choose pivot
            if P:
                # Pick vertex with max degree in P
                pivot = max(P, key=lambda v: len(adj[v] & P))
                candidates = P - adj[pivot]
            else:
                candidates = set()

            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bronk(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)

        bronk(set(), set(range(n)), set())

        # Also try to find large non-maximal cliques by greedy expansion
        # For each edge, try to grow it into a clique
        additional_cliques = []
        for u, v in edge_set:
            if time.time() > start_time + time_budget:
                break
            # Try to grow {u, v} into a larger clique
            clique = {u, v}
            candidates = adj[u] & adj[v]
            while candidates:
                # Pick vertex with max connections to current clique
                best_w = None
                best_score = -1
                for w in candidates:
                    score = len(adj[w] & clique)
                    if score > best_score:
                        best_score = score
                        best_w = w
                if best_w is None:
                    break
                clique.add(best_w)
                candidates = candidates & adj[best_w]
            if len(clique) >= 3:
                additional_cliques.append(frozenset(clique))

        # Combine all candidate cliques (deduplicate)
        candidate_cliques = list(set(all_maximal_cliques) | set(additional_cliques))

        # Filter: only cliques with at least 2 vertices
        candidate_cliques = [c for c in candidate_cliques if len(c) >= 2]

        # Map each candidate clique to the edges it covers
        clique_edges = []
        for c in candidate_cliques:
            verts = list(c)
            es = set()
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    u, v = verts[i], verts[j]
                    e = (u, v) if u < v else (v, u)
                    if e in edge_set:
                        es.add(e)
            if es:
                clique_edges.append(es)

        # Greedy set cover: pick clique covering most uncovered edges
        remaining = set(edge_set)
        selected = []
        clique_idx_remaining = list(range(len(clique_edges)))

        # Precompute edge counts for sorting
        edge_counts = [(len(es), i) for i, es in enumerate(clique_edges)]
        edge_counts.sort(reverse=True)

        while remaining:
            if time.time() > start_time + time_budget:
                break

            best_idx = -1
            best_cover = 0
            for count, idx in edge_counts:
                if idx not in clique_idx_remaining:
                    continue
                cover = len(clique_edges[idx] & remaining)
                if cover > best_cover:
                    best_cover = cover
                    best_idx = idx
                if cover == 0:
                    continue

            if best_idx == -1:
                # Should not happen, but fallback
                e = next(iter(remaining))
                selected.append(frozenset(e))
                remaining.discard(e)
                continue

            selected.append(candidate_cliques[best_idx])
            remaining -= clique_edges[best_idx]
            clique_idx_remaining.remove(best_idx)

            # Remove edge_counts entries for removed cliques (lazy)

        # If we ran out of time, add remaining edges as 2-cliques
        if remaining:
            for e in remaining:
                selected.append(frozenset(e))

        # Convert to lists
        return [list(c) for c in selected]

    c1 = solve(random_graph(32, 0.6, 601), time_budget=70)
    c2 = solve(random_graph(64, 0.55, 602), time_budget=65)

    return [c1, c2]

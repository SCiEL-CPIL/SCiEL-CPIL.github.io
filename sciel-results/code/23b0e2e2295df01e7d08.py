import time
import random
from itertools import combinations

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    results = []
    start_time = time.time()
    deadline = start_time + 110.0

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        rng = random.Random(seed)
        neighbors = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u + 1, k):
                neighbors[u].add(v)
                neighbors[v].add(u)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[u].add(v)
                    neighbors[v].add(u)

        cliques = []
        covered_edges_in_C = set()

        I_vertices = list(range(k, n))
        I_vertices.sort(key=lambda v: len(neighbors[v] & set(range(k))), reverse=True)

        for v in I_vertices:
            if time.time() > deadline:
                break
            N_v = neighbors[v] & set(range(k))
            if not N_v:
                continue

            # Find largest valid subset S of N_v where no pair is already covered
            S = set(N_v)
            changed = True
            while changed:
                changed = False
                # Find all conflicting vertices
                conflict_count = {}
                for u in S:
                    for w in S:
                        if u < w and (u, w) in covered_edges_in_C:
                            conflict_count[u] = conflict_count.get(u, 0) + 1
                            conflict_count[w] = conflict_count.get(w, 0) + 1
                # Remove the vertex with the most conflicts
                if conflict_count:
                    worst = max(conflict_count, key=lambda x: conflict_count[x])
                    S.discard(worst)
                    changed = True

            if len(S) >= 2:
                cliques.append([v] + sorted(S))
                for u1, u2 in combinations(sorted(S), 2):
                    covered_edges_in_C.add((u1, u2))
            elif len(S) == 1:
                u = next(iter(S))
                cliques.append([v, u])

            for u in N_v:
                if u not in S:
                    cliques.append([v, u])

        # Handle remaining edges in C
        remaining_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_edges_in_C:
                    remaining_edges.add((u, w))

        rem_set = set(remaining_edges)
        while rem_set:
            if time.time() > deadline:
                break
            # Pick edge with minimum degree in remaining graph to grow
            min_deg_edge = None
            min_deg = float('inf')
            for (u, w) in rem_set:
                deg = 0
                for x in range(k):
                    if x != u and x != w:
                        if (min(u, x), max(u, x)) in rem_set and (min(w, x), max(w, x)) in rem_set:
                            deg += 1
                if deg < min_deg:
                    min_deg = deg
                    min_deg_edge = (u, w)
                if deg == 0:
                    break

            u, w = min_deg_edge
            clique_vertices = {u, w}

            # Try to add other vertices that connect to all in clique_vertices
            candidates = []
            for x in range(k):
                if x not in clique_vertices:
                    is_connected = True
                    for y in clique_vertices:
                        if (min(x, y), max(x, y)) not in rem_set:
                            is_connected = False
                            break
                    if is_connected:
                        candidates.append(x)

            clique_vertices.update(candidates)

            for u1, u2 in combinations(sorted(clique_vertices), 2):
                rem_set.discard((u1, u2))

            cliques.append(sorted(clique_vertices))

        # Safety net: check for any uncovered edges
        all_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_edges.add((u, w))
        for v in range(k, n):
            for u in range(k):
                if u in neighbors[v]:
                    all_edges.add((min(u, v), max(u, v)))

        covered_all = set()
        for cl in cliques:
            for u1, u2 in combinations(cl, 2):
                covered_all.add((min(u1, u2), max(u1, u2)))

        uncovered = all_edges - covered_all
        unc_set = set(uncovered)
        while unc_set:
            u, w = next(iter(unc_set))
            clique_vertices = {u, w}
            candidates = []
            for x in range(n):
                if x not in clique_vertices:
                    is_connected = True
                    for y in clique_vertices:
                        if (min(x, y), max(x, y)) not in unc_set:
                            is_connected = False
                            break
                    if is_connected:
                        candidates.append(x)
            clique_vertices.update(candidates)
            for u1, u2 in combinations(sorted(clique_vertices), 2):
                unc_set.discard((u1, u2))
            if len(clique_vertices) >= 2:
                cliques.append(sorted(clique_vertices))

        # Convert to ensure proper types
        results.append([[int(x) for x in cl] for cl in cliques])

    return results

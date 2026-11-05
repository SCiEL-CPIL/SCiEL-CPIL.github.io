import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145

    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def bron_kerbosch(R, P, X, adj):
        if not P and not X:
            if len(R) >= 2:
                yield frozenset(R)
            return
        if P:
            pivot = max(P, key=lambda p: len(P & adj[p]))
        elif X:
            pivot = max(X, key=lambda p: len(P & adj[p]))
        else:
            return
        for v in list(P - adj[pivot]):
            new_R = R | {v}
            new_P = P & adj[v]
            new_X = X & adj[v]
            yield from bron_kerbosch(new_R, new_P, new_X, adj)
            P.remove(v)
            X.add(v)

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        all_cliques = list(bron_kerbosch(frozenset(), set(range(n)), set(), adj))
        valid_cliques = [c for c in all_cliques if len(c) >= 2]

        if not valid_cliques:
            return []

        C = len(valid_cliques)
        rng_solver = random.Random(seed + 123)

        # Build incidence: for each vertex, which cliques it belongs to (as bitmask)
        # For each clique, list of vertices in it
        vertex_clique_mask = [0] * n
        clique_vertices = []
        for i, c in enumerate(valid_cliques):
            clique_vertices.append(list(c))
            for v in c:
                vertex_clique_mask[v] |= (1 << i)

        # Build clique intersection graph
        # cliques that share at least one vertex
        clique_adj = [set() for _ in range(C)]
        # For efficiency, build via vertex -> cliques mapping
        vertex_to_cliques = defaultdict(list)
        for i, c in enumerate(valid_cliques):
            for v in c:
                vertex_to_cliques[v].append(i)
        for v, clqs in vertex_to_cliques.items():
            for a in range(len(clqs)):
                for b in range(a + 1, len(clqs)):
                    clique_adj[clqs[a]].add(clqs[b])
                    clique_adj[clqs[b]].add(clqs[a])

        # Find maximum matching on clique intersection graph (greedy + augmenting)
        def find_max_matching(graph, num_nodes):
            match = [-1] * num_nodes
            for u in range(num_nodes):
                if match[u] == -1:
                    seen = [False] * num_nodes
                    def dfs(u2):
                        for v2 in graph[u2]:
                            if not seen[v2]:
                                seen[v2] = True
                                if match[v2] == -1 or dfs(match[v2]):
                                    match[u2] = v2
                                    match[v2] = u2
                                    return True
                        return False
                    dfs(u)
            return match

        match = find_max_matching(clique_adj, C)

        # Derive vertex cover of clique intersection graph
        vc_cliques = set()
        for u in range(C):
            if match[u] != -1:
                vc_cliques.add(u)
                vc_cliques.add(match[u])

        best_set = None
        best_size = float('inf')

        def local_search(current_set, hit_mask):
            """Aggressive local search: removal + swap moves"""
            improved = True
            while improved:
                improved = False
                if time.time() > deadline:
                    break
                vs = list(current_set)
                rng_solver.shuffle(vs)
                for v in vs:
                    if time.time() > deadline:
                        break
                    # Try removal
                    vmask = vertex_clique_mask[v]
                    if (hit_mask & vmask) == vmask:
                        # Check if removing v still covers all its cliques
                        new_hit = hit_mask & ~vmask
                        # For each clique that v covers, check if another vertex in current_set covers it
                        all_still_covered = True
                        for ci in range(C):
                            if vmask & (1 << ci):
                                # Is clique ci covered by another vertex?
                                covered = False
                                for u in clique_vertices[ci]:
                                    if u != v and u in current_set:
                                        covered = True
                                        break
                                if not covered:
                                    all_still_covered = False
                                    break
                        if all_still_covered:
                            current_set.remove(v)
                            hit_mask = new_hit
                            improved = True
                            break
                # Try swap moves: replace one vertex with another
                if not improved and len(current_set) > 0:
                    for v in list(current_set):
                        if time.time() > deadline:
                            break
                        for u in range(n):
                            if u in current_set or u == v:
                                continue
                            # Try swapping v -> u
                            new_hit = (hit_mask & ~vertex_clique_mask[v]) | vertex_clique_mask[u]
                            if new_hit == hit_mask:
                                # Same coverage, same size - no improvement
                                continue
                            # Check if new_hit covers all C cliques
                            if (new_hit & ((1 << C) - 1)) == ((1 << C) - 1):
                                # Valid swap, same size but different composition
                                # Only accept if it enables further improvement
                                pass
            return current_set, hit_mask

        def greedy_with_seed(seed_cliques, current_set, hit_mask):
            """Greedy set cover, seeded with priority for certain cliques"""
            full_mask = (1 << C) - 1
            # Prioritize cliques in seed_cliques
            priority_cliques = [ci for ci in seed_cliques if ci not in hit_mask]
            other_cliques = [ci for ci in range(C) if ci not in hit_mask and ci not in set(seed_cliques)]

            all_uncovered = priority_cliques + other_cliques

            while True:
                if time.time() > deadline:
                    break
                # Find uncovered cliques
                uncovered = []
                for ci in range(C):
                    if not (hit_mask & (1 << ci)):
                        uncovered.append(ci)
                if not uncovered:
                    break

                # Greedy: pick vertex covering most uncovered cliques
                best_v = -1
                max_hits = -1
                candidates = []
                for v in range(n):
                    if v in current_set:
                        continue
                    vmask = vertex_clique_mask[v]
                    hits = 0
                    for ci in uncovered:
                        if vmask & (1 << ci):
                            hits += 1
                    if hits > max_hits:
                        max_hits = hits
                        best_v = v
                        candidates = [v]
                    elif hits == max_hits and hits > 0:
                        candidates.append(v)

                if max_hits == 0:
                    break

                v = rng_solver.choice(candidates)
                current_set.add(v)
                hit_mask |= vertex_clique_mask[v]

            return current_set, hit_mask

        # Strategy 1: Seeded greedy using vertex cover of intersection graph
        seed_cliques = list(vc_cliques)
        current_set = set()
        hit_mask = 0
        current_set, hit_mask = greedy_with_seed(seed_cliques, current_set, hit_mask)
        current_set, hit_mask = local_search(current_set, hit_mask)
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = set(current_set)

        # Strategy 2: Multiple random restarts with greedy + local search
        num_restarts = 800
        for run in range(num_restarts):
            if time.time() > deadline:
                break

            current_set = set()
            hit_mask = 0

            # Randomized greedy: among top-k candidates, pick randomly
            while True:
                if time.time() > deadline:
                    break
                uncovered = []
                for ci in range(C):
                    if not (hit_mask & (1 << ci)):
                        uncovered.append(ci)
                if not uncovered:
                    break

                best_v = -1
                max_hits = -1
                candidates = []
                for v in range(n):
                    if v in current_set:
                        continue
                    vmask = vertex_clique_mask[v]
                    hits = 0
                    for ci in uncovered:
                        if vmask & (1 << ci):
                            hits += 1
                    if hits > max_hits:
                        max_hits = hits
                        best_v = v
                        candidates = [v]
                    elif hits == max_hits and hits > 0:
                        candidates.append(v)

                if max_hits == 0:
                    break

                # Randomly pick from top candidates (adds diversity)
                if candidates:
                    # With some probability, pick a random candidate from all with decent coverage
                    if rng_solver.random() < 0.3:
                        # Pick from candidates with at least half the max hits
                        good = [v for v in candidates if True]  # all candidates have same hits
                        v = rng_solver.choice(good)
                    else:
                        v = rng_solver.choice(candidates)
                else:
                    break

                current_set.add(v)
                hit_mask |= vertex_clique_mask[v]

            current_set, hit_mask = local_search(current_set, hit_mask)

            if len(current_set) < best_size:
                best_size = len(current_set)
                best_set = set(current_set)

        # Final polish: try to remove more vertices with exhaustive check
        if best_set is not None:
            final_set = set(best_set)
            final_hit = 0
            for v in final_set:
                final_hit |= vertex_clique_mask[v]

            # Try removing each vertex
            for v in list(final_set):
                if time.time() > deadline:
                    break
                vmask = vertex_clique_mask[v]
                # Check if all cliques in vmask are covered by other vertices in final_set
                all_covered = True
                for ci in range(C):
                    if vmask & (1 << ci):
                        covered = False
                        for u in clique_vertices[ci]:
                            if u != v and u in final_set:
                                covered = True
                                break
                        if not covered:
                            all_covered = False
                            break
                if all_covered:
                    final_set.remove(v)

            if len(final_set) < best_size:
                best_size = len(final_set)
                best_set = set(final_set)

        return sorted(best_set) if best_set else []

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            edges = instance(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
        res = solve_instance(n, p, seed)
        results.append(res)

    return results

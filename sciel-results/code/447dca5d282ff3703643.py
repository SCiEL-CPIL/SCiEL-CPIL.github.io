def construct():
    import time
    from random import Random

    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build cross edges
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return cross_edges

    def solve_instance(n, k, p, seed, variant_seed=None):
        cross_edges = instance(n, k, p, seed)

        # N[v] is list of neighbors in C for v in I
        N = {v: [] for v in range(k, n)}
        for u, v in cross_edges:
            N[v].append(u)

        # Build adjacency structure for C-vertices
        # adj_c[i] = set of C-vertices still connected to i via remaining C-edges
        adj_c = [set(range(k)) for _ in range(k)]
        for i in range(k):
            adj_c[i].discard(i)

        deg = [k - 1] * k  # remaining degree in C-subgraph

        def mark_used(a, b):
            adj_c[a].discard(b)
            adj_c[b].discard(a)
            deg[a] -= 1
            deg[b] -= 1

        mixed_cliques = []

        # Determine order of I vertices
        if variant_seed is None:
            I_vertices = sorted(range(k, n), key=lambda x: len(N[x]))
        else:
            rng_var = Random(variant_seed)
            if variant_seed % 3 == 0:
                I_vertices = sorted(range(k, n), key=lambda x: len(N[x]))
            elif variant_seed % 3 == 1:
                I_vertices = sorted(range(k, n), key=lambda x: len(N[x]) + rng_var.random() * 0.5)
            else:
                I_vertices = list(range(k, n))
                rng_var.shuffle(I_vertices)

        for v in I_vertices:
            neighbors = N[v]
            if not neighbors:
                continue

            # Find all C-edges where both endpoints are neighbors of v
            # and the edge is still available
            rem_set = set(neighbors)
            available_pairs = []
            for c_a in rem_set:
                for c_b in rem_set:
                    if c_b > c_a and c_b in adj_c[c_a]:
                        available_pairs.append((c_a, c_b))

            if not available_pairs:
                # Only pairs (v, c) for each neighbor
                for c in rem_set:
                    mixed_cliques.append([int(v), int(c)])
                continue

            # Greedy matching: pair up C-vertices to maximize coverage
            # Sort by sum of remaining degrees (descending) to prioritize high-value edges
            available_pairs.sort(key=lambda p: deg[p[0]] + deg[p[1]], reverse=True)

            matched = set()
            pairings = []
            for c_a, c_b in available_pairs:
                if c_a in matched or c_b in matched:
                    continue
                # Check edge still available
                if c_b not in adj_c[c_a]:
                    continue
                matched.add(c_a)
                matched.add(c_b)
                pairings.append((c_a, c_b))
                mark_used(c_a, c_b)

            for c_a, c_b in pairings:
                mixed_cliques.append([int(v), int(c_a), int(c_b)])

            for c in rem_set:
                if c not in matched:
                    mixed_cliques.append([int(v), int(c)])

        # Cover remaining C-edges with pure-C cliques
        pure_cliques = []

        # Find all remaining edges
        remaining_edges = []
        for i in range(k):
            for j in adj_c[i]:
                if j > i:
                    remaining_edges.append((i, j))

        # Greedy: repeatedly find a maximal clique among remaining edges
        while remaining_edges:
            if time.time() > deadline - 1.0:
                for a, b in remaining_edges:
                    pure_cliques.append([int(a), int(b)])
                break

            # Pick the vertex with max remaining degree
            max_deg = -1
            seed_v = -1
            for a, b in remaining_edges:
                if deg[a] > max_deg:
                    max_deg = deg[a]
                    seed_v = a
                if deg[b] > max_deg:
                    max_deg = deg[b]
                    seed_v = b

            if seed_v < 0:
                break

            # Build maximal clique containing seed_v
            S = [seed_v]
            S_set = {seed_v}

            # Candidates: all vertices that have an edge to seed_v
            cand = set(adj_c[seed_v])

            while cand:
                # Pick candidate with max degree within current S
                best_v = None
                best_score = -1
                for w in cand:
                    # Check w is connected to all in S
                    ok = True
                    for s in S:
                        if s not in adj_c[w]:
                            ok = False
                            break
                    if not ok:
                        continue
                    score = len(adj_c[w] & S_set)
                    if score > best_score:
                        best_score = score
                        best_v = w
                if best_v is None:
                    break
                S.append(best_v)
                S_set.add(best_v)
                # Update candidates
                new_cand = set()
                for w in cand:
                    if w == best_v:
                        continue
                    if best_v in adj_c[w]:
                        new_cand.add(w)
                cand = new_cand

            if len(S) >= 2:
                pure_cliques.append([int(x) for x in S])
                # Remove all edges within S from remaining
                S_set2 = set(S)
                remaining_edges = [(a, b) for a, b in remaining_edges
                                   if not (a in S_set2 and b in S_set2)]
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        mark_used(S[i], S[j])
            else:
                # Just take the first remaining edge
                a, b = remaining_edges[0]
                pure_cliques.append([int(a), int(b)])
                remaining_edges.remove((a, b))
                mark_used(a, b)

        return mixed_cliques + pure_cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []

    for (n, k, p, seed) in specs:
        best_cliques = None
        best_count = float('inf')

        if time.time() > deadline - 5.0:
            ce = instance(n, k, p, seed)
            cliques = [[int(u) for u in range(k)]]
            for (u, v) in ce:
                cliques.append([int(u), int(v)])
            result.append(cliques)
            continue

        # Try base first
        cliques = solve_instance(n, k, p, seed, None)
        count = len(cliques)
        if count < best_count:
            best_count = count
            best_cliques = cliques

        # Try variants
        for i in range(30):
            if time.time() > deadline - 15.0:
                break
            v_seed = seed + 1000 + i
            cliques = solve_instance(n, k, p, seed, v_seed)
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques

        result.append(best_cliques)

    # Ensure types
    final = []
    for partition in result:
        clean = []
        for clique in partition:
            clean.append([int(x) for x in clique])
        final.append(clean)
    return final

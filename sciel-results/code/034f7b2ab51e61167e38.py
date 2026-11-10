"""Clique edge-cover via greedy maximal-clique expansion on dense graphs."""

def construct():
    import time
    import random as _random

    DEADLINE = time.time() + 130.0  # safety margin before 150s kill

    def gen_edges(n, p, seed):
        rng = _random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = gen_edges(n, p, seed)
        # adjacency bitsets
        adj = [0] * n
        for (u, v) in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        # map edge -> index for tracking; we track uncovered as a set of (u,v)
        uncovered = set(edges)
        cliques = []

        # Precompute for each vertex the set of neighbors as bitset
        # Greedy: at each step, for each vertex with uncovered incident edges,
        # build a greedy clique and evaluate how many uncovered edges it covers.
        # Pick the best clique, commit, remove its edges.

        # To be efficient, we limit candidate seeds to vertices with many uncovered edges.
        while uncovered:
            if time.time() > DEADLINE:
                break

            # Count uncovered degree per vertex
            unc_deg = [0] * n
            for (u, v) in uncovered:
                unc_deg[u] += 1
                unc_deg[v] += 1

            # Candidate seeds: vertices with high uncovered degree
            # Sort by unc_deg desc, take top few
            seeds = sorted(range(n), key=lambda x: -unc_deg[x])
            seeds = [s for s in seeds if unc_deg[s] > 0][:max(1, n // 4)]

            best_clique = None
            best_covered = -1
            best_score = -1.0

            for s in seeds:
                # Greedily build a clique starting from s
                # Candidate pool: neighbors of s that have uncovered edges to s
                # We want vertices that are mutually adjacent and cover many uncovered edges
                clique = [s]
                # Available candidates: neighbors of all current clique members
                # Start with neighbors of s
                cand = adj[s] & ~((1 << s))
                # We'll greedily add vertices that maximize uncovered-edge contribution
                # For each candidate, compute how many uncovered edges it has to current clique
                # and to other candidates (for future)
                # Simpler: add vertices in order of unc_deg, if adjacent to all in clique
                # and has at least one uncovered edge to clique
                cand_list = [v for v in range(n) if (cand >> v) & 1]
                cand_list.sort(key=lambda x: -unc_deg[x])
                for v in cand_list:
                    # v must be adjacent to all in clique
                    ok = True
                    for c in clique:
                        if not ((adj[v] >> c) & 1):
                            ok = False
                            break
                    if not ok:
                        continue
                    # v must have at least one uncovered edge to some clique member
                    has_unc = False
                    for c in clique:
                        if (c, v) in uncovered or (v, c) in uncovered:
                            has_unc = True
                            break
                    if not has_unc:
                        continue
                    clique.append(v)
                    # Update: only keep candidates adjacent to v
                    cand &= adj[v]

                # Evaluate this clique: count uncovered edges it covers
                covered = 0
                cl = clique
                for i in range(len(cl)):
                    for j in range(i + 1, len(cl)):
                        a, b = cl[i], cl[j]
                        if (a, b) in uncovered or (b, a) in uncovered:
                            covered += 1
                # Score: prefer more covered edges; break ties by smaller clique
                if covered > best_covered or (covered == best_covered and len(clique) < len(best_clique) if best_clique else False):
                    best_covered = covered
                    best_clique = clique
                    best_score = covered / max(1, len(clique))

            if best_clique is None or best_covered <= 0:
                # Fallback: pick a single edge
                e = next(iter(uncovered))
                cliques.append(list(e))
                uncovered.discard(e)
                continue

            # Commit: remove all edges of best_clique from uncovered
            cl = best_clique
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    uncovered.discard((a, b))
                    uncovered.discard((b, a))
            cliques.append(cl)

        # If we ran out of time, handle remaining edges as single-edge cliques
        if uncovered:
            for e in uncovered:
                cliques.append(list(e))
            uncovered.clear()

        return cliques

    result = [
        solve_instance(32, 0.6, 601),
        solve_instance(64, 0.55, 602),
    ]
    return result

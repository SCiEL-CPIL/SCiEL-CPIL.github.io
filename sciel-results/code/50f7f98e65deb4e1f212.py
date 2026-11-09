"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start = time.time()
    deadline = start + 80.0

    def time_left():
        return deadline - time.time()

    n = P['n']
    edges = P['edges']

    # Build adjacency sets
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))

    all_edges = list(edge_set)
    E = len(all_edges)

    # --- Biclique mining: expand a seed (L, R) to maximal biclique in full graph ---
    def expand(L0, R0):
        L = set(L0)
        R = set(R0)
        changed = True
        while changed:
            changed = False
            # Add to L: vertex w not in L∪R that connects to all of R
            for w in range(n):
                if w in L or w in R:
                    continue
                if all(w in adj[r] for r in R):
                    L.add(w)
                    changed = True
            # Add to R: vertex w not in L∪R that connects to all of L
            for w in range(n):
                if w in L or w in R:
                    continue
                if all(w in adj[l] for l in L):
                    R.add(w)
                    changed = True
        return L, R

    def biclique_edges(L, R):
        es = set()
        for u in L:
            for v in R:
                if u != v:
                    es.add((min(u, v), max(u, v)))
        return es

    # --- Mine a pool of candidate bicliques ---
    # Seeds: every edge, both orientations, plus random vertex pairs
    candidates = []  # list of (L_frozenset, R_frozenset, edge_set)
    seen = set()

    def add_candidate(L, R):
        if not L or not R:
            return
        if L & R:
            return
        key = (frozenset(L), frozenset(R))
        if key in seen:
            return
        # Verify all cross pairs are edges
        for u in L:
            for v in R:
                if v not in adj[u]:
                    return
        es = biclique_edges(L, R)
        if not es:
            return
        seen.add(key)
        candidates.append((L, R, es))

    # Edge-seeded expansion (both orientations)
    for (u, v) in all_edges:
        L, R = expand({u}, {v})
        add_candidate(L, R)
        L2, R2 = expand({v}, {u})
        add_candidate(L2, R2)

    # Vertex-pair seeds (both orientations) for more diversity
    for u in range(n):
        for v in range(u + 1, n):
            if v in adj[u]:
                continue  # already covered by edge seed
            # Even non-adjacent: try anyway (may yield empty)
            L, R = expand({u}, {v})
            if L and R and not (L & R):
                # Only add if valid (cross edges exist)
                valid = all(w in adj[x] for w in L for x in R if w != x)
                if valid:
                    add_candidate(L, R)

    # Also try single-vertex left/right with empty other side -> expand
    for u in range(n):
        L, R = expand({u}, set())
        add_candidate(L, R)
        L2, R2 = expand(set(), {u})
        add_candidate(L2, R2)

    # --- Greedy set-cover: pick bicliques maximizing marginal edge coverage ---
    def greedy_set_cover(cand_list, remaining_edges):
        """cand_list: list of (L, R, es). remaining_edges: set of edges to cover."""
        chosen = []
        covered = set()
        rem = set(remaining_edges)
        # Precompute for speed: use list of (es_set, L, R)
        items = [(es, L, R) for (L, R, es) in cand_list]

        while rem and time_left() > 0.5:
            best_gain = 0
            best_idx = -1
            for i, (es, L, R) in enumerate(items):
                if not es:
                    continue
                gain = len(es & rem)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx < 0:
                break
            es, L, R = items[best_idx]
            chosen.append((L, R))
            covered |= es
            rem -= es
        return chosen, rem

    # --- Local improvement: try to replace 2 bicliques with 1, or 1 with fewer ---
    def try_local_improve(bicliques):
        """Attempt to reduce count via merging or splitting."""
        if time_left() < 2.0:
            return bicliques

        improved = True
        while improved and time_left() > 2.0:
            improved = False
            cur = list(bicliques)
            m = len(cur)

            # Try merging pairs (i, j) into a single biclique
            for i in range(m):
                if time_left() < 2.0:
                    break
                for j in range(i + 1, m):
                    Li, Ri = cur[i]
                    Lj, Rj = cur[j]
                    # Candidate merged biclique: try (Li∪Lj, Ri∪Rj) and variants
                    for (Lc, Rc) in [
                        (Li | Lj, Ri | Rj),
                        (Li | Rj, Ri | Lj),
                        (Li, Ri | Rj),
                        (Li | Lj, Ri),
                    ]:
                        if not Lc or not Rc or (Lc & Rc):
                            continue
                        # Check validity
                        valid = True
                        for u in Lc:
                            for v in Rc:
                                if v not in adj[u]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        merged_es = biclique_edges(Lc, Rc)
                        # The merged biclique must cover all edges of i and j
                        es_i = biclique_edges(Li, Ri)
                        es_j = biclique_edges(Lj, Rj)
                        if (es_i | es_j) <= merged_es:
                            # Replace i and j with merged
                            new_bicliques = [cur[k] for k in range(m) if k != i and k != j]
                            new_bicliques.append((Lc, Rc))
                            return new_bicliques

            # Try replacing a single biclique with a smaller set (split)
            for i in range(m):
                if time_left() < 2.0:
                    break
                Li, Ri = cur[i]
                es_i = biclique_edges(Li, Ri)
                # Try to cover es_i with 2 bicliques from candidates
                # This is expensive; skip for now
        return bicliques

    # --- Main: greedy set cover over all candidates ---
    bicliques, leftover = greedy_set_cover(candidates, set(all_edges))

    # Handle leftover edges as single-edge bicliques
    for e in leftover:
        bicliques.append(({e[0]}, {e[1]}))

    # Local improvement
    bicliques = try_local_improve(bicliques)

    # --- Second pass: re-mine on leftover structure for better coverage ---
    if time_left() > 3.0:
        # Recompute remaining edges not covered
        covered = set()
        for (L, R) in bicliques:
            covered |= biclique_edges(L, R)
        rem = edge_set - covered
        if rem:
            # Mine new candidates restricted to rem
            new_cands = []
            seen2 = set()
            for (u, v) in rem:
                L, R = expand({u}, {v})
                key = (frozenset(L), frozenset(R))
                if key not in seen2:
                    valid = all(w in adj[x] for w in L for x in R if w != x)
                    if valid:
                        es = biclique_edges(L, R)
                        if es & rem:
                            seen2.add(key)
                            new_cands.append((L, R, es))
            if new_cands:
                # Greedy cover of rem
                chosen2, rem2 = greedy_set_cover(new_cands, rem)
                for (L, R) in chosen2:
                    bicliques.append((L, R))
                for e in rem2:
                    bicliques.append(({e[0]}, {e[1]}))

    # --- Final local improvement ---
    if time_left() > 1.0:
        bicliques = try_local_improve(bicliques)

    # --- Convert to required format: list of [left_list, right_list] ---
    # Ensure types are Python int and lists
    result = []
    for (L, R) in bicliques:
        left = [int(x) for x in sorted(L)]
        right = [int(x) for x in sorted(R)]
        if left and right:
            result.append([left, right])

    return result
# EVOLVE-BLOCK-END

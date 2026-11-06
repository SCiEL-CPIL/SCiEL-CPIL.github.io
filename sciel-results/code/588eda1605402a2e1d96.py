import time
from random import Random


def _generate_edges(n, p, seed):
    rng = Random(seed)
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
    return edges


def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _solve_instance(n, edges, deadline):
    adj = _build_adj(n, edges)
    uncovered = set(edges)
    cliques = []
    if not uncovered:
        return cliques

    while uncovered:
        if time.time() > deadline:
            break

        # Choose seed uncovered edge maximizing number of vertices w
        # that could extend the clique {u,v} using only uncovered edges.
        best_seed = None
        best_pot = -1
        for (u, v) in uncovered:
            common = adj[u] & adj[v]
            pot = 0
            for w in common:
                a, b = (u, w) if u < w else (w, u)
                c, d = (v, w) if v < w else (w, v)
                if a == b or c == d:
                    continue
                if (a, b) in uncovered and (c, d) in uncovered:
                    pot += 1
            if pot > best_pot:
                best_pot = pot
                best_seed = (u, v)
                if pot >= n - 2:
                    break
        if best_seed is None:
            break

        u, v = best_seed
        C = [u, v]
        Cset = {u, v}
        cand = (adj[u] & adj[v]) - Cset

        # Greedily extend the clique: add a vertex w if all its edges to C are uncovered.
        while cand:
            best_w = None
            best_gain = 0
            for w in cand:
                gain = 0
                for c in C:
                    a, b = (w, c) if w < c else (c, w)
                    if (a, b) in uncovered:
                        gain += 1
                if gain == len(C):
                    # All edges from w to C are uncovered, so w can join C.
                    # Prefer the one that maximizes potential further extensions.
                    extra = 0
                    for x in cand:
                        if x == w:
                            continue
                        # Check if x is adjacent to w (already in cand so yes) 
                        # and if edges (x, c) for c in C are uncovered
                        can_join = True
                        for c in C:
                            a2, b2 = (x, c) if x < c else (c, x)
                            if (a2, b2) not in uncovered:
                                can_join = False
                                break
                        if can_join:
                            extra += 1
                    if extra > best_gain:
                        best_gain = extra
                        best_w = w
                    # Early exit: any valid w is acceptable; we don't need the absolute best
                    # for correctness, but picking a good one helps. 
                    # For speed, we can just take the first valid w or the best by extra.
                    # Let's take the first valid w to save time, or the best if fast enough.
                    # Given dense graphs, cand is small. Let's keep the best selection.
            if best_w is None:
                break
            C.append(best_w)
            Cset.add(best_w)
            # Update candidates: only those in cand that are adjacent to best_w
            # and whose edges to all current C are uncovered (this is implicit in the next loop's check,
            # but we can prune cand to those adjacent to best_w to reduce search space)
            cand = cand & adj[best_w]
            # Further prune: remove vertices that don't have all edges to C uncovered.
            # This is done implicitly in the next iteration's gain check, 
            # but explicit pruning can speed things up.
            new_cand = set()
            for w in cand:
                valid = True
                for c in C:
                    a, b = (w, c) if w < c else (c, w)
                    if (a, b) not in uncovered:
                        valid = False
                        break
                if valid:
                    new_cand.add(w)
            cand = new_cand

        cliques.append(C)
        for i in range(len(C)):
            a = C[i]
            for j in range(i + 1, len(C)):
                b = C[j]
                uncovered.discard((a, b) if a < b else (b, a))

    # Cover remaining edges with individual edges
    for e in uncovered:
        cliques.append(list(e))
    return cliques


def construct():
    start = time.time()
    deadline = start + 140.0  # Leave 10s margin before the 150s kill
    results = []
    for (n, p, seed) in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # If we've already exceeded deadline, return empty lists to avoid timeout
            # But we must return something valid. Empty list is invalid if there are edges.
            # Let's just try to finish the current instance quickly or return a trivial cover.
            pass
        edges = _generate_edges(n, p, seed)
        cliques = _solve_instance(n, edges, deadline)
        results.append(cliques)
    return results

"""Clique partition of split graphs via greedy Type-B set cover + Max-Clique cover."""
import time
from random import Random
from itertools import combinations


def construct():
    _T0 = time.time()
    _DEADLINE = 130.0

    def _time_up():
        return (time.time() - _T0) > _DEADLINE

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _edge(a, b):
        return (a, b) if a < b else (b, a)

    def _max_clique_bronk(adj, n):
        """Bron-Kerbosch with pivot to find a maximum clique."""
        best = set()

        def bk(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = set(R)
                return
            # Prune: even adding all P can't beat best
            if len(R) + len(P) <= len(best):
                return
            # Pivot
            pivot = max(P | X, key=lambda u: len(adj[u] & P)) if (P | X) else None
            candidates = list(P - adj[pivot]) if pivot is not None else list(P)
            for v in candidates:
                R.add(v)
                bk(R, P & adj[v], X & adj[v])
                R.discard(v)
                P.discard(v)
                X.add(v)

        bk(set(), set(range(n)), set())
        return best

    def _cover_cc_edges(remaining_cc, k):
        """Cover C-C edges with C-only cliques using max clique extraction."""
        if not remaining_cc:
            return []
        cliques = []
        remaining = set(remaining_cc)
        while remaining:
            # Build adjacency on active vertices
            active = set()
            for (a, b) in remaining:
                active.add(a)
                active.add(b)
            if not active:
                break
            active_list = sorted(active)
            idx = {v: i for i, v in enumerate(active_list)}
            m = len(active_list)
            adj = [set() for _ in range(m)]
            for (a, b) in remaining:
                ia, ib = idx[a], idx[b]
                adj[ia].add(ib)
                adj[ib].add(ia)
            mc = _max_clique_bronk(adj, m)
            if not mc:
                # Fallback: pick any edge
                a, b = next(iter(remaining))
                mc = {idx[a], idx[b]}
            cl = sorted(active_list[i] for i in mc)
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    remaining.discard(_edge(cl[i], cl[j]))
            cliques.append(cl)
        return cliques

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set()
        for (a, b) in edges:
            edge_set.add(_edge(a, b))

        # All C-C edges
        all_cc = set(_edge(u, v) for u in range(k) for v in range(u + 1, k))

        # Build adjacency: for each I-vertex, which C-vertices it connects to
        adj_c = {}
        for w in range(k, n):
            adj_c[w] = set(u for u in range(k) if _edge(u, w) in edge_set)

        # Precompute all possible Type-B cliques: (u, v, w) where u<v in C, w in I,
        # and u-w, v-w are edges (u-v is always an edge since C is a clique)
        # Group by C-C edge: for each (u,v), list of w's that form a Type-B clique
        tb_by_cc = {}  # cc_edge -> list of w's
        for u in range(k):
            for v in range(u + 1, k):
                e = _edge(u, v)
                ws = []
                for w in range(k, n):
                    if u in adj_c[w] and v in adj_c[w]:
                        ws.append(w)
                if ws:
                    tb_by_cc[e] = ws

        # Greedy set cover: repeatedly pick the C-C edge with most available Type-B cliques,
        # then pick the w that covers the most additional uncovered cross edges
        covered_cc = set()
        covered_cross = set()
        type_b_cliques = []

        # Track remaining C-C edges that can still be covered by Type-B
        remaining_cc_with_tb = {e: list(ws) for e, ws in tb_by_cc.items()}

        while remaining_cc_with_tb:
            # Find the C-C edge with the most available w's
            best_e = None
            best_count = 0
            for e, ws in remaining_cc_with_tb.items():
                if len(ws) > best_count:
                    best_count = len(ws)
                    best_e = e
            if best_e is None:
                break

            u, v = best_e
            ws = remaining_cc_with_tb[best_e]
            if not ws:
                del remaining_cc_with_tb[best_e]
                continue

            # Pick the w that maximizes number of uncovered cross edges it covers
            # Cross edges for w: (u, w) and (v, w)
            # But w might also be useful for other C-C edges... however w can only be in ONE clique
            # So we just pick the w that covers the most uncovered cross edges
            best_w = None
            best_score = -1
            for w in ws:
                score = 0
                if _edge(u, w) not in covered_cross:
                    score += 1
                if _edge(v, w) not in covered_cross:
                    score += 1
                # Also consider: this w might be the only option for other C-C edges
                # But since w can only be in one clique, we just pick greedily
                if score > best_score:
                    best_score = score
                    best_w = w

            if best_w is None:
                best_w = ws[0]

            # Add the Type-B clique
            type_b_cliques.append([u, v, best_w])
            covered_cc.add(best_e)
            covered_cross.add(_edge(u, best_w))
            covered_cross.add(_edge(v, best_w))

            # Remove best_e from remaining
            del remaining_cc_with_tb[best_e]

            # Remove best_w from all other C-C edges' w lists
            for e in list(remaining_cc_with_tb.keys()):
                if best_w in remaining_cc_with_tb[e]:
                    remaining_cc_with_tb[e].remove(best_w)
                    if not remaining_cc_with_tb[e]:
                        del remaining_cc_with_tb[e]

        # Cover remaining C-C edges with C-only cliques
        remaining_cc = all_cc - covered_cc
        c_only_cliques = _cover_cc_edges(remaining_cc, k)

        # Cover remaining cross edges as 2-cliques
        cross_cliques = []
        for w in range(k, n):
            for u in range(k):
                e = _edge(u, w)
                if e in edge_set and e not in covered_cross:
                    cross_cliques.append([u, w])
                    covered_cross.add(e)

        cliques = type_b_cliques + c_only_cliques + cross_cliques

        # Validate: every clique must be a valid clique with at least 2 vertices
        validated = []
        for cl in cliques:
            if len(cl) < 2:
                continue
            valid = True
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    if _edge(cl[i], cl[j]) not in edge_set:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                validated.append(cl)

        return validated

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if _time_up():
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            edge_set = set()
            for (a, b) in edges:
                edge_set.add(_edge(a, b))
            cliques = [list(range(k))]
            for w in range(k, n):
                for u in range(k):
                    if _edge(u, w) in edge_set:
                        cliques.append([u, w])
            results.append(cliques)
            continue
        results.append(solve_one(*spec))

    out = []
    for clist in results:
        nl = []
        for c in clist:
            nl.append([int(x) for x in c])
        out.append(nl)
    return out

"""Exact biclique edge partition via maximal-biclique pool + Algorithm X (DLX) exact cover."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _biclique_edges(left, right):
    return {(min(u, v), max(u, v)) for u in left for v in right}


def _validate(bicliques, adj, all_edges_set):
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        if set(left).intersection(right):
            return False
        for u in left:
            for v in right:
                if v not in adj[u]:
                    return False
                e = (min(u, v), max(u, v))
                if e in covered:
                    return False
                covered.add(e)
    return covered == all_edges_set


def _maximal_bicliques(adj, n, deadline, cap=200000):
    """Bron-Kerbosch on bipartite non-edge conflict graph.
    Returns list of (left_frozenset, right_frozenset) maximal bicliques."""
    # conflict within same side: vertices not adjacent in G
    Lconf = [set() for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            if b not in adj[a]:
                Lconf[a].add(b)
                Lconf[b].add(a)
    # cross conflict: edge present in G -> cannot be on opposite sides of a biclique
    Rconf = [set() for _ in range(n)]
    for a in range(n):
        for b in range(n):
            if a != b and b in adj[a]:
                Rconf[a].add(b)
                Rconf[b].add(a)

    # Deduplicate: only keep one canonical orientation per (L,R) pair
    seen = set()
    out = []
    count = [0]

    def emit(L, R):
        if not L or not R:
            return
        key = (frozenset(L), frozenset(R))
        key2 = (frozenset(R), frozenset(L))
        if key in seen or key2 in seen:
            return
        seen.add(key)
        out.append(key)

    def expand(Pset, X, Y):
        if time.time() > deadline or count[0] >= cap:
            return
        if not Pset:
            if X and Y:
                emit(X, Y)
            return
        # choose pivot with max conflicts in P
        best_p = None
        best_cnt = -1
        for v in Pset:
            c = len((Lconf[v] & Pset) | (Rconf[v] & Pset))
            if c > best_cnt:
                best_cnt = c
                best_p = v
        if best_p is not None:
            cand = Pset - (Lconf[best_p] | Rconf[best_p])
        else:
            cand = set(Pset)
        for v in list(cand):
            count[0] += 1
            newP = (Pset & Lconf[v]) & (Pset & Rconf[v])
            expand(newP, X | {v}, Y | {v})
            Pset.remove(v)
            X.add(v)

    expand(set(range(n)), set(), set())
    return out


def _greedy_partition(cands, all_edges_set, deadline):
    """Greedy set cover over candidate bicliques (by edge count)."""
    cand_edges = [_biclique_edges(set(l), set(r)) for l, r in cands]
    remaining = set(all_edges_set)
    chosen = []
    used = [False] * len(cands)
    while remaining:
        if time.time() > deadline:
            break
        best_i = -1
        best_cov = 0
        for i in range(len(cands)):
            if used[i]:
                continue
            cov = len(cand_edges[i] & remaining)
            if cov > best_cov:
                best_cov = cov
                best_i = i
        if best_i < 0:
            e = next(iter(remaining))
            chosen.append((frozenset({e[0]}), frozenset({e[1]})))
            remaining.discard(e)
        else:
            used[best_i] = True
            chosen.append(cands[best_i])
            remaining -= cand_edges[best_i]
    return chosen


def _local_search(bicliques, adj, all_edges_set, deadline):
    bicliques = [[list(l), list(r)] for l, r in bicliques]

    def edge_set_of(bc):
        return _biclique_edges(set(bc[0]), set(bc[1]))

    improved = True
    while improved and time.time() < deadline:
        improved = False
        edge_sets = [edge_set_of(bc) for bc in bicliques]
        for i in range(len(bicliques)):
            for j in range(i + 1, len(bicliques)):
                if time.time() > deadline:
                    break
                Li, Ri = set(bicliques[i][0]), set(bicliques[i][1])
                Lj, Rj = set(bicliques[j][0]), set(bicliques[j][1])
                left_opts = [Li, Lj, Li | Lj, Li & Lj]
                right_opts = [Ri, Rj, Ri | Rj, Ri & Rj]
                found = False
                for nl in left_opts:
                    if not nl:
                        continue
                    for nr in right_opts:
                        if not nr or (nl & nr):
                            continue
                        if not all(v in adj[u] for u in nl for v in nr):
                            continue
                        merged_edges = _biclique_edges(nl, nr)
                        if merged_edges != edge_sets[i] | edge_sets[j]:
                            continue
                        overlap = False
                        for k in range(len(bicliques)):
                            if k == i or k == j:
                                continue
                            if merged_edges & edge_sets[k]:
                                overlap = True
                                break
                        if overlap:
                            continue
                        bicliques[i] = [list(nl), list(nr)]
                        del bicliques[j]
                        improved = True
                        found = True
                        break
                    if found:
                        break
                if improved:
                    break
            if improved:
                break
    return bicliques


def _dlx_exact_cover(cands, all_edges_set, deadline, incumbent_count):
    """Algorithm X over edge->biclique incidence. Prunes with incumbent_count."""
    edge_list = list(all_edges_set)
    edge_index = {e: i for i, e in enumerate(edge_list)}
    m = len(edge_list)
    k = len(cands)
    if k == 0:
        return None

    cand_edges = [_biclique_edges(set(l), set(r)) for l, r in cands]
    # rows: for each candidate, list of edge indices it covers
    rows = []
    for ce in cand_edges:
        rows.append([edge_index[e] for e in ce])
    # columns: for each edge, list of candidate rows covering it
    cols = [[] for _ in range(m)]
    for j, r in enumerate(rows):
        for ci in r:
            cols[ci].append(j)

    # If any edge has no candidate -> infeasible with this pool
    for ci in range(m):
        if not cols[ci]:
            return None

    best = [incumbent_count]
    best_sol = [None]
    nodes = [0]

    def solve(active_cols, active_rows, chosen):
        if time.time() > deadline:
            return
        nodes[0] += 1
        if nodes[0] % 200000 == 0 and time.time() > deadline:
            return
        if len(chosen) >= best[0]:
            return
        # pick column with fewest active rows
        best_c = -1
        best_n = 10 ** 9
        for c in active_cols:
            n_active = len(active_rows[c])
            if n_active < best_n:
                best_n = n_active
                best_c = c
            if best_n <= 1:
                break
        if best_c < 0:
            if len(chosen) < best[0]:
                best[0] = len(chosen)
                best_sol[0] = list(chosen)
            return
        if best_n == 0:
            return
        # branch on each row covering best_c
        for r in list(active_rows[best_c]):
            if len(chosen) + 1 >= best[0]:
                break
            chosen.append(r)
            # remove rows that share any column with r
            covered_cols = rows[r]
            new_active_rows = {}
            feasible = True
            for c in active_cols:
                if c in covered_cols:
                    continue
                new_rows = []
                for rr in active_rows[c]:
                    if rr == r:
                        continue
                    if any(cc in covered_cols for cc in rows[rr]):
                        continue
                    new_rows.append(rr)
                if not new_rows:
                    feasible = False
                    break
                new_active_rows[c] = new_rows
            if feasible:
                new_active_cols = [c for c in active_cols if c not in covered_cols]
                solve(new_active_cols, new_active_rows, chosen)
            chosen.pop()
            if time.time() > deadline:
                return

    active_rows_init = {c: list(cols[c]) for c in range(m)}
    active_cols_init = list(range(m))
    solve(active_cols_init, active_rows_init, [])
    return best_sol[0]


def construct():
    start = time.time()
    deadline = start + 80  # safety margin before 90s kill

    n = P['n']
    adj = _build_adjacency()
    all_edges_set = set((min(u, v), max(u, v)) for u, v in P['edges'])

    # 1. Maximal biclique pool
    max_bcs = _maximal_bicliques(adj, n, deadline)
    # Convert to (frozenset, frozenset)
    cands = list(max_bcs)

    # 2. Greedy incumbent
    incumbent = _greedy_partition(cands, all_edges_set, deadline)
    if not _validate(incumbent, adj, all_edges_set):
        incumbent = [({u}, {v}) for (u, v) in all_edges_set]
    best_result = incumbent
    best_count = len(incumbent)

    # 3. Local search polish on incumbent
    if time.time() < deadline:
        improved = _local_search(incumbent, adj, all_edges_set, deadline)
        if _validate(improved, adj, all_edges_set) and len(improved) < best_count:
            best_count = len(improved)
            best_result = improved

    # 4. Exact DLX search with incumbent bound
    if time.time() < deadline:
        exact = _dlx_exact_cover(cands, all_edges_set, deadline, best_count)
        if exact is not None:
            candidate_sol = [cands[i] for i in exact]
            if _validate(candidate_sol, adj, all_edges_set) and len(candidate_sol) < best_count:
                best_count = len(candidate_sol)
                best_result = candidate_sol
                if time.time() < deadline:
                    improved = _local_search(candidate_sol, adj, all_edges_set, deadline)
                    if _validate(improved, adj, all_edges_set) and len(improved) < best_count:
                        best_count = len(improved)
                        best_result = improved

    # Convert to required types
    result = []
    for left, right in best_result:
        result.append([list(int(x) for x in left), list(int(x) for x in right)])
    return result
# EVOLVE-BLOCK-END

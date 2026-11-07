"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _edge_key(a, b):
    return (a, b) if a < b else (b, a)


def _build_host_adj(n, edges):
    host_adj = [set() for _ in range(n)]
    for u, v in edges:
        host_adj[u].add(v)
        host_adj[v].add(u)
    return host_adj


def _edge_set_of(edges):
    s = set()
    for u, v in edges:
        s.add(_edge_key(u, v))
    return s


def _biclique_edges(L, R):
    es = set()
    for l in L:
        for r in R:
            es.add(_edge_key(l, r))
    return es


def _expand_biclique(start_v, remaining_edges, n, max_iter=100):
    """Grow a maximal biclique from start_v, maximizing product of side sizes."""
    R = set()
    for e in remaining_edges:
        a, b = e
        if a == start_v:
            R.add(b)
        elif b == start_v:
            R.add(a)
    if not R:
        return set(), set(), 0
    L = {start_v}
    rem_adj = [set() for _ in range(n)]
    for e in remaining_edges:
        a, b = e
        rem_adj[a].add(b)
        rem_adj[b].add(a)
    changed = True
    it = 0
    while changed and it < max_iter:
        changed = False
        it += 1
        # Try adding to L: candidate w must be adjacent to all of R
        best_w = None
        best_sc = -1.0
        for w in range(n):
            if w in L or w in R:
                continue
            if R.issubset(rem_adj[w]):
                # Product score: (len(L)+1) * len(R)
                sc = float(len(L) + 1) * float(len(R))
                if sc > best_sc:
                    best_sc = sc
                    best_w = w
        if best_w is not None:
            L.add(best_w)
            changed = True
        # Try adding to R: candidate w must be adjacent to all of L
        best_w = None
        best_sc = -1.0
        for w in range(n):
            if w in L or w in R:
                continue
            if L.issubset(rem_adj[w]):
                # Product score: len(L) * (len(R)+1)
                sc = float(len(L)) * float(len(R) + 1)
                if sc > best_sc:
                    best_sc = sc
                    best_w = w
        if best_w is not None:
            R.add(best_w)
            changed = True
    removed = 0
    for l in L:
        for r in R:
            t = _edge_key(l, r)
            if t in remaining_edges:
                removed += 1
    return L, R, removed


def _expand_biclique_from_pair(a, b, remaining_edges, n, max_iter=100):
    """Grow a maximal biclique from edge (a,b), maximizing product of side sizes."""
    L = {a}
    R = {b}
    rem_adj = [set() for _ in range(n)]
    for e in remaining_edges:
        u, v = e
        rem_adj[u].add(v)
        rem_adj[v].add(u)
    changed = True
    it = 0
    while changed and it < max_iter:
        changed = False
        it += 1
        best_w = None
        best_sc = -1.0
        for w in range(n):
            if w in L or w in R:
                continue
            if R.issubset(rem_adj[w]):
                sc = float(len(L) + 1) * float(len(R))
                if sc > best_sc:
                    best_sc = sc
                    best_w = w
        if best_w is not None:
            L.add(best_w)
            changed = True
        best_w = None
        best_sc = -1.0
        for w in range(n):
            if w in L or w in R:
                continue
            if L.issubset(rem_adj[w]):
                sc = float(len(L)) * float(len(R) + 1)
                if sc > best_sc:
                    best_sc = sc
                    best_w = w
        if best_w is not None:
            R.add(best_w)
            changed = True
    removed = 0
    for l in L:
        for r in R:
            t = _edge_key(l, r)
            if t in remaining_edges:
                removed += 1
    return L, R, removed


def _build_biclique_candidates(n, all_edges_set, deadline):
    """Pre-compute a pool of maximal bicliques (hyperedges) from the host graph."""
    cands = []
    seen = set()
    
    # Vertex stars
    for v in range(n):
        R = [u for u in range(n) if u != v and _edge_key(v, u) in all_edges_set]
        if R:
            L = [v]
            key = (tuple(sorted(L)), tuple(sorted(R)))
            if key not in seen:
                seen.add(key)
                cands.append((L, R))
    
    # Greedy maximal bicliques from many starts
    starts = list(range(n))
    random.seed(12345)
    random.shuffle(starts)
    for v in starts:
        if time.time() > deadline:
            break
        L, R, removed = _expand_biclique(v, all_edges_set, n)
        if removed > 0:
            key = (tuple(sorted(L)), tuple(sorted(R)))
            if key not in seen:
                seen.add(key)
                cands.append((sorted(L), sorted(R)))
    
    # Also try from high-degree edges
    deg = [0] * n
    for e in all_edges_set:
        deg[e[0]] += 1
        deg[e[1]] += 1
    sorted_edges = sorted(all_edges_set, key=lambda e: deg[e[0]] + deg[e[1]], reverse=True)
    for a, b in sorted_edges[:50]:
        if time.time() > deadline:
            break
        L, R, removed = _expand_biclique_from_pair(a, b, all_edges_set, n)
        if removed > 0:
            key = (tuple(sorted(L)), tuple(sorted(R)))
            if key not in seen:
                seen.add(key)
                cands.append((sorted(L), sorted(R)))
    
    # Deduplicate / drop subsets (keep maximal-ish)
    filtered = []
    for L, R in cands:
        cset = _biclique_edges(L, R)
        is_sub = False
        for L2, R2 in filtered:
            if cset.issubset(_biclique_edges(L2, R2)):
                is_sub = True
                break
        if not is_sub:
            filtered.append((L, R))
    return filtered


def _try_merge_bicliques(bicliques, host_adj, n):
    merged = True
    while merged:
        merged = False
        new_bicliques = []
        i = 0
        while i < len(bicliques):
            L1, R1 = set(bicliques[i][0]), set(bicliques[i][1])
            j = i + 1
            found = False
            while j < len(bicliques):
                L2, R2 = set(bicliques[j][0]), set(bicliques[j][1])
                new_L = L1 | L2
                new_R = R1 | R2
                if new_L & new_R:
                    j += 1
                    continue
                valid = True
                for l in new_L:
                    for r in new_R:
                        if r not in host_adj[l]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    new_bicliques.append([sorted(new_L), sorted(new_R)])
                    merged = True
                    found = True
                    break
                j += 1
            if not found:
                new_bicliques.append(bicliques[i])
            i += 1
        bicliques = new_bicliques
    return bicliques


def _validate_partition(bicliques, all_edges_set, n):
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        for l in left:
            for r in right:
                e = _edge_key(l, r)
                if e in covered:
                    return False
                covered.add(e)
                if e not in all_edges_set:
                    return False
    return covered == all_edges_set


def _set_cover_partition(cands, all_edges_set, n, deadline):
    """Ratio set-cover: maximize uncovered coverage, tie-break by product size."""
    uncovered = set(all_edges_set)
    chosen = []
    cand_edge_sets = []
    cand_sizes = []  # product of side sizes for tie-breaking
    for L, R in cands:
        es = _biclique_edges(L, R)
        cand_edge_sets.append(es)
        cand_sizes.append(len(L) * len(R))
    while uncovered:
        if time.time() > deadline:
            return None
        best_i = -1
        best_cov = 0
        best_size = -1
        for i, es in enumerate(cand_edge_sets):
            if not es:
                continue
            cov = len(es & uncovered)
            if cov > best_cov or (cov == best_cov and cand_sizes[i] > best_size):
                best_cov = cov
                best_size = cand_sizes[i]
                best_i = i
        if best_i < 0 or best_cov == 0:
            for e in list(uncovered):
                a, b = e
                chosen.append([[a], [b]])
            uncovered = set()
            break
        L, R = cands[best_i]
        chosen.append([sorted(L), sorted(R)])
        uncovered -= cand_edge_sets[best_i]
    return chosen


def _greedy_partition(remaining_edges, n, deadline):
    current_remaining = set(remaining_edges)
    bicliques = []
    while current_remaining:
        if time.time() > deadline:
            return None
        deg = [0] * n
        for e in current_remaining:
            deg[e[0]] += 1
            deg[e[1]] += 1
        sorted_verts = sorted(range(n), key=lambda x: deg[x], reverse=True)
        top_candidates = [v for v in sorted_verts[:8] if deg[v] > 0]
        if not top_candidates:
            break
        best_L = None
        best_R = None
        best_removed = 0
        for v in top_candidates:
            L, R, removed = _expand_biclique(v, current_remaining, n)
            # Score by product of side sizes, tie-break by removed edges
            prod = len(L) * len(R)
            if (removed > best_removed) or (removed == best_removed and prod > len(best_L or []) * len(best_R or [])):
                best_removed = removed
                best_L = L
                best_R = R
        if best_removed == 0 or best_L is None:
            e = next(iter(current_remaining))
            bicliques.append([[e[0]], [e[1]]])
            current_remaining.remove(e)
            continue
        for l in best_L:
            for r in best_R:
                t = _edge_key(l, r)
                if t in current_remaining:
                    current_remaining.remove(t)
        bicliques.append([sorted(best_L), sorted(best_R)])
    return bicliques


def _shrink_remerge(bicliques, all_edges_set, host_adj, n, deadline):
    """Iteratively try to drop a biclique by re-covering its edges with others,
    then remerge. Keeps only valid, smaller partitions."""
    current = [list(b) for b in bicliques]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best = None
        for idx in range(len(current)):
            if time.time() > deadline:
                break
            removed = current[idx]
            others = [b for k, b in enumerate(current) if k != idx]
            rem_edges = set()
            for l in removed[0]:
                for r in removed[1]:
                    rem_edges.add(_edge_key(l, r))
            coverable = True
            trial = [list(b) for b in others]
            for e in rem_edges:
                a, b = e
                placed = False
                for bi in range(len(trial)):
                    L = set(trial[bi][0])
                    R = set(trial[bi][1])
                    ok = False
                    if a not in R and b not in L:
                        nL = L | {a}
                        nR = R | {b}
                        if not (nL & nR):
                            valid = True
                            for l in nL:
                                for r in nR:
                                    if r not in host_adj[l]:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                trial[bi] = [sorted(nL), sorted(nR)]
                                placed = True
                                break
                    if b not in R and a not in L:
                        nL = L | {b}
                        nR = R | {a}
                        if not (nL & nR):
                            valid = True
                            for l in nL:
                                for r in nR:
                                    if r not in host_adj[l]:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                trial[bi] = [sorted(nL), sorted(nR)]
                                placed = True
                                break
                if not placed:
                    coverable = False
                    break
            if coverable:
                merged = _try_merge_bicliques(trial, host_adj, n)
                if _validate_partition(merged, all_edges_set, n) and len(merged) < len(current):
                    best = merged
                    improved = True
                    break
        if best is not None:
            current = best
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 75.0

    n = P['n']
    host_adj = _build_host_adj(n, P['edges'])
    all_edges_set = _edge_set_of(P['edges'])

    best_solution = None
    best_count = float('inf')

    # 1) Pre-compute biclique candidates (hyperedges)
    cands = _build_biclique_candidates(n, all_edges_set, deadline)

    # 2) Deterministic ratio set-cover
    sc = _set_cover_partition(cands, all_edges_set, n, deadline)
    if sc is not None:
        sc = _try_merge_bicliques(sc, host_adj, n)
        if _validate_partition(sc, all_edges_set, n):
            if len(sc) < best_count:
                best_count = len(sc)
                best_solution = sc

    # 3) Greedy maximal-biclique extraction (multiple randomizations)
    for attempt in range(30):
        if time.time() > deadline:
            break
        random.seed(attempt)
        gp = _greedy_partition(all_edges_set, n, deadline)
        if gp is None:
            continue
        gp = _try_merge_bicliques(gp, host_adj, n)
        if _validate_partition(gp, all_edges_set, n):
            if len(gp) < best_count:
                best_count = len(gp)
                best_solution = gp
                if best_count <= 3:
                    break

    # 4) Polish: shrink-then-remerge on the best so far
    if best_solution is not None:
        polished = _shrink_remerge(best_solution, all_edges_set, host_adj, n, deadline)
        if _validate_partition(polished, all_edges_set, n) and len(polished) < best_count:
            best_count = len(polished)
            best_solution = polished

    if best_solution is None:
        bicliques = []
        for u in range(n):
            R_u = [v for v in host_adj[u] if v > u]
            if R_u:
                bicliques.append([[u], sorted(R_u)])
        best_solution = bicliques
        best_count = len(bicliques)

    final_result = []
    for bic in best_solution:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        final_result.append([left, right])
    return final_result

# EVOLVE-BLOCK-END

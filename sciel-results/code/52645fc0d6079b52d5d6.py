"""Combined program: exact cover over rich biclique candidates + greedy finder + local search polish."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _edge_set(L, R, edges_set):
    """Return the set of graph edges covered by biclique (L, R)."""
    s = set()
    for u in L:
        for v in R:
            e = tuple(sorted((u, v)))
            if e in edges_set:
                s.add(e)
    return s


def _find_maximal_bicliques(adj, edges_set, n):
    """Find maximal bicliques using greedy expansion from each edge, with dedup."""
    maximal_bicliques = []
    seen_keys = set()
    edge_list = list(edges_set)

    for u, v in edge_list:
        L = {u}
        R = {v}
        changed = True
        while changed:
            changed = False
            for x in range(n):
                if x in L or x in R:
                    continue
                if all(x in adj[r] for r in R):
                    L.add(x)
                    changed = True
            for x in range(n):
                if x in L or x in R:
                    continue
                if all(x in adj[l] for l in L):
                    R.add(x)
                    changed = True

        if not L or not R:
            continue

        valid = True
        for a in L:
            for b in R:
                if b not in adj[a]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        key = (frozenset(L), frozenset(R))
        key2 = (frozenset(R), frozenset(L))
        if key not in seen_keys and key2 not in seen_keys:
            seen_keys.add(key)
            seen_keys.add(key2)
            maximal_bicliques.append((sorted(L), sorted(R)))

    return maximal_bicliques


def _find_edge_pair_bicliques(adj, edges_set, n, max_pairs=200):
    """Find bicliques by expanding from pairs of edges sharing a vertex.
    This captures smaller bicliques that maximal expansion might miss due to
    the greedy nature of the maximal finder."""
    bicliques = []
    seen_keys = set()
    edge_list = list(edges_set)
    pairs_checked = 0

    for i, (u, v) in enumerate(edge_list):
        if pairs_checked >= max_pairs:
            break
        for j in range(i + 1, len(edge_list)):
            if pairs_checked >= max_pairs:
                break
            pairs_checked += 1
            a, b = edge_list[j]
            # Check if the two edges share a vertex
            if v == a or v == b:
                # Shared vertex is v (or a/b). Try biclique with shared vertex on one side.
                L = {v}
                R = {u, a, b}
                # But u, a, b must all be adjacent to v (they are, by edge definition)
                # and must NOT have non-edges between them? No - within-side edges are allowed.
                # So just expand.
                changed = True
                while changed:
                    changed = False
                    for x in range(n):
                        if x in L or x in R:
                            continue
                        if all(x in adj[r] for r in R):
                            L.add(x)
                            changed = True
                    for x in range(n):
                        if x in L or x in R:
                            continue
                        if all(x in adj[l] for l in L):
                            R.add(x)
                            changed = True
                if not L or not R:
                    continue
                valid = True
                for xx in L:
                    for yy in R:
                        if yy not in adj[xx]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                key = (frozenset(L), frozenset(R))
                key2 = (frozenset(R), frozenset(L))
                if key not in seen_keys and key2 not in seen_keys:
                    seen_keys.add(key)
                    seen_keys.add(key2)
                    bicliques.append((sorted(L), sorted(R)))
            elif u == a or u == b:
                L = {u}
                R = {v, a, b}
                changed = True
                while changed:
                    changed = False
                    for x in range(n):
                        if x in L or x in R:
                            continue
                        if all(x in adj[r] for r in R):
                            L.add(x)
                            changed = True
                    for x in range(n):
                        if x in L or x in R:
                            continue
                        if all(x in adj[l] for l in L):
                            R.add(x)
                            changed = True
                if not L or not R:
                    continue
                valid = True
                for xx in L:
                    for yy in R:
                        if yy not in adj[xx]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                key = (frozenset(L), frozenset(R))
                key2 = (frozenset(R), frozenset(L))
                if key not in seen_keys and key2 not in seen_keys:
                    seen_keys.add(key)
                    seen_keys.add(key2)
                    bicliques.append((sorted(L), sorted(R)))

    return bicliques


def _find_greedy_bicliques(adj, edges_set, n, time_limit=3.0):
    """Find large bicliques using the inspiration's iterative refinement + expansion.
    Multiple anchors, each refined to a maximal biclique in the remaining-edge subgraph."""
    start_time = time.time()
    bicliques = []
    seen_keys = set()

    # Degree in the full graph
    deg = [0] * n
    for u, v in edges_set:
        deg[u] += 1
        deg[v] += 1
    candidates = [v for v in range(n) if deg[v] > 0]
    candidates.sort(key=lambda x: -deg[x])

    max_anchors = min(len(candidates), 60)

    for anchor in candidates[:max_anchors]:
        if time.time() - start_time > time_limit:
            break

        neighbors = set(adj[anchor])
        left = {anchor}
        right = set(neighbors)

        # Iterative refinement via intersection
        changed = True
        while changed:
            changed = False
            if left:
                new_right = set(adj[list(left)[0]])
                for u in list(left)[1:]:
                    new_right &= adj[u]
                new_right &= right
                if new_right != right:
                    right = new_right
                    changed = True
            if right:
                new_left = set(adj[list(right)[0]])
                for v in list(right)[1:]:
                    new_left &= adj[v]
                new_left &= left
                if new_left != left:
                    left = new_left
                    changed = True

        if not left or not right:
            continue

        # Expansion: add vertices adjacent to all of opposite side
        expanded = True
        while expanded:
            expanded = False
            for u in candidates:
                if u not in left and u not in right:
                    if all(u in adj[v] for v in right):
                        left.add(u)
                        expanded = True
            for v in candidates:
                if v not in left and v not in right:
                    if all(v in adj[u] for u in left):
                        right.add(v)
                        expanded = True

        if not left or not right:
            continue

        # Verify
        valid = True
        for u in left:
            for v in right:
                if v not in adj[u]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        key = (frozenset(left), frozenset(right))
        key2 = (frozenset(right), frozenset(left))
        if key not in seen_keys and key2 not in seen_keys:
            seen_keys.add(key)
            seen_keys.add(key2)
            bicliques.append((sorted(left), sorted(right)))

    return bicliques


def _solve_exact_cover(bicliques, edges_set, time_limit=15.0):
    """DLX-style exact cover over biclique edge sets."""
    start_time = time.time()

    biclique_edges = []
    for L, R in bicliques:
        edges = _edge_set(L, R, edges_set)
        if edges:
            biclique_edges.append(edges)
    valid_bicliques = [bicliques[i] for i in range(len(bicliques)) if i < len(biclique_edges)]
    # Re-index: we need to keep alignment. Let's rebuild properly.
    biclique_edges = []
    valid_bicliques = []
    for i, (L, R) in enumerate(bicliques):
        edges = _edge_set(L, R, edges_set)
        if edges:
            biclique_edges.append(edges)
            valid_bicliques.append((L, R))

    if not biclique_edges:
        return None

    # Map edge -> list of biclique indices
    edge_to_bicliques = {}
    for i, edges in enumerate(biclique_edges):
        for e in edges:
            if e not in edge_to_bicliques:
                edge_to_bicliques[e] = []
            edge_to_bicliques[e].append(i)

    all_edges = set(edges_set)
    # Check every edge is covered by at least one biclique
    for e in all_edges:
        if e not in edge_to_bicliques:
            return None

    best_solution = None
    best_count = float('inf')
    node_count = [0]

    def backtrack(covered, remaining, current, depth=0):
        nonlocal best_solution, best_count
        node_count[0] += 1

        if time.time() - start_time > time_limit:
            return
        if len(current) >= best_count:
            return
        if not remaining:
            if len(current) < best_count:
                best_count = len(current)
                best_solution = list(current)
            return

        # MRV: pick edge with fewest candidate bicliques
        min_edge = None
        min_cands = float('inf')
        for e in remaining:
            cands = [b for b in edge_to_bicliques.get(e, []) if b not in covered]
            if len(cands) < min_cands:
                min_cands = len(cands)
                min_edge = e
                if min_cands == 0:
                    return
        if min_cands == 0:
            return

        candidates = [b for b in edge_to_bicliques.get(min_edge, []) if b not in covered]
        candidates.sort(key=lambda b: -len(biclique_edges[b]))

        for b in candidates:
            if b in covered:
                continue
            conflict = False
            for e in biclique_edges[b]:
                if e in covered:
                    conflict = True
                    break
            if conflict:
                continue
            new_covered = covered | set(biclique_edges[b])
            new_remaining = remaining - set(biclique_edges[b])
            current.append(b)
            backtrack(new_covered, new_remaining, current, depth + 1)
            current.pop()

    backtrack(frozenset(), all_edges, [], 0)

    if best_solution is not None:
        return [valid_bicliques[i] for i in best_solution]
    return None


def _validate_bicliques(bicliques, edges_set):
    covered = set()
    for left, right in bicliques:
        if not left or not right:
            return False
        for u in left:
            for v in right:
                if u == v:
                    return False
                edge = tuple(sorted((u, v)))
                if edge not in edges_set:
                    return False
                if edge in covered:
                    return False
                covered.add(edge)
    return covered == edges_set


def _greedy_partition(adj, edges_set, n, time_limit=10.0):
    """Greedy: repeatedly find the largest biclique in remaining edges."""
    start_time = time.time()
    remaining = set(edges_set)
    bicliques = []

    while remaining:
        if time.time() - start_time > time_limit:
            break

        # Find largest biclique in remaining edges
        rem_deg = [0] * n
        for u, v in remaining:
            rem_deg[u] += 1
            rem_deg[v] += 1
        rem_cands = [v for v in range(n) if rem_deg[v] > 0]
        rem_cands.sort(key=lambda x: -rem_deg[x])

        best_left = None
        best_right = None
        best_count = 0
        rem_set = set()
        for u, v in remaining:
            rem_set.add((u, v))
            rem_set.add((v, u))

        for anchor in rem_cands[:min(len(rem_cands), 30)]:
            if time.time() - start_time > time_limit:
                break
            neighbors = set()
            for u, v in remaining:
                if u == anchor:
                    neighbors.add(v)
                elif v == anchor:
                    neighbors.add(u)
            if not neighbors:
                continue
            left = {anchor}
            right = set(neighbors)
            changed = True
            while changed:
                changed = False
                if left:
                    new_right = set(adj[list(left)[0]])
                    for u in list(left)[1:]:
                        new_right &= adj[u]
                    new_right &= right
                    if new_right != right:
                        right = new_right
                        changed = True
                if right:
                    new_left = set(adj[list(right)[0]])
                    for v in list(right)[1:]:
                        new_left &= adj[v]
                    new_left &= left
                    if new_left != left:
                        left = new_left
                        changed = True
            if not left or not right:
                continue
            # Expand
            expanded = True
            while expanded:
                expanded = False
                for u in rem_cands:
                    if u not in left and u not in right:
                        if all(u in adj[v] for v in right):
                            if any((u, v) in rem_set for v in right):
                                left.add(u)
                                expanded = True
                for v in rem_cands:
                    if v not in left and v not in right:
                        if all(v in adj[u] for u in left):
                            if any((u, v) in rem_set for u in left):
                                right.add(v)
                                expanded = True
            if not left or not right:
                continue
            cnt = 0
            for u in left:
                for v in right:
                    if (u, v) in rem_set:
                        cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_left = sorted(left)
                best_right = sorted(right)

        if best_left is None or best_right is None:
            u, v = next(iter(remaining))
            bicliques.append([[u], [v]])
            remaining.discard(tuple(sorted((u, v))))
            continue

        # Shrink to only cover remaining edges
        valid_left = [u for u in best_left if all((u, v) in rem_set or (v, u) in rem_set for v in best_right)]
        valid_right = [v for v in best_right if all((u, v) in rem_set or (v, u) in rem_set for u in valid_left)]
        if not valid_left or not valid_right:
            u, v = next(iter(remaining))
            bicliques.append([[u], [v]])
            remaining.discard(tuple(sorted((u, v))))
            continue

        for u in valid_left:
            for v in valid_right:
                remaining.discard(tuple(sorted((u, v))))
        bicliques.append([valid_left, valid_right])

    for edge in remaining:
        u, v = edge
        bicliques.append([[u], [v]])

    return bicliques


def _polish_merge(bicliques, edges_set, time_limit=5.0):
    """Try to merge pairs of bicliques if their union forms a valid biclique
    whose edge set equals the union of the two."""
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        n_b = len(bicliques)
        for i in range(n_b):
            if time.time() - start_time > time_limit:
                break
            li, ri = bicliques[i][0], bicliques[i][1]
            for j in range(i + 1, n_b):
                if time.time() - start_time > time_limit:
                    break
                lj, rj = bicliques[j][0], bicliques[j][1]
                new_left = set(li) | set(lj)
                new_right = set(ri) | set(rj)
                if new_left & new_right:
                    continue
                valid = True
                for u in new_left:
                    for v in new_right:
                        if tuple(sorted((u, v))) not in edges_set:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                cov_i = _edge_set(li, ri, edges_set)
                cov_j = _edge_set(lj, rj, edges_set)
                cov_new = _edge_set(sorted(new_left), sorted(new_right), edges_set)
                if cov_new == cov_i | cov_j:
                    bicliques[i] = [sorted(new_left), sorted(new_right)]
                    del bicliques[j]
                    improved = True
                    break
            if improved:
                break
    return bicliques


def _polish_relocate(bicliques, edges_set, time_limit=5.0):
    """Local search: try to move a vertex from one biclique to another if it
    maintains validity and reduces the total number of bicliques (by emptying
    one) or keeps the count the same but improves structure.
    Here we focus on: moving a vertex to an adjacent biclique to enable a merge,
    or simply absorbing a singleton biclique into another."""
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        n_b = len(bicliques)
        for i in range(n_b):
            if time.time() - start_time > time_limit:
                break
            li, ri = bicliques[i][0], bicliques[i][1]
            for j in range(n_b):
                if i == j or time.time() - start_time > time_limit:
                    continue
                lj, rj = bicliques[j][0], bicliques[j][1]

                # Try moving a vertex from i to j (left to left, left to right, etc.)
                for side_src in ['L', 'R']:
                    for side_dst in ['L', 'R']:
                        src_list = li if side_src == 'L' else ri
                        dst_list = lj if side_dst == 'L' else rj
                        for u in src_list:
                            if u in dst_list:
                                continue
                            # Check u is adjacent to all in dst_list
                            if not all(u in adj_global[v] for v in dst_list):
                                continue
                            # Build test
                            test = [list(x) for x in bicliques]
                            if side_src == 'L':
                                test[i][0] = [x for x in li if x != u]
                                test[i][1] = ri
                            else:
                                test[i][0] = li
                                test[i][1] = [x for x in ri if x != u]
                            if side_dst == 'L':
                                test[j][0] = lj + [u]
                                test[j][1] = rj
                            else:
                                test[j][0] = lj
                                test[j][1] = rj + [u]
                            # Remove empty sides
                            if not test[i][0] or not test[i][1]:
                                test[i] = None
                            if not test[j][0] or not test[j][1]:
                                test[j] = None
                            test = [t for t in test if t is not None]
                            if len(test) >= n_b:
                                continue  # No reduction
                            if _validate_bicliques(test, edges_set):
                                bicliques = test
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return bicliques


# Global adjacency for use in polish_relocate
adj_global = None


def construct():
    global adj_global
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    n = P['n']
    adj = _build_adjacency()
    adj_global = adj
    edges_set = set(tuple(sorted(e)) for e in P['edges'])

    # Fallback: star decomposition
    def star_partition():
        result = []
        for u in range(n):
            neighbors = sorted(adj[u])
            if neighbors:
                result.append([neighbors, [u]])
        return result

    fallback = star_partition()
    best = fallback
    best_count = len(fallback)

    # Step 1: Generate candidate bicliques from multiple sources
    candidates = []

    if time.time() < deadline - 30:
        max_bics = _find_maximal_bicliques(adj, edges_set, n)
        candidates.extend(max_bics)

    if time.time() < deadline - 25:
        pair_bics = _find_edge_pair_bicliques(adj, edges_set, n, max_pairs=300)
        # Deduplicate against existing
        existing_keys = set()
        for L, R in candidates:
            existing_keys.add((frozenset(L), frozenset(R)))
            existing_keys.add((frozenset(R), frozenset(L)))
        for L, R in pair_bics:
            k1 = (frozenset(L), frozenset(R))
            k2 = (frozenset(R), frozenset(L))
            if k1 not in existing_keys and k2 not in existing_keys:
                existing_keys.add(k1)
                existing_keys.add(k2)
                candidates.append((L, R))

    if time.time() < deadline - 20:
        greedy_bics = _find_greedy_bicliques(adj, edges_set, n, time_limit=3.0)
        existing_keys = set()
        for L, R in candidates:
            existing_keys.add((frozenset(L), frozenset(R)))
            existing_keys.add((frozenset(R), frozenset(L)))
        for L, R in greedy_bics:
            k1 = (frozenset(L), frozenset(R))
            k2 = (frozenset(R), frozenset(L))
            if k1 not in existing_keys and k2 not in existing_keys:
                existing_keys.add(k1)
                existing_keys.add(k2)
                candidates.append((L, R))

    # Step 2: Solve exact cover over candidates
    if candidates and time.time() < deadline - 15:
        solution = _solve_exact_cover(candidates, edges_set, time_limit=min(15.0, deadline - 15 - time.time()))
        if solution is not None and _validate_bicliques(solution, edges_set):
            if len(solution) < best_count:
                best = solution
                best_count = len(solution)

    # Step 3: Greedy partition as backup/improvement
    if time.time() < deadline - 10:
        greedy_sol = _greedy_partition(adj, edges_set, n, time_limit=min(8.0, deadline - 10 - time.time()))
        if _validate_bicliques(greedy_sol, edges_set):
            if len(greedy_sol) < best_count:
                best = greedy_sol
                best_count = len(greedy_sol)

    # Step 4: Polish - merge
    if time.time() < deadline - 8:
        best = _polish_merge([list(x) for x in best], edges_set, time_limit=min(5.0, deadline - 8 - time.time()))
        if _validate_bicliques(best, edges_set):
            if len(best) < best_count:
                best_count = len(best)

    # Step 5: Polish - relocate
    if time.time() < deadline - 5:
        best = _polish_relocate([list(x) for x in best], edges_set, time_limit=min(4.0, deadline - 5 - time.time()))
        if _validate_bicliques(best, edges_set):
            if len(best) < best_count:
                best_count = len(best)

    # Final validation
    if not _validate_bicliques(best, edges_set):
        best = fallback

    # Ensure types
    result = []
    for bic in best:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        result.append([left, right])

    return result

# EVOLVE-BLOCK-END

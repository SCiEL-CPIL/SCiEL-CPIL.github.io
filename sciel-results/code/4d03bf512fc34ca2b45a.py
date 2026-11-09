import time
import random
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start = time.time()
    deadline = start + 80  # 10s safety margin

    n = P['n']
    edges = P['edges']

    # Build adjacency sets and edge set
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((u, v))

    total_edges = len(edge_set)

    def make_edge(u, v):
        return (min(u, v), max(u, v))

    def expand_biclique(L, R, remaining):
        """Expand biclique to maximal size."""
        L = set(L)
        R = set(R)
        changed = True
        while changed:
            changed = False
            new_L = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                valid = True
                for r in R:
                    if make_edge(w, r) not in remaining:
                        valid = False
                        break
                if valid:
                    new_L.add(w)
            if new_L:
                L |= new_L
                changed = True

            new_R = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                valid = True
                for l in L:
                    if make_edge(w, l) not in remaining:
                        valid = False
                        break
                if valid:
                    new_R.add(w)
            if new_R:
                R |= new_R
                changed = True
        return L, R

    def is_valid_biclique(L, R, remaining):
        """Check if (L, R) is a valid biclique in the remaining edge set."""
        for l in L:
            for r in R:
                if l == r:
                    continue
                if make_edge(l, r) not in remaining:
                    return False
        return True

    def find_bicliques_by_seeds(remaining, max_left_size=2, rng=None):
        """Find bicliques using seeds of size 1 and 2 on the left side."""
        if not remaining:
            return []

        results = []
        seen = set()

        # Build remaining adjacency
        rem_adj = [set() for _ in range(n)]
        for (u, v) in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)

        vertices_with_edges = [u for u in range(n) if rem_adj[u]]
        if not vertices_with_edges:
            return []

        if rng:
            rng.shuffle(vertices_with_edges)

        # Size-1 seeds: single vertex on left
        for u in vertices_with_edges:
            R = set(rem_adj[u])
            if not R:
                continue
            L = {u}
            L, R = expand_biclique(L, R, remaining)
            if L and R and not (L & R):
                key = (frozenset(L), frozenset(R))
                if key not in seen:
                    seen.add(key)
                    results.append((L, R))

        # Size-2 seeds: pair of vertices on left
        for i, u in enumerate(vertices_with_edges):
            if time.time() > deadline - 5:
                break
            for v in vertices_with_edges[i+1:]:
                if make_edge(u, v) in remaining:
                    continue  # They'd be on the same side, need cross-edge
                # Common neighbors in remaining
                common = rem_adj[u] & rem_adj[v]
                if not common:
                    continue
                R = common
                L = {u, v}
                L, R = expand_biclique(L, R, remaining)
                if L and R and not (L & R):
                    key = (frozenset(L), frozenset(R))
                    if key not in seen:
                        seen.add(key)
                        results.append((L, R))

        return results

    def find_best_biclique(remaining, rng=None):
        """Find the best biclique using multiple strategies."""
        if not remaining:
            return None

        best_bc = None
        best_size = 0

        # Strategy A: Size-1 and size-2 seeds
        candidates = find_bicliques_by_seeds(remaining, max_left_size=2, rng=rng)
        for L, R in candidates:
            size = len(L) * len(R)
            if size > best_size:
                best_size = size
                best_bc = (L, R)

        # Strategy B: Edge seeds (try a subset for speed)
        edge_list = list(remaining)
        if rng:
            rng.shuffle(edge_list)
        else:
            edge_list.sort(key=lambda e: (e[0], e[1]))

        # Try top edges by degree
        deg = {}
        for (u, v) in remaining:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        edge_list.sort(key=lambda e: -(deg.get(e[0], 0) + deg.get(e[1], 0)))

        for (u, v) in edge_list[:min(len(edge_list), 40)]:
            L, R = expand_biclique({u}, {v}, remaining)
            size = len(L) * len(R)
            if size > best_size:
                best_size = size
                best_bc = (L, R)

        return best_bc

    def greedy_partition(remaining, use_coverage_heuristic=False, rng=None):
        """Greedy: repeatedly find largest biclique, remove its edges."""
        bicliques = []
        remaining = set(remaining)

        while remaining:
            if time.time() > deadline:
                break

            if use_coverage_heuristic and rng:
                # Try multiple random restarts for this step
                best_bc = None
                best_size = 0
                for _ in range(3):
                    bc = find_best_biclique(remaining, rng=rng)
                    if bc is not None:
                        L, R = bc
                        size = len(L) * len(R)
                        if size > best_size:
                            best_size = size
                            best_bc = bc
            else:
                best_bc = find_best_biclique(remaining, rng=rng)

            if best_bc is None:
                for (u, v) in remaining:
                    bicliques.append((frozenset([u]), frozenset([v])))
                remaining = set()
                break

            L, R = best_bc
            bicliques.append((frozenset(L), frozenset(R)))

            to_remove = set()
            for l in L:
                for r in R:
                    if l != r:
                        e = make_edge(l, r)
                        if e in remaining:
                            to_remove.add(e)
            remaining -= to_remove

        return bicliques, remaining

    def try_merge(bicliques, edge_set):
        """Try to merge pairs of bicliques."""
        n_bics = len(bicliques)
        for i in range(n_bics):
            for j in range(i + 1, n_bics):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]

                candidates = []

                # Union both sides
                newL = L1 | L2
                newR = R1 | R2
                if not (newL & newR):
                    candidates.append((newL, newR))

                # Cross merge
                newL = L1 | R2
                newR = R1 | L2
                if not (newL & newR):
                    candidates.append((newL, newR))

                # L2 subset of L1
                if L2 <= L1:
                    newR = R1 | R2
                    if not (L1 & newR):
                        candidates.append((L1, newR))

                # R2 subset of R1
                if R2 <= R1:
                    newL = L1 | L2
                    if not (newL & R1):
                        candidates.append((newL, R1))

                # L1 subset of L2
                if L1 <= L2:
                    newR = R1 | R2
                    if not (L2 & newR):
                        candidates.append((L2, newR))

                # R1 subset of R2
                if R1 <= R2:
                    newL = L1 | L2
                    if not (newL & R2):
                        candidates.append((newL, R2))

                for newL, newR in candidates:
                    if not newL or not newR:
                        continue
                    if newL & newR:
                        continue

                    valid = True
                    for l in newL:
                        for r in newR:
                            if l == r:
                                continue
                            if make_edge(l, r) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break

                    if valid:
                        result = list(bicliques)
                        result.pop(j)
                        result.pop(i)
                        result.append((frozenset(newL), frozenset(newR)))
                        return result

        return None

    def local_search_improve(bicliques, edge_set, max_iters=30):
        """Iteratively try to merge and re-optimize."""
        bicliques = list(bicliques)

        for iteration in range(max_iters):
            if time.time() > deadline:
                break

            improved = False

            # Try merging
            merged = try_merge(bicliques, edge_set)
            if merged is not None:
                bicliques = merged
                improved = True

            if not improved:
                break

        return bicliques

    def validate_result(result, edge_set):
        """Validate the result."""
        covered_edges = set()
        for bc in result:
            if not isinstance(bc, list) or len(bc) != 2:
                return False
            left, right = bc
            if not isinstance(left, list) or not isinstance(right, list):
                return False
            if len(left) == 0 or len(right) == 0:
                return False
            if set(left) & set(right):
                return False
            for l in left:
                if not isinstance(l, int) or isinstance(l, bool):
                    return False
                for r in right:
                    if not isinstance(r, int) or isinstance(r, bool):
                        return False
                    if l == r:
                        return False
                    e = make_edge(l, r)
                    if e not in edge_set:
                        return False
                    if e in covered_edges:
                        return False
                    covered_edges.add(e)
        return covered_edges == edge_set

    # Run multiple strategies
    best_result = None
    best_count = total_edges + 1

    # Strategy 1: Deterministic greedy with size-2 seeds
    bicliques, leftover = greedy_partition(edge_set, use_coverage_heuristic=False, rng=None)
    if leftover:
        for (u, v) in leftover:
            bicliques.append((frozenset([u]), frozenset([v])))
    if len(bicliques) < best_count:
        best_count = len(bicliques)
        best_result = list(bicliques)

    # Strategy 2: Randomized runs with different seeds
    for trial in range(8):
        if time.time() > deadline - 8:
            break

        seed = 100 + trial * 7
        rng_trial = random.Random(seed)
        bicliques_t, leftover_t = greedy_partition(edge_set, use_coverage_heuristic=True, rng=rng_trial)
        if leftover_t:
            for (u, v) in leftover_t:
                bicliques_t.append((frozenset([u]), frozenset([v])))

        if len(bicliques_t) < best_count:
            best_count = len(bicliques_t)
            best_result = list(bicliques_t)

    # Try to improve via merging
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        merged = try_merge(best_result, edge_set)
        if merged is not None:
            best_result = merged
            best_count = len(merged)
            improved = True

    # Local search
    if time.time() < deadline - 2:
        local_result = local_search_improve(best_result, edge_set, max_iters=20)
        if len(local_result) < best_count:
            best_count = len(local_result)
            best_result = local_result

    # Final conversion to required format
    result = []
    for L, R in best_result:
        result.append([list(L), list(R)])

    # Validate the result
    if not validate_result(result, edge_set):
        # Fallback: one biclique per edge
        result = []
        for u, v in sorted(edge_set):
            result.append([[u], [v]])

    return result

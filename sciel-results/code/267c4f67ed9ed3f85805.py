import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start = time.time()
    deadline = start + 80

    n = P['n']
    edges = P['edges']

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

    def is_biclique(L, R, remaining):
        for l in L:
            for r in R:
                if l == r:
                    continue
                if make_edge(l, r) not in remaining:
                    return False
        return True

    def expand_biclique(L, R, remaining, n):
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

    def find_max_biclique_seeded(remaining, n, seed_edges=None):
        """Find a large biclique in remaining. If seed_edges given, must include them."""
        if not remaining:
            return None

        best_biclique = None
        best_size = 0

        if seed_edges is None:
            # Try all edges as seeds
            for (u, v) in remaining:
                L, R = expand_biclique({u}, {v}, remaining, n)
                # Also try swapped
                L2, R2 = expand_biclique({v}, {u}, remaining, n)
                for (L_try, R_try) in [(L, R), (L2, R2)]:
                    if not L_try or not R_try or (L_try & R_try):
                        continue
                    size = len(L_try) * len(R_try)
                    if size > best_size:
                        best_size = size
                        best_biclique = (L_try, R_try)
        else:
            # Must include all seed edges
            # L must contain all left endpoints, R must contain all right endpoints (or swap)
            lefts = set()
            rights = set()
            for (u, v) in seed_edges:
                if u > v:
                    u, v = v, u
                lefts.add(u)
                rights.add(v)

            # Option 1: L=lefts, R=rights
            L, R = expand_biclique(lefts, rights, remaining, n)
            if L and R and not (L & R):
                if is_biclique(L, R, remaining):
                    size = len(L) * len(R)
                    if size > best_size:
                        best_size = size
                        best_biclique = (L, R)

            # Option 2: L=rights, R=lefts
            L2, R2 = expand_biclique(rights, lefts, remaining, n)
            if L2 and R2 and not (L2 & R2):
                if is_biclique(L2, R2, remaining):
                    size = len(L2) * len(R2)
                    if size > best_size:
                        best_size = size
                        best_biclique = (L2, R2)

        return best_biclique

    def greedy_partition(remaining, n, strategy='edge'):
        bicliques = []
        remaining = set(remaining)

        while remaining:
            if time.time() > deadline:
                break

            best_bc = None
            best_size = 0

            if strategy == 'edge':
                bc = find_max_biclique_seeded(remaining, n)
                if bc:
                    best_bc = bc
                    best_size = len(bc[0]) * len(bc[1])
            elif strategy == 'vertex':
                # Try seeding with each vertex's neighbors
                for u in range(n):
                    if time.time() > deadline:
                        break
                    neighbors = [v for v in adj[u] if make_edge(u, v) in remaining]
                    if not neighbors:
                        continue
                    # Seed with (u, neighbor) pairs
                    for v in neighbors[:5]:  # Limit for speed
                        L, R = expand_biclique({u}, {v}, remaining, n)
                        if L and R and not (L & R):
                            if is_biclique(L, R, remaining):
                                size = len(L) * len(R)
                                if size > best_size:
                                    best_size = size
                                    best_bc = (L, R)

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

                # Subset merges
                if L2 <= L1:
                    newR = R1 | R2
                    if not (L1 & newR):
                        candidates.append((L1, newR))
                if R2 <= R1:
                    newL = L1 | L2
                    if not (newL & R1):
                        candidates.append((newL, R1))
                if L1 <= L2:
                    newR = R1 | R2
                    if not (L2 & newR):
                        candidates.append((L2, newR))
                if R1 <= R2:
                    newL = L1 | L2
                    if not (newL & R2):
                        candidates.append((newL, R2))

                for newL, newR in candidates:
                    if not newL or not newR or (newL & newR):
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

    def local_search(bicliques, edge_set, n, deadline):
        """Try to split and recombine bicliques to reduce count."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            merged = try_merge(bicliques, edge_set)
            if merged is not None:
                bicliques = merged
                improved = True
        return bicliques

    # Run multiple strategies
    best_result = None
    best_count = total_edges + 1

    # Strategy 1: Edge-based greedy
    bicliques, leftover = greedy_partition(edge_set, n, strategy='edge')
    if leftover:
        for (u, v) in leftover:
            bicliques.append((frozenset([u]), frozenset([v])))
    if len(bicliques) < best_count:
        best_count = len(bicliques)
        best_result = list(bicliques)

    # Strategy 2: Vertex-based greedy
    bicliques2, leftover2 = greedy_partition(edge_set, n, strategy='vertex')
    if leftover2:
        for (u, v) in leftover2:
            bicliques2.append((frozenset([u]), frozenset([v])))
    if len(bicliques2) < best_count:
        best_count = len(bicliques2)
        best_result = list(bicliques2)

    # Improve via merging
    best_result = local_search(best_result, edge_set, n, deadline)
    best_count = len(best_result)

    # Try randomized restarts
    random.seed(42)
    for restart in range(10):
        if time.time() > deadline:
            break

        # Randomize the edge order for greedy
        edge_list = list(edge_set)
        random.shuffle(edge_list)

        bicliques_r = []
        remaining = set(edge_list)

        while remaining:
            if time.time() > deadline:
                break

            # Pick a random edge from remaining as seed
            seed_edge = random.choice(list(remaining))
            u, v = seed_edge

            L, R = expand_biclique({u}, {v}, remaining, n)
            if not L or not R or (L & R):
                # Fallback: single edge
                bicliques_r.append((frozenset([u]), frozenset([v])))
                remaining.discard(make_edge(u, v))
                continue

            if not is_biclique(L, R, remaining):
                bicliques_r.append((frozenset([u]), frozenset([v])))
                remaining.discard(make_edge(u, v))
                continue

            bicliques_r.append((frozenset(L), frozenset(R)))

            to_remove = set()
            for l in L:
                for r in R:
                    if l != r:
                        e = make_edge(l, r)
                        if e in remaining:
                            to_remove.add(e)
            remaining -= to_remove

        if remaining:
            for (u, v) in remaining:
                bicliques_r.append((frozenset([u]), frozenset([v])))

        # Improve via merging
        bicliques_r = local_search(bicliques_r, edge_set, n, deadline)

        if len(bicliques_r) < best_count:
            best_count = len(bicliques_r)
            best_result = list(bicliques_r)

    # Final conversion
    result = []
    for L, R in best_result:
        result.append([list(L), list(R)])

    # Validate
    covered = set()
    valid = True
    for bc in result:
        if not isinstance(bc, list) or len(bc) != 2:
            valid = False
            break
        left, right = bc
        if not isinstance(left, list) or not isinstance(right, list):
            valid = False
            break
        if not left or not right:
            valid = False
            break
        if set(left) & set(right):
            valid = False
            break
        for l in left:
            if not isinstance(l, int) or isinstance(l, bool):
                valid = False
                break
            for r in right:
                if not isinstance(r, int) or isinstance(r, bool):
                    valid = False
                    break
                if l == r:
                    valid = False
                    break
                e = make_edge(l, r)
                if e not in edge_set:
                    valid = False
                    break
                if e in covered:
                    valid = False
                    break
                covered.add(e)
        if not valid:
            break

    if not valid or covered != edge_set:
        result = []
        for u, v in edge_set:
            result.append([[u], [v]])

    return result

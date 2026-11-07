import time
import random
import math
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin for 90s limit

    n = P['n']
    edges = P['edges']

    # Build adjacency list and bitmask
    adj = [0] * n
    edge_set = set()
    edge_list = []
    for u, v in edges:
        if u > v:
            u, v = v, u
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v))
        edge_list.append((u, v))

    m = len(edge_list)

    # Helper: Expand a partial biclique (L_mask, R_mask) to maximal within a given edge set
    # The edge set is represented by rem_mask (adjacency list of remaining edges)
    def expand_biclique(L_mask, R_mask, rem_mask):
        changed = True
        while changed:
            changed = False
            # Expand L: find u not in L such that u connected to all in R (in rem_mask)
            for u in range(n):
                if (L_mask >> u) & 1:
                    continue
                if (rem_mask[u] & R_mask) == R_mask and R_mask != 0:
                    L_mask |= (1 << u)
                    changed = True
            # Expand R: find v not in R such that v connected to all in L (in rem_mask)
            for v in range(n):
                if (R_mask >> v) & 1:
                    continue
                if (rem_mask[v] & L_mask) == L_mask and L_mask != 0:
                    R_mask |= (1 << v)
                    changed = True
        return L_mask, R_mask

    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]

    def get_random_edge(rem_mask):
        candidates = [u for u in range(n) if rem_mask[u] != 0]
        if not candidates:
            return None
        u = random.choice(candidates)
        mask = rem_mask[u]
        vs = []
        mm = mask
        while mm:
            lsb = mm & -mm
            v = lsb.bit_length() - 1
            vs.append(v)
            mm ^= lsb
        if not vs:
            return None
        v = random.choice(vs)
        if u > v:
            u, v = v, u
        return (u, v)

    def get_deterministic_edge(rem_mask, edge_idx):
        count = -1
        for u in range(n):
            mask = rem_mask[u]
            mm = mask
            while mm:
                lsb = mm & -mm
                v = lsb.bit_length() - 1
                mm ^= lsb
                if u < v:
                    count += 1
                    if count == edge_idx:
                        return (u, v)
        return None

    def try_greedy(restarts=300):
        best_bicliques = None
        best_count = float('inf')

        for r in range(restarts):
            if time.time() > deadline:
                break

            # Initialize remaining masks
            rem_mask = adj[:]
            edges_left = m
            current_bicliques = []

            # Use a mix of random and deterministic strategies
            use_deterministic = (r < 20)
            edge_counter = 0

            while edges_left > 0:
                if time.time() > deadline:
                    break

                best_local = None
                best_size = 0

                # Try a few starting edges
                num_trials = 10 if use_deterministic else 6
                for trial in range(num_trials):
                    if time.time() > deadline:
                        break

                    if use_deterministic and trial < 4:
                        edge = get_deterministic_edge(rem_mask, edge_counter)
                        if edge is None:
                            break
                    else:
                        edge = get_random_edge(rem_mask)
                        if edge is None:
                            break

                    u, v = edge
                    L_mask = (1 << u)
                    R_mask = (1 << v)

                    # Expand
                    L_mask, R_mask = expand_biclique(L_mask, R_mask, rem_mask)

                    size = L_mask.bit_count() * R_mask.bit_count()
                    if size > best_size:
                        best_size = size
                        best_local = (L_mask, R_mask)

                if not best_local:
                    edge = get_random_edge(rem_mask)
                    if edge is None:
                        break
                    u, v = edge
                    best_local = ((1 << u), (1 << v))
                    best_size = 1

                L_mask, R_mask = best_local
                current_bicliques.append([mask_to_list(L_mask), mask_to_list(R_mask)])

                # Update rem_mask: remove all edges between L and R
                l_list = mask_to_list(L_mask)
                r_list = mask_to_list(R_mask)
                for u in l_list:
                    rem_mask[u] &= ~R_mask
                for v in r_list:
                    rem_mask[v] &= ~L_mask

                edges_left -= best_size
                edge_counter += 1

                if edges_left < 0:
                    break

            if len(current_bicliques) < best_count:
                best_count = len(current_bicliques)
                best_bicliques = current_bicliques
                if best_count == 1:
                    break

        return best_bicliques, best_count

    # Phase 1: Greedy with many restarts
    best_bicliques, best_count = try_greedy(restarts=500)

    if best_bicliques is None:
        # Fallback: one biclique per vertex
        result = []
        for u in range(n):
            right_side = [v for u2, v in edges if u2 == u]
            if right_side:
                result.append([[u], right_side])
        return result

    # Phase 2: Try to improve by local search / merging
    def try_merge_and_improve():
        nonlocal best_bicliques, best_count

        if best_count <= 1:
            return

        improved = True
        while improved and time.time() < deadline:
            improved = False
            current = [b[:] for b in best_bicliques]
            current_count = len(current)

            # Try removing redundant bicliques
            for i in range(len(current)):
                if time.time() > deadline:
                    return

                L_i = current[i][0]
                R_i = current[i][1]

                cross_edges_i = set()
                for u in L_i:
                    for v in R_i:
                        if u < v:
                            cross_edges_i.add((u, v))
                        else:
                            cross_edges_i.add((v, u))

                all_covered = True
                for (u, v) in cross_edges_i:
                    covered = False
                    for j in range(len(current)):
                        if j == i:
                            continue
                        L_j = current[j][0]
                        R_j = current[j][1]
                        if (u in L_j and v in R_j) or (u in R_j and v in L_j):
                            covered = True
                            break
                    if not covered:
                        all_covered = False
                        break

                if all_covered:
                    current.pop(i)
                    improved = True
                    break

            if improved:
                best_bicliques = current
                best_count = len(current)
                continue

            # Try merging two bicliques
            for i in range(len(current)):
                if time.time() > deadline:
                    return
                for j in range(i + 1, len(current)):
                    if time.time() > deadline:
                        return

                    L_i, R_i = current[i]
                    L_j, R_j = current[j]

                    # Try to merge: L_new = L_i ∪ L_j, R_new = R_i ∪ R_j
                    # Check if all cross-edges in L_new × R_new are covered by the union
                    # Actually, we need to check if the union of cross-edges equals the union of the two bicliques' cross-edges
                    # and if L_new × R_new is a valid biclique (all cross-edges are graph edges)

                    L_new = set(L_i) | set(L_j)
                    R_new = set(R_i) | set(R_j)

                    # Check if L_new × R_new is a biclique
                    valid = True
                    for u in L_new:
                        for v in R_new:
                            if u == v:
                                continue
                            if (u, v) not in edge_set and (v, u) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break

                    if not valid:
                        continue

                    # Check if the union of cross-edges covers exactly the union of the two bicliques' cross-edges
                    cross_i = set()
                    for u in L_i:
                        for v in R_i:
                            if u < v:
                                cross_i.add((u, v))
                            else:
                                cross_i.add((v, u))

                    cross_j = set()
                    for u in L_j:
                        for v in R_j:
                            if u < v:
                                cross_j.add((u, v))
                            else:
                                cross_j.add((v, u))

                    cross_new = set()
                    for u in L_new:
                        for v in R_new:
                            if u < v:
                                cross_new.add((u, v))
                            else:
                                cross_new.add((v, u))

                    if cross_new == cross_i | cross_j:
                        # Merge is valid
                        new_current = current[:i] + current[i+1:j] + current[j+1:]
                        new_current.append([list(L_new), list(R_new)])
                        if len(new_current) < best_count:
                            best_bicliques = new_current
                            best_count = len(new_current)
                            improved = True
                            break
                if improved:
                    break

    try_merge_and_improve()

    # Phase 3: Try a few more greedy restarts with different random seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        random.seed(seed)
        b, c = try_greedy(restarts=10)
        if b is not None and c < best_count:
            best_bicliques = b
            best_count = c

    # Return best found
    result = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])

    return result

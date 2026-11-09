import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start = time.time()
    deadline = start + 78  # Safety margin before 90s kill

    n = P['n']
    edges = P['edges']

    # Build global adjacency as bitmasks
    adj_mask = [0] * n
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edge_set.add((u, v))

    total_edges = len(edge_set)
    all_mask = (1 << n) - 1

    def popcount(x):
        return bin(x).count('1')

    def bits_to_list(x):
        lst = []
        i = 0
        while x:
            if x & 1:
                lst.append(i)
            x >>= 1
            i += 1
        return lst

    def expand_biclique(L, R, adj):
        """Expand biclique to maximal using bitmask operations."""
        changed = True
        while changed:
            changed = False
            # Expand L: vertices connected to ALL of R
            if R:
                common = all_mask
                r = R
                while r:
                    low = r & (-r)
                    idx = low.bit_length() - 1
                    common &= adj[idx]
                    r &= r - 1
                new_L = common & ~(L | R)
                if new_L:
                    L |= new_L
                    changed = True
            # Expand R: vertices connected to ALL of L
            if L:
                common = all_mask
                l = L
                while l:
                    low = l & (-l)
                    idx = low.bit_length() - 1
                    common &= adj[idx]
                    l &= l - 1
                new_R = common & ~(L | R)
                if new_R:
                    R |= new_R
                    changed = True
        return L, R

    def find_max_biclique_from_edge(u, v, rem_adj):
        """Find maximal biclique containing edge (u,v) using both orientations."""
        best_L = 0
        best_R = 0
        best_size = 0

        # Orientation 1: u in L, v in R
        L1 = 1 << u
        R1 = 1 << v
        L1, R1 = expand_biclique(L1, R1, rem_adj)
        if L1 and R1 and not (L1 & R1):
            size1 = popcount(L1) * popcount(R1)
            if size1 > best_size:
                best_size = size1
                best_L = L1
                best_R = R1

        # Orientation 2: v in L, u in R
        L2 = 1 << v
        R2 = 1 << u
        L2, R2 = expand_biclique(L2, R2, rem_adj)
        if L2 and R2 and not (L2 & R2):
            size2 = popcount(L2) * popcount(R2)
            if size2 > best_size:
                best_size = size2
                best_L = L2
                best_R = R2

        if best_size == 0:
            return None
        return (best_L, best_R)

    def find_max_biclique_vertex(rem_adj):
        """Find largest biclique using vertex-seeded expansion."""
        best_L = 0
        best_R = 0
        best_size = 0

        for u in range(n):
            R_cand = rem_adj[u]
            if R_cand == 0:
                continue

            L = 1 << u
            R = R_cand
            L, R = expand_biclique(L, R, rem_adj)

            if not L or not R or (L & R):
                continue

            size = popcount(L) * popcount(R)
            if size > best_size:
                best_size = size
                best_L = L
                best_R = R

        if best_size == 0:
            return None
        return (best_L, best_R)

    def greedy_partition(edge_set, rng, strategy='edge'):
        """Greedy: repeatedly find largest biclique in remaining edges."""
        bicliques = []
        remaining = set(edge_set)

        while remaining:
            if time.time() > deadline:
                break

            # Build remaining adjacency
            rem_adj = [0] * n
            for (u, v) in remaining:
                rem_adj[u] |= (1 << v)
                rem_adj[v] |= (1 << u)

            if strategy == 'edge':
                # Try each edge, pick the one giving largest biclique
                edges_list = list(remaining)
                rng.shuffle(edges_list)
                best_bc = None
                best_size = 0
                for (u, v) in edges_list:
                    bc = find_max_biclique_from_edge(u, v, rem_adj)
                    if bc:
                        L, R = bc
                        sz = popcount(L) * popcount(R)
                        if sz > best_size:
                            best_size = sz
                            best_bc = bc
                if best_bc is None:
                    for (u, v) in remaining:
                        bicliques.append((1 << u, 1 << v))
                    remaining = set()
                    break
                L, R = best_bc
            else:
                bc = find_max_biclique_vertex(rem_adj)
                if bc is None:
                    for (u, v) in remaining:
                        bicliques.append((1 << u, 1 << v))
                    remaining = set()
                    break
                L, R = bc

            bicliques.append((L, R))

            # Remove all edges between L and R
            to_remove = set()
            l = L
            while l:
                low = l & (-l)
                u = low.bit_length() - 1
                r = R
                while r:
                    low_r = r & (-r)
                    v = low_r.bit_length() - 1
                    if u != v:
                        e = (min(u, v), max(u, v))
                        if e in remaining:
                            to_remove.add(e)
                    r &= r - 1
                l &= l - 1
            remaining -= to_remove

        return bicliques

    def try_merge_all(bicliques, edge_set, adj_mask):
        """Repeatedly try to merge pairs of bicliques until no more merges possible."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n_bics = len(bicliques)
            for i in range(n_bics):
                if time.time() > deadline:
                    return bicliques
                for j in range(i + 1, n_bics):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]

                    candidates = []

                    # Union both sides
                    newL = L1 | L2
                    newR = R1 | R2
                    if not (newL & newR) and newL and newR:
                        candidates.append((newL, newR))

                    # Cross merge
                    newL = L1 | R2
                    newR = R1 | L2
                    if not (newL & newR) and newL and newR:
                        candidates.append((newL, newR))

                    for newL, newR in candidates:
                        # Verify all cross edges exist in original graph
                        valid = True
                        l = newL
                        while l and valid:
                            low = l & (-l)
                            u = low.bit_length() - 1
                            r = newR
                            while r and valid:
                                low_r = r & (-r)
                                v = low_r.bit_length() - 1
                                if u != v:
                                    if not (adj_mask[u] & (1 << v)):
                                        valid = False
                                r &= r - 1
                            l &= l - 1

                        if valid:
                            result = list(bicliques)
                            result.pop(j)
                            result.pop(i)
                            result.append((newL, newR))
                            return result

        return bicliques

    def local_search(bicliques, edge_set, adj_mask, rng, max_iters=200):
        """Randomized local search: try to reduce biclique count by reassigning edges."""
        if time.time() > deadline:
            return bicliques

        current = list(bicliques)
        current_count = len(current)
        best = list(current)
        best_count = current_count

        for iteration in range(max_iters):
            if time.time() > deadline:
                break

            if len(current) <= 1:
                break

            # Randomly pick two bicliques to try combining/splitting
            if len(current) < 2:
                break

            i, j = rng.sample(range(len(current)), 2)
            L1, R1 = current[i]
            L2, R2 = current[j]

            # Try various combinations
            candidates = []

            # Union
            newL = L1 | L2
            newR = R1 | R2
            if not (newL & newR) and newL and newR:
                candidates.append((newL, newR))

            # Cross
            newL = L1 | R2
            newR = R1 | L2
            if not (newL & newR) and newL and newR:
                candidates.append((newL, newR))

            # Try swapping some vertices between bicliques
            # Move a vertex from L1 to L2 (if valid)
            l1_list = bits_to_list(L1)
            r1_list = bits_to_list(R1)
            l2_list = bits_to_list(L2)
            r2_list = bits_to_list(R2)

            if l1_list and l2_list:
                v = rng.choice(l1_list)
                newL1 = L1 & ~(1 << v)
                newL2 = L2 | (1 << v)
                if newL1 and newL2 and not (newL1 & R1) and not (newL2 & R2):
                    # Check all cross edges
                    valid1 = all((adj_mask[a] & (1 << b)) for a in bits_to_list(newL1) for b in r1_list)
                    valid2 = all((adj_mask[a] & (1 << b)) for a in [v] for b in r2_list)
                    if valid1 and valid2:
                        candidates.append(('swap', newL1, R1, newL2, R2))

            if not candidates:
                continue

            # Try the first valid candidate
            applied = False
            for cand in candidates:
                if isinstance(cand, tuple) and len(cand) == 2:
                    newL, newR = cand
                    # Verify
                    valid = True
                    l = newL
                    while l and valid:
                        low = l & (-l)
                        u = low.bit_length() - 1
                        r = newR
                        while r and valid:
                            low_r = r & (-r)
                            v = low_r.bit_length() - 1
                            if u != v:
                                if not (adj_mask[u] & (1 << v)):
                                    valid = False
                            r &= r - 1
                        l &= l - 1
                    if valid:
                        new_current = list(current)
                        new_current.pop(j)
                        new_current.pop(i)
                        new_current.append((newL, newR))
                        if len(new_current) < len(current):
                            current = new_current
                            if len(current) < best_count:
                                best_count = len(current)
                                best = list(current)
                            applied = True
                            break
                elif isinstance(cand, tuple) and len(cand) == 5 and cand[0] == 'swap':
                    _, newL1, newR1, newL2, newR2 = cand
                    new_current = list(current)
                    new_current[i] = (newL1, newR1)
                    new_current[j] = (newL2, newR2)
                    # This doesn't reduce count, but might enable future merges
                    current = new_current
                    applied = True
                    break

            if not applied and iteration > 50:
                # Reset to best and try different strategy
                current = list(best)

        return best

    # Run multiple greedy strategies
    best_result = None
    best_count = total_edges + 1

    rng = random.Random(42)

    strategies = ['edge', 'vertex']
    for restart in range(60):
        if time.time() > deadline:
            break

        strat = strategies[restart % len(strategies)]
        bicliques = greedy_partition(edge_set, rng, strategy=strat)
        bicliques = try_merge_all(bicliques, edge_set, adj_mask)

        # Local search refinement
        bicliques = local_search(bicliques, edge_set, adj_mask, rng, max_iters=100)

        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_result = bicliques

        if best_count <= 8:
            break

    # Final merge pass on best result
    if best_result is not None:
        best_result = try_merge_all(best_result, edge_set, adj_mask)
        best_result = local_search(best_result, edge_set, adj_mask, rng, max_iters=150)

    # Convert to required format
    result = []
    for L, R in best_result:
        result.append([bits_to_list(L), bits_to_list(R)])

    # Validate
    covered_edges = set()
    valid = True
    for bc in result:
        if not isinstance(bc, list) or len(bc) != 2:
            valid = False
            break
        left, right = bc
        if not isinstance(left, list) or not isinstance(right, list):
            valid = False
            break
        if len(left) == 0 or len(right) == 0:
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
                e = (min(l, r), max(l, r))
                if e not in edge_set:
                    valid = False
                    break
                if e in covered_edges:
                    valid = False
                    break
                covered_edges.add(e)
        if not valid:
            break

    if not valid or covered_edges != edge_set:
        # Fallback: one biclique per vertex
        result = []
        for u in range(n):
            neighbors = [v for v in range(n) if adj_mask[u] & (1 << v)]
            if neighbors:
                result.append([[u], neighbors])

    return result

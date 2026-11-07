import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start_time = time.time()
    deadline = start_time + 84.0

    n = P['n']
    edges = P['edges']
    edge_set = set((min(a, b), max(a, b)) for a, b in edges)

    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Seed: one biclique per vertex u covering edges (u,v) where u < v in canonical form
    seed_bicliques = []
    for u in range(n):
        right_side = [v for (a, b) in edge_set if a == u]
        if right_side:
            seed_bicliques.append([list(set([u])), sorted(right_side)])
    best_count = len(seed_bicliques)
    best_bicliques = seed_bicliques

    def bits(mask):
        out = []
        m = mask
        while m:
            lsb = m & -m
            out.append(lsb.bit_length() - 1)
            m ^= lsb
        return out

    def expand_biclique(u, v, rem_adj, order):
        L = {u}
        R = {v}
        # Iterate to fixpoint with randomized order
        for _ in range(40):
            changed = False
            # Try to add to L
            mask_R = 0
            for rv in R:
                mask_R |= (1 << rv)
            for w in order:
                if w in L:
                    continue
                if (rem_adj[w] & mask_R) == mask_R and mask_R != 0:
                    L.add(w)
                    changed = True
            # Try to add to R
            mask_L = 0
            for lu in L:
                mask_L |= (1 << lu)
            for w in order:
                if w in R:
                    continue
                ok = True
                for lu in L:
                    if not (rem_adj[lu] & (1 << w)):
                        ok = False
                        break
                if ok and mask_L != 0:
                    R.add(w)
                    changed = True
            if not changed:
                break
        return L, R

    def greedy_partition(seed_order=None):
        rem_adj = adj[:]
        bicliques = []
        # Build list of remaining edges as set
        rem_edges = set(edge_set)
        while rem_edges:
            if time.time() > deadline:
                return None
            # Pick an edge
            if seed_order is not None:
                # Find first edge in seed_order that is in rem_edges
                chosen = None
                for (u, v) in seed_order:
                    if (u, v) in rem_edges:
                        chosen = (u, v)
                        break
                if chosen is None:
                    # fallback to arbitrary
                    chosen = next(iter(rem_edges))
            else:
                chosen = random.choice(list(rem_edges))
            u, v = chosen
            order = list(range(n))
            random.shuffle(order)
            L, R = expand_biclique(u, v, rem_adj, order)
            bicliques.append([sorted(L), sorted(R)])
            # Remove edges covered
            mask_R = 0
            for rv in R:
                mask_R |= (1 << rv)
            for lu in L:
                rem_adj[lu] &= ~mask_R
            mask_L = 0
            for lu in L:
                mask_L |= (1 << lu)
            for rv in R:
                rem_adj[rv] &= ~mask_L
            # Remove from rem_edges
            new_edges = set()
            for (a, b) in rem_edges:
                if (a in L and b in R) or (a in R and b in L):
                    continue
                new_edges.add((a, b))
            rem_edges = new_edges
        return bicliques

    def try_merge(bicliques):
        # Try to merge bicliques with identical L or identical R
        improved = True
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(bicliques):
                L1 = frozenset(bicliques[i][0])
                R1 = frozenset(bicliques[i][1])
                j = i + 1
                while j < len(bicliques):
                    L2 = frozenset(bicliques[j][0])
                    R2 = frozenset(bicliques[j][1])
                    if L1 == L2:
                        # Can merge: new R = R1 | R2? No, need R1 ∩ R2 empty? 
                        # Actually, if L1 == L2, then biclique i covers L1 x R1 and biclique j covers L1 x R2.
                        # If R1 and R2 are disjoint, we can merge to L1 x (R1 ∪ R2).
                        if not (R1 & R2):
                            new_R = sorted(set(R1) | set(R2))
                            bicliques[i][1] = new_R
                            bicliques.pop(j)
                            changed = True
                            improved = True
                            break
                    elif R1 == R2:
                        if not (L1 & L2):
                            new_L = sorted(set(L1) | set(L2))
                            bicliques[i][0] = new_L
                            bicliques.pop(j)
                            changed = True
                            improved = True
                            break
                    j += 1
                if changed:
                    break
                i += 1
        return bicliques if improved else bicliques

    def local_search(bicliques):
        # Try to remove a biclique if its edges are covered by others?
        # Or try to shift vertices between bicliques to enable merges.
        # For simplicity, just try merges iteratively.
        return try_merge(bicliques)

    # Run greedy with different restarts
    # First, try deterministic order based on degree
    degrees = [bin(adj[u]).count('1') for u in range(n)]
    # Sort edges by (degree[u]+degree[v]) descending for deterministic greedy
    edges_sorted = sorted(edge_set, key=lambda e: degrees[e[0]] + degrees[e[1]], reverse=True)

    for r in range(300):
        if time.time() > deadline:
            break
        if r == 0:
            # Deterministic greedy
            bicliques = greedy_partition(edges_sorted)
        elif r == 1:
            # Another deterministic: reverse
            bicliques = greedy_partition(list(reversed(edges_sorted)))
        else:
            bicliques = greedy_partition()
        if bicliques is None:
            continue
        bicliques = local_search(bicliques)
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
            if best_count == 1:
                break

    # Fallback
    if not best_bicliques or len(best_bicliques) >= best_count:
        best_bicliques = seed_bicliques

    # Convert to required format
    result = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        if len(left) > 0 and len(right) > 0:
            result.append([left, right])

    return result

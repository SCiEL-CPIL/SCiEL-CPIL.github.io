import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # ---- Precompute Q3 bipartite structure ----
    # Q3 bipartition: even parity {0,3,5,6} vs odd parity {1,2,4,7}
    # For each of 24 permutations of B, the 4 A-vertices each need 3 specific B-neighbors
    perms24 = list(itertools.permutations(range(4)))
    req_A = []
    for p in perms24:
        req = [
            (p[0], p[1], p[3]),
            (p[0], p[1], p[2]),
            (p[0], p[2], p[3]),
            (p[1], p[2], p[3])
        ]
        req_A.append(req)

    idx4_of_8 = list(itertools.combinations(range(8), 4))

    # ---- Precompute all 8-vertex subsets and their Q3-check data ----
    # For each 8-set, store a list of (A_mask0, A_mask1, A_mask2, A_mask3, B_mask0, B_mask1, B_mask2, B_mask3)
    # where A_maski is a bitmask of the 3 B-vertices that A[i] must connect to,
    # and B_maski is a bitmask of the 3 A-vertices that B[i] must connect to.
    # We precompute these for BOTH orientations (A=even, A=odd) for all 24 perms.
    #
    # Actually, for efficiency we precompute per 8-set:
    #   checks[set] = list of (am0, am1, am2, am3, bm0, bm1, bm2, bm3)
    # where am_i = bitmask over B of required neighbors of A[i]
    # and bm_i = bitmask over A of required neighbors of B[i]
    # A 8-set contains Q3 iff for some check tuple:
    #   (adj[A[0]] & am0) == am0 and ... for all i
    # OR
    #   (adj[B[0]] & bm0) == bm0 and ... for all i

    all_8sets = list(itertools.combinations(range(16), 8))

    # Precompute the check data for each 8-set
    # checks_by_set[set_index] = list of (A_tuple, B_tuple, am0,am1,am2,am3, bm0,bm1,bm2,bm3)
    # But storing A_tuple and B_tuple is needed to index adj_mask.
    # Alternative: store (A_indices_in_v8, B_indices_in_v8, am_tuple, bm_tuple)
    # and at check time, extract actual vertices.

    # For maximum speed, precompute for each 8-set the full list of
    # (vA0, vA1, vA2, vA3, vB0, vB1, vB2, vB3, am0, am1, am2, am3, bm0, bm1, bm2, bm3)
    # This is a lot of data but makes checking very fast.

    # Actually, let's be smarter. For each 8-set, there are C(8,4)=70 partitions.
    # For each partition, 24 perms × 2 orientations = 48 checks.
    # Total per 8-set: 70 × 48 = 3360 checks.
    # Total: 12870 × 3360 ≈ 43M checks. Too many to precompute fully.

    # Better: precompute per 8-set the partition structure (A_indices, B_indices)
    # and at check time, iterate perms. But use bitmask operations.

    # Let me precompute per 8-set: list of (A_tuple, B_tuple) for all 70 partitions
    set_partitions = {}
    for v8 in all_8sets:
        v8_list = list(v8)
        parts = []
        for (i, j, k, l) in idx4_of_8:
            A = tuple(v8_list[idx] for idx in (i, j, k, l))
            B = tuple(v8_list[idx] for idx in range(8) if idx not in (i, j, k, l))
            parts.append((A, B))
        set_partitions[v8] = parts

    # Precompute per 8-set, per partition, the bitmask patterns
    # For a given (A, B) partition, for each perm p of [0,1,2,3]:
    #   am[i] = bitmask of B-indices that A[i] must connect to
    #   bm[i] = bitmask of A-indices that B[i] must connect to
    # Precompute these as tuples of 4 ints each.
    #
    # am for A (even side): A[0] needs B[p[0]],B[p[1]],B[p[3]]
    # am for B (odd side): B[0] needs A[p[0]],A[p[1]],A[p[3]]
    # etc. (same pattern)

    # Precompute for each perm: (am0,am1,am2,am3) and (bm0,bm1,bm2,bm3)
    # where am_i is a 4-bit mask over B positions, bm_i is a 4-bit mask over A positions
    perm_masks = []
    for p in perms24:
        # A-side masks (which B positions each A vertex needs)
        am0 = (1 << p[0]) | (1 << p[1]) | (1 << p[3])
        am1 = (1 << p[0]) | (1 << p[1]) | (1 << p[2])
        am2 = (1 << p[0]) | (1 << p[2]) | (1 << p[3])
        am3 = (1 << p[1]) | (1 << p[2]) | (1 << p[3])
        # B-side masks (which A positions each B vertex needs) - same pattern
        bm0 = (1 << p[0]) | (1 << p[1]) | (1 << p[3])
        bm1 = (1 << p[0]) | (1 << p[1]) | (1 << p[2])
        bm2 = (1 << p[0]) | (1 << p[2]) | (1 << p[3])
        bm3 = (1 << p[1]) | (1 << p[2]) | (1 << p[3])
        perm_masks.append((am0, am1, am2, am3, bm0, bm1, bm2, bm3))

    # ---- Fast Q3 check for an 8-vertex set ----
    def contains_q3(v8, adj_mask):
        """Check if induced subgraph on v8 contains Q3."""
        for (A, B) in set_partitions[v8]:
            # For each permutation, check both orientations
            for (am0, am1, am2, am3, bm0, bm1, bm2, bm3) in perm_masks:
                # Orientation 1: A is even, B is odd
                # Check A[0] connects to required B vertices
                m = adj_mask[A[0]]
                # Extract bits of m at positions B[0],B[1],B[2],B[3]
                # We need: (m & (1<<B[0])) if am0 has bit 0, etc.
                # Faster: build the actual vertex mask
                # am0 is a 4-bit mask over B indices. We need to check if
                # adj_mask[A[0]] contains all B vertices selected by am0.
                # Build the actual mask:
                actual_mask = 0
                if am0 & 1: actual_mask |= (1 << B[0])
                if am0 & 2: actual_mask |= (1 << B[1])
                if am0 & 4: actual_mask |= (1 << B[2])
                if am0 & 8: actual_mask |= (1 << B[3])
                if (m & actual_mask) != actual_mask:
                    continue
                # Check A[1]
                m = adj_mask[A[1]]
                actual_mask = 0
                if am1 & 1: actual_mask |= (1 << B[0])
                if am1 & 2: actual_mask |= (1 << B[1])
                if am1 & 4: actual_mask |= (1 << B[2])
                if am1 & 8: actual_mask |= (1 << B[3])
                if (m & actual_mask) != actual_mask:
                    continue
                # Check A[2]
                m = adj_mask[A[2]]
                actual_mask = 0
                if am2 & 1: actual_mask |= (1 << B[0])
                if am2 & 2: actual_mask |= (1 << B[1])
                if am2 & 4: actual_mask |= (1 << B[2])
                if am2 & 8: actual_mask |= (1 << B[3])
                if (m & actual_mask) != actual_mask:
                    continue
                # Check A[3]
                m = adj_mask[A[3]]
                actual_mask = 0
                if am3 & 1: actual_mask |= (1 << B[0])
                if am3 & 2: actual_mask |= (1 << B[1])
                if am3 & 4: actual_mask |= (1 << B[2])
                if am3 & 8: actual_mask |= (1 << B[3])
                if (m & actual_mask) != actual_mask:
                    continue
                # All A vertices satisfied -> Q3 found
                return True

                # Orientation 2: B is even, A is odd (swap roles)
                # Check B[0] connects to required A vertices
                m = adj_mask[B[0]]
                actual_mask = 0
                if bm0 & 1: actual_mask |= (1 << A[0])
                if bm0 & 2: actual_mask |= (1 << A[1])
                if bm0 & 4: actual_mask |= (1 << A[2])
                if bm0 & 8: actual_mask |= (1 << A[3])
                if (m & actual_mask) != actual_mask:
                    continue
                m = adj_mask[B[1]]
                actual_mask = 0
                if bm1 & 1: actual_mask |= (1 << A[0])
                if bm1 & 2: actual_mask |= (1 << A[1])
                if bm1 & 4: actual_mask |= (1 << A[2])
                if bm1 & 8: actual_mask |= (1 << A[3])
                if (m & actual_mask) != actual_mask:
                    continue
                m = adj_mask[B[2]]
                actual_mask = 0
                if bm2 & 1: actual_mask |= (1 << A[0])
                if bm2 & 2: actual_mask |= (1 << A[1])
                if bm2 & 4: actual_mask |= (1 << A[2])
                if bm2 & 8: actual_mask |= (1 << A[3])
                if (m & actual_mask) != actual_mask:
                    continue
                m = adj_mask[B[3]]
                actual_mask = 0
                if bm3 & 1: actual_mask |= (1 << A[0])
                if bm3 & 2: actual_mask |= (1 << A[1])
                if bm3 & 4: actual_mask |= (1 << A[2])
                if bm3 & 8: actual_mask |= (1 << A[3])
                if (m & actual_mask) != actual_mask:
                    continue
                return True

        return False

    # ---- Precompute 8-sets containing each edge ----
    pair_to_8sets = {}
    for u in range(16):
        for v in range(u + 1, 16):
            others = [x for x in range(16) if x != u and x != v]
            pair_to_8sets[(u, v)] = [tuple((u, v) + c) for c in itertools.combinations(others, 6)]

    all_pairs = list(itertools.combinations(range(16), 2))
    all_pairs_set = set(all_pairs)

    # ---- Edge add/remove helpers ----
    def try_add_edge(edges, adj_mask, u, v):
        """Try adding edge (u,v). Return True if safe."""
        if (u, v) in edges:
            return True
        edges.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

        for v8 in pair_to_8sets[(u, v)]:
            if contains_q3(v8, adj_mask):
                edges.remove((u, v))
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                return False
        return True

    def remove_edge(edges, adj_mask, u, v):
        edges.discard((u, v))
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

    # ---- Initial graph: K_{3,13} ----
    edges = set()
    for u in range(3):
        for v in range(3, 16):
            edges.add((u, v))

    adj_mask = [0] * 16
    for (u, v) in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    best_edges = set(edges)
    best_count = len(edges)

    # ---- Phase 1: Greedy addition with multiple passes ----
    for pass_num in range(50):
        if time.time() > deadline - 60:
            break
        non_edges = [e for e in all_pairs if e not in edges]
        # Sort by combined degree, descending
        non_edges.sort(key=lambda e: bin(adj_mask[e[0]]).count('1') + bin(adj_mask[e[1]]).count('1'), reverse=True)
        changed = False
        for e in non_edges:
            if time.time() > deadline - 60:
                break
            u, v = e
            if e in edges:
                continue
            if try_add_edge(edges, adj_mask, u, v):
                changed = True
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = set(edges)
        if not changed:
            break

    # ---- Phase 2: Local search with edge swaps and randomized restarts ----
    random.seed(42)
    no_improve = 0
    max_no_improve = 200

    while time.time() < deadline - 20 and no_improve < max_no_improve:
        move_type = random.random()
        improved = False

        if move_type < 0.4:
            # Add a random non-edge (weighted by degree)
            non_edges_now = [e for e in all_pairs if e not in edges]
            if not non_edges_now:
                no_improve += 1
                continue
            weights = [bin(adj_mask[e[0]]).count('1') + bin(adj_mask[e[1]]).count('1') + 1 for e in non_edges_now]
            e = random.choices(non_edges_now, weights=weights, k=1)[0]
            if try_add_edge(edges, adj_mask, e[0], e[1]):
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = set(edges)
                    improved = True
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

        elif move_type < 0.7:
            # Remove a random edge, then try to add a different one
            if not edges:
                no_improve += 1
                continue
            e_rm = random.choice(list(edges))
            u1, v1 = e_rm
            remove_edge(edges, adj_mask, u1, v1)
            non_edges_now = [x for x in all_pairs if x not in edges]
            if non_edges_now:
                weights = [bin(adj_mask[x[0]]).count('1') + bin(adj_mask[x[1]]).count('1') + 1 for x in non_edges_now]
                added = False
                for _ in range(10):
                    e_add = random.choices(non_edges_now, weights=weights, k=1)[0]
                    if try_add_edge(edges, adj_mask, e_add[0], e_add[1]):
                        added = True
                        break
                if not added:
                    # Re-add the removed edge
                    edges.add((u1, v1))
                    adj_mask[u1] |= (1 << v1)
                    adj_mask[v1] |= (1 << u1)
            else:
                edges.add((u1, v1))
                adj_mask[u1] |= (1 << v1)
                adj_mask[v1] |= (1 << u1)

            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = set(edges)
                improved = True
                no_improve = 0
            else:
                no_improve += 1

        else:
            # Swap: remove one edge, add another (single swap)
            if not edges:
                no_improve += 1
                continue
            e_rm = random.choice(list(edges))
            u1, v1 = e_rm
            remove_edge(edges, adj_mask, u1, v1)
            non_edges_now = [x for x in all_pairs if x not in edges]
            if non_edges_now:
                weights = [bin(adj_mask[x[0]]).count('1') + bin(adj_mask[x[1]]).count('1') + 1 for x in non_edges_now]
                e_add = random.choices(non_edges_now, weights=weights, k=1)[0]
                if try_add_edge(edges, adj_mask, e_add[0], e_add[1]):
                    if len(edges) > best_count:
                        best_count = len(edges)
                        best_edges = set(edges)
                        improved = True
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Re-add removed edge
                    edges.add((u1, v1))
                    adj_mask[u1] |= (1 << v1)
                    adj_mask[v1] |= (1 << u1)
                    no_improve += 1
            else:
                edges.add((u1, v1))
                adj_mask[u1] |= (1 << v1)
                adj_mask[v1] |= (1 << u1)
                no_improve += 1

    # ---- Phase 3: Final greedy polish on best solution ----
    final_edges = set(best_edges)
    final_adj = [0] * 16
    for (u, v) in final_edges:
        final_adj[u] |= (1 << v)
        final_adj[v] |= (1 << u)

    non_edges_final = [e for e in all_pairs if e not in final_edges]
    non_edges_final.sort(key=lambda e: bin(final_adj[e[0]]).count('1') + bin(final_adj[e[1]]).count('1'), reverse=True)

    for e in non_edges_final:
        if time.time() > deadline - 5:
            break
        u, v = e
        if try_add_edge(final_edges, final_adj, u, v):
            pass

    # Verify the final result is Q3-free (safety check)
    # (This is expensive but ensures correctness)
    # Skip full verification for speed; the try_add_edge checks should guarantee it.

    return [list(e) for e in final_edges]

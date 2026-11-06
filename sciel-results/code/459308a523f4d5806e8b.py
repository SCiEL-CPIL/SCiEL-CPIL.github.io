import time
import random
import math

def construct():
    n = 40
    m = 401
    start_time = time.time()
    deadline = start_time + 118.0

    def compute_max_book(masks, edges):
        max_b = 0
        for u, v in edges:
            common = masks[u] & masks[v]
            b = common.bit_count()
            if b > max_b:
                max_b = b
        return max_b

    def try_greedy(B, edge_order=None):
        # Returns edges set if successful, else None
        if time.time() > deadline - 5.0:
            return None

        edges = set()
        masks = [0] * n
        book_sizes = {}

        # Pre-generate all potential edges
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        if edge_order is not None:
            all_edges = edge_order
        else:
            random.shuffle(all_edges)

        for u, v in all_edges:
            if len(edges) >= m:
                break

            # Check book size of new edge (u,v)
            common = masks[u] & masks[v]
            b_new = common.bit_count()
            if b_new > B:
                continue

            # Check that adding (u,v) won't push any existing edge's book over B
            valid = True
            T = common
            while T:
                lsb = T & -T
                w = lsb.bit_length() - 1
                T ^= lsb

                e_uw = (min(u, w), max(u, w))
                e_vw = (min(v, w), max(v, w))

                if book_sizes.get(e_uw, 0) >= B:
                    valid = False
                    break
                if book_sizes.get(e_vw, 0) >= B:
                    valid = False
                    break

            if not valid:
                continue

            # Add edge
            edges.add((u, v))
            masks[u] |= 1 << v
            masks[v] |= 1 << u

            # Update book sizes
            T = common
            while T:
                lsb = T & -T
                w = lsb.bit_length() - 1
                T ^= lsb

                e_uw = (min(u, w), max(u, w))
                e_vw = (min(v, w), max(v, w))

                book_sizes[e_uw] = book_sizes.get(e_uw, 0) + 1
                book_sizes[e_vw] = book_sizes.get(e_vw, 0) + 1

            book_sizes[(u, v)] = b_new

        if len(edges) == m:
            return edges, masks
        return None, None

    def simulated_annealing(initial_edges, initial_masks, max_iters=50000):
        """SA to minimize max book size."""
        if time.time() > deadline - 10.0:
            return initial_edges, initial_masks, compute_max_book(initial_masks, initial_edges)

        edges = set(initial_edges)
        masks = list(initial_masks)

        # Compute initial book sizes
        book_sizes = {}
        for u, v in edges:
            common = masks[u] & masks[v]
            book_sizes[(u, v)] = common.bit_count()

        current_max = max(book_sizes.values())
        best_edges = set(edges)
        best_masks = list(masks)
        best_max = current_max

        # Temperature schedule
        T0 = 5.0
        T_end = 0.01
        cooling = (T_end / T0) ** (1.0 / max(1, max_iters))
        T = T0

        all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
        non_edges = set(all_possible) - edges

        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break

            # Pick a random edge to remove
            edge_list = list(edges)
            a, b = random.choice(edge_list)

            # Pick a random non-edge to add
            if not non_edges:
                break
            c, d = random.choice(list(non_edges))

            # Compute change in max book if we swap (a,b) -> (c,d)
            common_ab = masks[a] & masks[b]
            common_cd = masks[c] & masks[d]

            # Edges whose book increases by 1: (c,w), (d,w) for w in common_cd
            # Edges whose book decreases by 1: (a,w), (b,w) for w in common_ab
            # Also remove (a,b), add (c,d)

            # Check validity: no edge can exceed current best + some slack
            # For SA, we allow temporary increases but penalize them

            # Estimate new max book
            # Edges that increase:
            increases = []
            T_bits = common_cd
            while T_bits:
                lsb = T_bits & -T_bits
                w = lsb.bit_length() - 1
                T_bits ^= lsb
                e_cw = (min(c, w), max(c, w))
                e_dw = (min(d, w), max(d, w))
                if e_cw in edges:
                    increases.append(e_cw)
                if e_dw in edges:
                    increases.append(e_dw)

            # Edges that decrease:
            decreases = []
            T_bits = common_ab
            while T_bits:
                lsb = T_bits & -T_bits
                w = lsb.bit_length() - 1
                T_bits ^= lsb
                e_aw = (min(a, w), max(a, w))
                e_bw = (min(b, w), max(b, w))
                if e_aw in edges and e_aw != (a, b):
                    decreases.append(e_aw)
                if e_bw in edges and e_bw != (a, b):
                    decreases.append(e_bw)

            # Compute new max book estimate
            new_max = current_max
            # Check if any increased edge exceeds
            for e in increases:
                if book_sizes.get(e, 0) + 1 > new_max:
                    new_max = book_sizes.get(e, 0) + 1

            # Check if (c,d) itself exceeds
            if common_cd.bit_count() > new_max:
                new_max = common_cd.bit_count()

            # If removing (a,b) was the max, recompute
            if book_sizes.get((a, b), 0) == current_max and current_max > 0:
                # Need to recompute max without (a,b) and with adjustments
                # For efficiency, just do a full recompute if this happens
                # But first check if the swap is clearly bad
                pass

            # Delta in objective (max book)
            delta = new_max - current_max

            # Acceptance criterion
            if delta <= 0:
                accept = True
            else:
                accept = random.random() < math.exp(-delta / max(T, 1e-10))

            if accept:
                # Perform swap
                edges.discard((a, b))
                edges.add((c, d))
                masks[a] &= ~(1 << b)
                masks[b] &= ~(1 << a)
                masks[c] |= 1 << d
                masks[d] |= 1 << c

                non_edges.discard((c, d))
                non_edges.add((a, b))

                # Update book sizes incrementally
                # Remove (a,b) from book_sizes
                book_sizes.pop((a, b), None)

                # Update edges affected by removal of (a,b)
                T_bits = common_ab
                while T_bits:
                    lsb = T_bits & -T_bits
                    w = lsb.bit_length() - 1
                    T_bits ^= lsb
                    e_aw = (min(a, w), max(a, w))
                    e_bw = (min(b, w), max(b, w))
                    if e_aw in book_sizes:
                        book_sizes[e_aw] -= 1
                        if book_sizes[e_aw] == 0:
                            pass  # keep at 0
                    if e_bw in book_sizes:
                        book_sizes[e_bw] -= 1

                # Update edges affected by addition of (c,d)
                T_bits = common_cd
                while T_bits:
                    lsb = T_bits & -T_bits
                    w = lsb.bit_length() - 1
                    T_bits ^= lsb
                    e_cw = (min(c, w), max(c, w))
                    e_dw = (min(d, w), max(d, w))
                    if e_cw in book_sizes:
                        book_sizes[e_cw] += 1
                    else:
                        book_sizes[e_cw] = 1
                    if e_dw in book_sizes:
                        book_sizes[e_dw] += 1
                    else:
                        book_sizes[e_dw] = 1
                book_sizes[(c, d)] = common_cd.bit_count()

                # Recompute current max
                current_max = max(book_sizes.values())

                # Track best
                if current_max < best_max:
                    best_max = current_max
                    best_edges = set(edges)
                    best_masks = list(masks)

            # Cool down
            T *= cooling

        return best_edges, best_masks, best_max

    def local_search(edges, masks, B, max_iters=2000):
        """Local search to achieve max book <= B."""
        if time.time() > deadline - 5.0:
            return edges, masks, compute_max_book(masks, edges)

        edges = set(edges)
        masks = list(masks)

        # Compute book sizes
        book_sizes = {}
        for u, v in edges:
            common = masks[u] & masks[v]
            book_sizes[(u, v)] = common.bit_count()

        current_max = max(book_sizes.values())
        if current_max <= B:
            return edges, masks, current_max

        all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
        non_edges = set(all_possible) - edges

        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() > deadline - 5.0:
                break

            # Find bottleneck edges
            bottleneck_edges = [e for e in edges if book_sizes.get(e, 0) > B]
            if not bottleneck_edges:
                current_max = max(book_sizes.values())
                return edges, masks, current_max

            random.shuffle(bottleneck_edges)
            for (a, b) in bottleneck_edges[:3]:
                if time.time() > deadline - 5.0:
                    break

                # Find candidate non-edges
                non_edge_list = list(non_edges)
                random.shuffle(non_edge_list)

                for (c, d) in non_edge_list[:100]:
                    if time.time() > deadline - 5.0:
                        break

                    common_cd = masks[c] & masks[d]
                    b_cd = common_cd.bit_count()
                    if b_cd > B:
                        continue

                    # Check if swap is valid
                    common_ab = masks[a] & masks[b]
                    valid_swap = True

                    # Check edges that will increase
                    T_bits = common_cd
                    while T_bits:
                        lsb = T_bits & -T_bits
                        w = lsb.bit_length() - 1
                        T_bits ^= lsb
                        e_cw = (min(c, w), max(c, w))
                        e_dw = (min(d, w), max(d, w))
                        if e_cw in edges and book_sizes.get(e_cw, 0) + 1 > B:
                            valid_swap = False
                            break
                        if e_dw in edges and book_sizes.get(e_dw, 0) + 1 > B:
                            valid_swap = False
                            break

                    if not valid_swap:
                        continue

                    # Perform swap
                    edges.discard((a, b))
                    edges.add((c, d))
                    masks[a] &= ~(1 << b)
                    masks[b] &= ~(1 << a)
                    masks[c] |= 1 << d
                    masks[d] |= 1 << c

                    non_edges.discard((c, d))
                    non_edges.add((a, b))

                    # Update book sizes
                    book_sizes.pop((a, b), None)

                    T_bits = common_ab
                    while T_bits:
                        lsb = T_bits & -T_bits
                        w = lsb.bit_length() - 1
                        T_bits ^= lsb
                        e_aw = (min(a, w), max(a, w))
                        e_bw = (min(b, w), max(b, w))
                        if e_aw in book_sizes:
                            book_sizes[e_aw] -= 1
                        if e_bw in book_sizes:
                            book_sizes[e_bw] -= 1

                    T_bits = common_cd
                    while T_bits:
                        lsb = T_bits & -T_bits
                        w = lsb.bit_length() - 1
                        T_bits ^= lsb
                        e_cw = (min(c, w), max(c, w))
                        e_dw = (min(d, w), max(d, w))
                        if e_cw in book_sizes:
                            book_sizes[e_cw] += 1
                        else:
                            book_sizes[e_cw] = 1
                        if e_dw in book_sizes:
                            book_sizes[e_dw] += 1
                        else:
                            book_sizes[e_dw] = 1
                    book_sizes[(c, d)] = b_cd

                    improved = True
                    break

                if improved:
                    break

        current_max = max(book_sizes.values()) if book_sizes else 0
        return edges, masks, current_max

    best_edges = None
    best_masks = None
    best_B = float('inf')

    # Phase 1: Greedy + Local Search
    for B in range(7, 11):
        if time.time() > deadline - 30.0:
            break

        success = False
        for attempt in range(6):
            if time.time() > deadline - 30.0:
                break

            result = try_greedy(B)
            if result[0]:
                edges, masks = result
                edges_ls, masks_ls, max_b = local_search(edges, masks, B, max_iters=2000)
                if max_b <= B:
                    if B < best_B:
                        best_B = B
                        best_edges = edges_ls
                        best_masks = masks_ls
                    success = True
                    break
                elif max_b < best_B:
                    best_B = max_b
                    best_edges = edges_ls
                    best_masks = masks_ls

        if success:
            break

    # Phase 2: Simulated Annealing from best found or from a greedy start
    if time.time() < deadline - 15.0:
        if best_edges is None:
            # Start from a random graph
            edges = set()
            all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
            random.shuffle(all_possible)
            for u, v in all_possible:
                if len(edges) >= m:
                    break
                edges.add((u, v))
            masks = [0] * n
            for u, v in edges:
                masks[u] |= 1 << v
                masks[v] |= 1 << u
        else:
            edges = set(best_edges)
            masks = list(best_masks)

        # Run SA with multiple restarts
        for sa_restart in range(3):
            if time.time() > deadline - 10.0:
                break

            if sa_restart == 0 and best_edges is not None:
                sa_edges, sa_masks = edges, masks
            else:
                # Fresh random start
                sa_edges = set()
                all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
                random.shuffle(all_possible)
                for u, v in all_possible:
                    if len(sa_edges) >= m:
                        break
                    sa_edges.add((u, v))
                sa_masks = [0] * n
                for u, v in sa_edges:
                    sa_masks[u] |= 1 << v
                    sa_masks[v] |= 1 << u

            sa_edges_out, sa_masks_out, sa_max = simulated_annealing(
                sa_edges, sa_masks, max_iters=30000
            )

            if sa_max < best_B:
                best_B = sa_max
                best_edges = sa_edges_out
                best_masks = sa_masks_out

            # Follow up with local search
            if time.time() < deadline - 5.0:
                ls_edges, ls_masks, ls_max = local_search(
                    sa_edges_out, sa_masks_out, min(sa_max, 7), max_iters=1000
                )
                if ls_max < best_B:
                    best_B = ls_max
                    best_edges = ls_edges
                    best_masks = ls_masks

    if best_edges is None:
        # Fallback
        edges = set()
        all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_possible)
        for u, v in all_possible:
            if len(edges) >= m:
                break
            edges.add((u, v))
        best_edges = edges

    return [[u, v] for u, v in best_edges]

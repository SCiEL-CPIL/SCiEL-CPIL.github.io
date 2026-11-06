import time
import random

def construct():
    n = 40
    m = 401
    start_time = time.time()
    deadline = start_time + 115.0

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
            return edges
        return None

    def local_search(edges, B, max_iters=5000):
        """Try to improve edge set via swaps to achieve max book <= B."""
        if time.time() > deadline - 5.0:
            return edges

        edges = set(edges)
        masks = [0] * n
        for u, v in edges:
            masks[u] |= 1 << v
            masks[v] |= 1 << u

        # Compute book sizes
        book_sizes = {}
        for u, v in edges:
            common = masks[u] & masks[v]
            book_sizes[(u, v)] = common.bit_count()

        # Check if already valid
        if max(book_sizes.values()) <= B:
            return edges

        # Try swaps: remove edge (a,b), add edge (c,d)
        # This reduces book sizes of edges affected by removing (a,b)
        # and increases book sizes of edges affected by adding (c,d)
        all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
        non_edges = [e for e in all_possible if e not in edges]

        random.shuffle(non_edges)

        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() > deadline - 5.0:
                break

            # Recompute book sizes
            book_sizes = {}
            for u, v in edges:
                common = masks[u] & masks[v]
                book_sizes[(u, v)] = common.bit_count()

            # Find edges with book > B (bottleneck edges)
            bottleneck_edges = [e for e in edges if book_sizes.get(e, 0) > B]
            if not bottleneck_edges:
                return edges

            # Try to fix one bottleneck edge via a swap
            random.shuffle(bottleneck_edges)
            for (a, b) in bottleneck_edges[:5]:
                if time.time() > deadline - 5.0:
                    break

                # Try adding a random non-edge that doesn't create too many new triangles
                candidates = []
                for (c, d) in non_edges[:200]:
                    # Skip if (c,d) is already an edge
                    if (c, d) in edges:
                        continue
                    # Check if adding (c,d) would push any edge over B
                    common_cd = masks[c] & masks[d]
                    b_cd = common_cd.bit_count()
                    if b_cd > B:
                        continue

                    # Check neighbors
                    valid = True
                    T = common_cd
                    while T:
                        lsb = T & -T
                        w = lsb.bit_length() - 1
                        T ^= lsb
                        e_cw = (min(c, w), max(c, w))
                        e_dw = (min(d, w), max(d, w))
                        if book_sizes.get(e_cw, 0) >= B:
                            valid = False
                            break
                        if book_sizes.get(e_dw, 0) >= B:
                            valid = False
                            break
                    if valid:
                        candidates.append((c, d))

                if not candidates:
                    continue

                random.shuffle(candidates)
                for (c, d) in candidates[:10]:
                    if time.time() > deadline - 5.0:
                        break

                    # Simulate swap: remove (a,b), add (c,d)
                    # First, check what happens to book sizes when we remove (a,b)
                    # Removing (a,b) decreases book of edges (a,w) and (b,w) for w in N(a)&N(b)
                    common_ab = masks[a] & masks[b]

                    # Check if the swap is beneficial
                    # New book sizes after swap:
                    # Edges affected by removal of (a,b): (a,w), (b,w) for w in common_ab -> decrease by 1
                    # Edges affected by addition of (c,d): (c,w), (d,w) for w in common_cd -> increase by 1
                    # Also (a,b) is removed, (c,d) is added

                    # Check that no edge exceeds B after swap
                    common_cd = masks[c] & masks[d]
                    valid_swap = True

                    # Check edges that will increase
                    T = common_cd
                    while T:
                        lsb = T & -T
                        w = lsb.bit_length() - 1
                        T ^= lsb
                        e_cw = (min(c, w), max(c, w))
                        e_dw = (min(d, w), max(d, w))
                        # These edges must exist
                        if e_cw in edges and book_sizes.get(e_cw, 0) + 1 > B:
                            valid_swap = False
                            break
                        if e_dw in edges and book_sizes.get(e_dw, 0) + 1 > B:
                            valid_swap = False
                            break

                    if not valid_swap:
                        continue

                    # Perform swap
                    edges.remove((a, b))
                    del book_sizes[(a, b)]
                    masks[a] &= ~(1 << b)
                    masks[b] &= ~(1 << a)

                    edges.add((c, d))
                    masks[c] |= 1 << d
                    masks[d] |= 1 << c

                    # Update book sizes
                    T = common_cd
                    while T:
                        lsb = T & -T
                        w = lsb.bit_length() - 1
                        T ^= lsb
                        e_cw = (min(c, w), max(c, w))
                        e_dw = (min(d, w), max(d, w))
                        book_sizes[e_cw] = book_sizes.get(e_cw, 0) + 1
                        book_sizes[e_dw] = book_sizes.get(e_dw, 0) + 1
                    book_sizes[(c, d)] = common_cd.bit_count()

                    # Update book sizes for edges affected by removal
                    T = common_ab
                    while T:
                        lsb = T & -T
                        w = lsb.bit_length() - 1
                        T ^= lsb
                        e_aw = (min(a, w), max(a, w))
                        e_bw = (min(b, w), max(b, w))
                        if e_aw in book_sizes:
                            book_sizes[e_aw] -= 1
                        if e_bw in book_sizes:
                            book_sizes[e_bw] -= 1

                    # Update non_edges list
                    non_edges.remove((c, d))
                    non_edges.append((a, b))

                    improved = True
                    break

                if improved:
                    break

        return edges

    def compute_max_book(edges):
        masks = [0] * n
        for u, v in edges:
            masks[u] |= 1 << v
            masks[v] |= 1 << u
        max_b = 0
        for u, v in edges:
            common = masks[u] & masks[v]
            b = common.bit_count()
            if b > max_b:
                max_b = b
        return max_b

    best_edges = None
    best_B = float('inf')

    # Try B from 7 upward
    for B in range(7, 12):
        if time.time() > deadline - 15.0:
            break

        success = False
        for attempt in range(8):
            if time.time() > deadline - 15.0:
                break

            # Different random seeds via different shuffles
            edges = try_greedy(B)
            if edges:
                # Try local search to see if we can achieve B
                edges_ls = local_search(edges, B, max_iters=3000)
                max_b = compute_max_book(edges_ls)
                if max_b <= B:
                    if B < best_B:
                        best_B = B
                        best_edges = edges_ls
                    success = True
                    break
                elif max_b < best_B:
                    best_B = max_b
                    best_edges = edges_ls

        if success:
            break

    if best_edges is None:
        # Fallback: just return a random graph
        edges = set()
        all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_possible)
        for u, v in all_possible:
            if len(edges) >= m:
                break
            edges.add((u, v))
        best_edges = edges

    return [[u, v] for u, v in best_edges]

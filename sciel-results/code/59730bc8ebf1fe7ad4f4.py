import time
import random
import sys

def construct():
    n = 40
    m = 401
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    def compute_books(edges_set):
        """Compute book size for each edge. Returns (book_dict, max_book)."""
        adj = [0] * n
        for u, v in edges_set:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        books = {}
        max_b = 0
        for u, v in edges_set:
            b = (adj[u] & adj[v]).bit_count()
            books[(u, v)] = b
            if b > max_b:
                max_b = b
        return books, max_b

    def near_bipartite_start(seed):
        """Start from K_20,20 (400 edges) + 1 intra-part edge, then shuffle."""
        rng = random.Random(seed)
        # Partition: 0..19 and 20..39
        edges = set()
        for u in range(20):
            for v in range(20, 40):
                edges.add((u, v))
        # Add one intra-part edge
        intra = rng.choice([(u, v) for u in range(20) for v in range(u + 1, 20)])
        edges.add(intra)
        return edges

    def random_start(seed):
        rng = random.Random(seed)
        pool = list(all_edges)
        rng.shuffle(pool)
        return set(pool[:m])

    def local_search(edges, max_iters, deadline, target_B=None):
        """
        Local search via edge swaps.
        Objective: minimize max book.
        If target_B is set, try to achieve max_book <= target_B.
        Returns (edges, max_book).
        """
        edges = set(edges)
        books, max_b = compute_books(edges)
        if target_B is not None and max_b <= target_B:
            return edges, max_b

        # Build non-edge list
        non_edges = [e for e in all_edges if e not in edges]

        adj = [0] * n
        for u, v in edges:
            adj[u] |= 1 << v
            adj[v] |= 1 << u

        def simulate_swap_remove(a, b):
            """Return list of (edge, delta) for removing (a,b)."""
            changes = {}
            common = adj[a] & adj[b]
            T = common
            while T:
                lsb = T & -T
                w = lsb.bit_length() - 1
                T ^= lsb
                e_aw = (min(a, w), max(a, w))
                e_bw = (min(b, w), max(b, w))
                changes[e_aw] = changes.get(e_aw, 0) - 1
                changes[e_bw] = changes.get(e_bw, 0) - 1
            changes[(a, b)] = -1000000  # mark removed
            return changes

        def simulate_swap_add(c, d):
            """Return list of (edge, delta) for adding (c,d)."""
            changes = {}
            common = adj[c] & adj[d]
            T = common
            while T:
                lsb = T & -T
                w = lsb.bit_length() - 1
                T ^= lsb
                e_cw = (min(c, w), max(c, w))
                e_dw = (min(d, w), max(d, w))
                changes[e_cw] = changes.get(e_cw, 0) + 1
                changes[e_dw] = changes.get(e_dw, 0) + 1
            changes[(c, d)] = common.bit_count()
            return changes

        def apply_swap_remove(a, b):
            common = adj[a] & adj[b]
            T = common
            while T:
                lsb = T & -T
                w = lsb.bit_length() - 1
                T ^= lsb
                e_aw = (min(a, w), max(a, w))
                e_bw = (min(b, w), max(b, w))
                if e_aw in books:
                    books[e_aw] -= 1
                if e_bw in books:
                    books[e_bw] -= 1
            del books[(a, b)]
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            edges.discard((a, b))

        def apply_swap_add(c, d):
            common = adj[c] & adj[d]
            T = common
            while T:
                lsb = T & -T
                w = lsb.bit_length() - 1
                T ^= lsb
                e_cw = (min(c, w), max(c, w))
                e_dw = (min(d, w), max(d, w))
                books[e_cw] = books.get(e_cw, 0) + 1
                books[e_dw] = books.get(e_dw, 0) + 1
            books[(c, d)] = common.bit_count()
            adj[c] |= 1 << d
            adj[d] |= 1 << c
            edges.add((c, d))

        best_max = max_b
        best_edges = set(edges)
        no_improve_count = 0

        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break

            # Recompute max book
            current_max = max(books.values()) if books else 0
            if current_max < best_max:
                best_max = current_max
                best_edges = set(edges)
                no_improve_count = 0
            else:
                no_improve_count += 1

            if target_B is not None and current_max <= target_B:
                return edges, current_max

            # Find bottleneck edges (book > target or book == current_max)
            if target_B is not None:
                threshold = target_B
            else:
                threshold = current_max

            bottleneck = [e for e in edges if books.get(e, 0) >= threshold]
            if not bottleneck:
                break

            random.shuffle(bottleneck)
            improved = False

            for (a, b) in bottleneck[:8]:
                if time.time() > deadline - 5.0:
                    break

                # Try to find a swap that reduces the max book
                # Strategy: remove (a,b) which is a bottleneck, add a low-book edge
                # First compute what removing (a,b) does
                common_ab = adj[a] & adj[b]
                remove_changes = {}
                T = common_ab
                while T:
                    lsb = T & -T
                    w = lsb.bit_length() - 1
                    T ^= lsb
                    e_aw = (min(a, w), max(a, w))
                    e_bw = (min(b, w), max(b, w))
                    remove_changes[e_aw] = remove_changes.get(e_aw, 0) - 1
                    remove_changes[e_bw] = remove_changes.get(e_bw, 0) - 1

                # Candidate new edges: prefer non-edges with low book
                # Score each non-edge by its book size
                candidates = []
                for (c, d) in non_edges:
                    if (c, d) in edges:
                        continue
                    common_cd = adj[c] & adj[d]
                    b_cd = common_cd.bit_count()
                    if b_cd >= threshold:
                        continue
                    # Check if adding (c,d) would push any edge over threshold
                    valid = True
                    T2 = common_cd
                    while T2:
                        lsb = T2 & -T2
                        w = lsb.bit_length() - 1
                        T2 ^= lsb
                        e_cw = (min(c, w), max(c, w))
                        e_dw = (min(d, w), max(d, w))
                        # After removal of (a,b), the book of e_cw/e_dw might be reduced
                        new_b_cw = books.get(e_cw, 0) + 1 + remove_changes.get(e_cw, 0)
                        new_b_dw = books.get(e_dw, 0) + 1 + remove_changes.get(e_dw, 0)
                        if new_b_cw > threshold or new_b_dw > threshold:
                            valid = False
                            break
                    if valid:
                        candidates.append((b_cd, c, d))

                if not candidates:
                    continue

                candidates.sort(key=lambda x: x[0])
                random.shuffle(candidates[:30])

                for (b_cd, c, d) in candidates[:15]:
                    if time.time() > deadline - 5.0:
                        break

                    # Simulate the swap fully
                    # Remove (a,b), add (c,d)
                    # Check all affected edges don't exceed threshold
                    common_cd = adj[c] & adj[d]

                    # Edges that increase: (c,w), (d,w) for w in common_cd
                    # Edges that decrease: (a,w), (b,w) for w in common_ab
                    # Net effect on each edge
                    affected = {}
                    T = common_cd
                    while T:
                        lsb = T & -T
                        w = lsb.bit_length() - 1
                        T ^= lsb
                        e_cw = (min(c, w), max(c, w))
                        e_dw = (min(d, w), max(d, w))
                        affected[e_cw] = affected.get(e_cw, 0) + 1
                        affected[e_dw] = affected.get(e_dw, 0) + 1
                    T = common_ab
                    while T:
                        lsb = T & -T
                        w = lsb.bit_length() - 1
                        T ^= lsb
                        e_aw = (min(a, w), max(a, w))
                        e_bw = (min(b, w), max(b, w))
                        affected[e_aw] = affected.get(e_aw, 0) - 1
                        affected[e_bw] = affected.get(e_bw, 0) - 1

                    # Check validity: no edge exceeds threshold after swap
                    valid_swap = True
                    for e, delta in affected.items():
                        if e in edges:
                            new_b = books.get(e, 0) + delta
                            if new_b > threshold:
                                valid_swap = False
                                break
                    if not valid_swap:
                        continue

                    # Also check the new edge (c,d) itself
                    if common_cd.bit_count() > threshold:
                        continue

                    # Compute new max book after swap
                    new_max = 0
                    for e in edges:
                        if e == (a, b):
                            continue
                        if e == (c, d):
                            nb = common_cd.bit_count()
                        else:
                            nb = books.get(e, 0) + affected.get(e, 0)
                        if nb > new_max:
                            new_max = nb
                    # Add (c,d)
                    if common_cd.bit_count() > new_max:
                        new_max = common_cd.bit_count()

                    if new_max >= current_max:
                        continue

                    # Apply swap
                    apply_swap_remove(a, b)
                    non_edges.append((a, b))
                    non_edges.remove((c, d))
                    apply_swap_add(c, d)

                    max_b = new_max
                    improved = True
                    no_improve_count = 0
                    break

                if improved:
                    break

            if not improved and no_improve_count > 200:
                # Perturb: do a random swap to escape local optimum
                if non_edges:
                    random.shuffle(list(edges))
                    edge_list = list(edges)
                    if edge_list:
                        (a, b) = random.choice(edge_list)
                        (c, d) = random.choice(non_edges)
                        apply_swap_remove(a, b)
                        non_edges.append((a, b))
                        non_edges.remove((c, d))
                        apply_swap_add(c, d)
                        books, max_b = compute_books(edges)
                        no_improve_count = 0

        # Final compute
        books, max_b = compute_books(edges)
        if max_b < best_max:
            best_max = max_b
            best_edges = set(edges)
        return best_edges, best_max

    def local_search_target(edges, target_B, max_iters, deadline):
        """Local search specifically targeting max_book <= target_B."""
        return local_search(edges, max_iters, deadline, target_B=target_B)

    best_edges = None
    best_score = float('inf')

    # Strategy: Try to achieve target B from 7 upward
    # Start from near-bipartite (low triangle count) and random starts

    starts = []
    # Near-bipartite starts with different random seeds
    for s in range(6):
        starts.append(('nb', s))
    # Random starts
    for s in range(6):
        starts.append(('rand', s))

    # First pass: find the best achievable max_book with general local search
    for kind, seed in starts:
        if time.time() > deadline - 30.0:
            break
        if kind == 'nb':
            edges = near_bipartite_start(seed)
        else:
            edges = random_start(seed)

        edges, max_b = local_search(edges, max_iters=3000, deadline=deadline)
        if max_b < best_score:
            best_score = max_b
            best_edges = set(edges)
            print(f"Found max_book={max_b} from {kind} seed={seed}", file=sys.stderr)

    # Second pass: try to push down to specific targets starting from best
    if best_edges is not None:
        for target in range(7, best_score):
            if time.time() > deadline - 20.0:
                break
            # Try multiple times from perturbed versions of best_edges
            for attempt in range(5):
                if time.time() > deadline - 15.0:
                    break
                # Perturb best_edges
                perturbed = set(best_edges)
                for _ in range(3):
                    if time.time() > deadline - 15.0:
                        break
                    edge_list = list(perturbed)
                    if not edge_list:
                        break
                    non_edge_pool = [e for e in all_edges if e not in perturbed]
                    if not non_edge_pool:
                        break
                    (a, b) = random.choice(edge_list)
                    (c, d) = random.choice(non_edge_pool)
                    perturbed.discard((a, b))
                    perturbed.add((c, d))

                edges, max_b = local_search_target(perturbed, target, max_iters=2000, deadline=deadline)
                if max_b <= target:
                    best_score = max_b
                    best_edges = set(edges)
                    print(f"Achieved target {target}! max_book={max_b}", file=sys.stderr)
                    break
                elif max_b < best_score:
                    best_score = max_b
                    best_edges = set(edges)

    # Final: if we have best_edges, try one more round of general local search
    if best_edges is not None and time.time() < deadline - 10.0:
        edges, max_b = local_search(best_edges, max_iters=5000, deadline=deadline)
        if max_b < best_score:
            best_score = max_b
            best_edges = set(edges)

    if best_edges is None:
        # Fallback
        edges = near_bipartite_start(0)
        best_edges = edges

    return [[u, v] for u, v in best_edges]

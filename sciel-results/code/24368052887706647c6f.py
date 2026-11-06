"""
Spectral relaxation + local search to minimize largest book.

Strategy:
1. Initialize with a near-regular graph (as balanced as possible).
2. Use projected gradient descent on a continuous relaxation where edge weights
   are in [0,1], minimizing a smooth surrogate for the max book size.
3. Round via stochastic rounding, then refine with aggressive local search.
"""
import time
import random
import math
import numpy as np

def construct():
    n = 40
    m = 401
    deadline = time.time() + 140  # safety margin

    # Precompute all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
    edge_index = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)  # 780

    def build_adj_from_edges(edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def compute_books(adj, edge_set):
        books = {}
        for u, v in edge_set:
            books[(u, v)] = len(adj[u] & adj[v])
        return books

    def max_book(books):
        return max(books.values()) if books else 0

    def sum_books(books):
        return sum(books.values())

    # ---------- Phase 1: Build a near-regular initial graph ----------
    def make_near_regular():
        """
        Build a graph with 401 edges that is as close to regular as possible.
        401 / 40 = 10.025, so most vertices have degree 10, one has degree 11.
        We use a circulant-like construction.
        """
        edges = set()
        # Base: cycle + chords for regularity
        # Each vertex connects to next 5 vertices (undirected) => degree 10
        for u in range(n):
            for offset in range(1, 6):
                v = (u + offset) % n
                if u < v:
                    edges.add((u, v))
                elif v < u:
                    edges.add((v, u))
        # That gives 40*5 = 200 edges (each undirected edge counted once)
        # Wait: for each u, we add (u, (u+offset)%n) for offset 1..5.
        # Since we deduplicate, this is a 10-regular circulant: 40*10/2 = 200 edges.
        # We need 401 edges, so we need 201 more.

        # Add a second layer of chords with larger offsets
        for u in range(n):
            for offset in range(6, 11):
                v = (u + offset) % n
                if u < v:
                    edges.add((u, v))
                elif v < u:
                    edges.add((v, u))
        # Now we have 40*10/2 = 200 edges from first layer, plus second layer.
        # Second layer: offsets 6..10, that's 5 more per vertex => another 200 edges.
        # Total: 400 edges. Need 1 more.

        # Add one more edge
        if len(edges) < m:
            # Add a random edge not already present
            remaining = [e for e in all_edges if e not in edges]
            random.shuffle(remaining)
            for e in remaining:
                edges.add(e)
                if len(edges) == m:
                    break

        # If we have too many (shouldn't happen), trim
        if len(edges) > m:
            edge_list = list(edges)
            random.shuffle(edge_list)
            edges = set(edge_list[:m])

        return edges

    # ---------- Phase 2: Spectral relaxation via projected gradient ----------
    def spectral_relaxation(edge_set, max_iters=200, lr=0.05):
        """
        Treat edge weights as continuous variables in [0,1].
        Objective: minimize a smooth surrogate for max book.
        We use: sum over edges of (book_size)^p for large p, approximating max.
        Also add a penalty for deviating from exactly m edges.
        """
        # Convert to numpy adjacency
        A = np.zeros((n, n), dtype=np.float64)
        for u, v in edge_set:
            A[u, v] = 1.0
            A[v, u] = 1.0

        # Track which edges are "active" (in the set)
        # We'll optimize over all 780 possible edge weights
        w = np.zeros(num_edges, dtype=np.float64)
        for u, v in edge_set:
            w[edge_index[(u, v)]] = 1.0

        best_w = w.copy()
        best_score = float('inf')

        for it in range(max_iters):
            if time.time() > deadline - 60:
                break

            # Build adjacency matrix from weights
            A = np.zeros((n, n), dtype=np.float64)
            for i, (u, v) in enumerate(all_edges):
                A[u, v] = w[i]
                A[v, u] = w[i]

            # A^2: number of length-2 paths between each pair
            A2 = A @ A

            # Book size for edge (u,v) = (A2)[u,v] - A[u,v] (common neighbors)
            # Actually, book(u,v) = |N(u) ∩ N(v)| = A2[u,v] - A[u,v] for unweighted
            # For weighted: it's A2[u,v] - A[u,v]
            book = A2 - A  # This gives common neighbors for all pairs

            # Only consider edges where w[i] > 0.5 (active edges)
            # Surrogate: sum of book^p for active edges
            p = 8  # large power to approximate max
            active_mask = w > 0.3
            books_active = book[all_edges]  # shape (num_edges,)
            # book[i] for edge i is book[u,v]
            # Extract book sizes for each edge
            bk_sizes = np.array([book[u, v] for u, v in all_edges])

            # Objective: sum of (bk_sizes + eps)^p for edges with weight > 0
            eps = 1e-6
            obj = np.sum(np.power(bk_sizes + eps, p) * (w ** 2))
            # Penalty for not having exactly m edges
            edge_count = np.sum(w)
            penalty = 10.0 * (edge_count - m) ** 2

            total_obj = obj + penalty

            if total_obj < best_score:
                best_score = total_obj
                best_w = w.copy()

            # Gradient computation (approximate via finite differences for a few coords,
            # or use automatic differentiation... let's do coordinate descent)
            # For speed, do coordinate descent: update one edge weight at a time
            for _ in range(10):  # 10 random coordinates per iteration
                idx = random.randrange(num_edges)
                u, v = all_edges[idx]
                old_w = w[idx]

                # Try a small perturbation
                for delta in [0.1, -0.1, 0.05, -0.05]:
                    new_w_val = np.clip(old_w + delta, 0, 1)
                    if new_w_val == old_w:
                        continue
                    w[idx] = new_w_val

                    # Recompute objective
                    A = np.zeros((n, n), dtype=np.float64)
                    for i, (eu, ev) in enumerate(all_edges):
                        A[eu, ev] = w[i]
                        A[ev, eu] = w[i]
                    A2 = A @ A
                    bk_sizes = np.array([A2[eu, ev] - A[eu, ev] for eu, ev in all_edges])
                    obj_new = np.sum(np.power(bk_sizes + eps, p) * (w ** 2))
                    edge_count = np.sum(w)
                    penalty_new = 10.0 * (edge_count - m) ** 2
                    total_new = obj_new + penalty_new

                    if total_new < total_obj:
                        total_obj = total_new
                        break
                    else:
                        w[idx] = old_w
                        break

        # Round the weights
        edge_set_final = set()
        # Sort by weight, take top m
        sorted_indices = np.argsort(-best_w)
        for idx in sorted_indices:
            if len(edge_set_final) >= m:
                break
            edge_set_final.add(all_edges[idx])

        # Ensure we have exactly m edges
        if len(edge_set_final) < m:
            for e in all_edges:
                if len(edge_set_final) >= m:
                    break
                if e not in edge_set_final:
                    edge_set_final.add(e)

        return edge_set_final

    # ---------- Phase 3: Aggressive local search ----------
    def local_search(edge_set, adj, books, max_rounds=500):
        """
        Local search: try to reduce max book by swapping edges.
        Focus on edges with the largest book sizes.
        """
        current_max = max_book(books)
        current_sum = sum_books(books)
        best_max = current_max
        best_sum = current_sum
        best_edge_set = set(edge_set)
        best_books = dict(books)
        best_adj = [set(s) for s in adj]

        for round_num in range(max_rounds):
            if time.time() > deadline - 15:
                break

            # Get edges sorted by book size descending
            sorted_edges = sorted(books.items(), key=lambda x: -x[1])

            improved_this_round = False
            for (u, v), bk in sorted_edges:
                if time.time() > deadline - 15:
                    break
                if bk < current_max:
                    break

                # Try to find a replacement edge
                # Sample candidate non-edges
                candidates = []
                for _ in range(30):
                    c = random.randrange(n)
                    d = random.randrange(n)
                    if c == d:
                        continue
                    c, d = min(c, d), max(c, d)
                    if (c, d) in edge_set or (c, d) == (u, v):
                        continue
                    candidates.append((c, d))

                random.shuffle(candidates)

                for (c, d) in candidates:
                    # Tentative swap
                    edge_set.discard((u, v))
                    edge_set.add((c, d))
                    adj[u].discard(v)
                    adj[v].discard(u)
                    adj[c].add(d)
                    adj[d].add(c)

                    # Recompute books
                    new_books = {}
                    for eu, ev in edge_set:
                        new_books[(eu, ev)] = len(adj[eu] & adj[ev])
                    new_max = max_book(new_books)
                    new_sum = sum_books(new_books)

                    if new_max < current_max or (new_max == current_max and new_sum < current_sum):
                        books = new_books
                        current_max = new_max
                        current_sum = new_sum
                        improved_this_round = True
                        if new_max < best_max or (new_max == best_max and new_sum < best_sum):
                            best_max = new_max
                            best_sum = new_sum
                            best_edge_set = set(edge_set)
                            best_books = dict(new_books)
                            best_adj = [set(s) for s in adj]
                        break
                    else:
                        # Revert
                        edge_set.discard((c, d))
                        edge_set.add((u, v))
                        adj[u].add(v)
                        adj[v].add(u)
                        adj[c].discard(d)
                        adj[d].discard(c)

            if not improved_this_round:
                # Try more random swaps
                for _ in range(50):
                    if time.time() > deadline - 15:
                        break
                    # Pick random edge to remove
                    edge_list = list(edge_set)
                    u, v = random.choice(edge_list)
                    # Pick random non-edge to add
                    for _ in range(20):
                        c = random.randrange(n)
                        d = random.randrange(n)
                        if c == d:
                            continue
                        c, d = min(c, d), max(c, d)
                        if (c, d) in edge_set or (c, d) == (u, v):
                            continue

                        edge_set.discard((u, v))
                        edge_set.add((c, d))
                        adj[u].discard(v)
                        adj[v].discard(u)
                        adj[c].add(d)
                        adj[d].add(c)

                        new_books = {}
                        for eu, ev in edge_set:
                            new_books[(eu, ev)] = len(adj[eu] & adj[ev])
                        new_max = max_book(new_books)
                        new_sum = sum_books(new_books)

                        # Accept if not worse, or with small probability if slightly worse
                        if new_max <= current_max:
                            books = new_books
                            current_max = new_max
                            current_sum = new_sum
                            if new_max < best_max or (new_max == best_max and new_sum < best_sum):
                                best_max = new_max
                                best_sum = new_sum
                                best_edge_set = set(edge_set)
                                best_books = dict(new_books)
                                best_adj = [set(s) for s in adj]
                            break
                        else:
                            # Revert
                            edge_set.discard((c, d))
                            edge_set.add((u, v))
                            adj[u].add(v)
                            adj[v].add(u)
                            adj[c].discard(d)
                            adj[d].discard(c)

        return best_edge_set, best_adj, best_books, best_max

    # ---------- Main: Multi-start ----------
    best_overall_max = float('inf')
    best_overall_edge_set = None
    best_overall_adj = None
    best_overall_books = None

    # Start 1: Near-regular construction
    edge_set = make_near_regular()
    adj = build_adj_from_edges(edge_set)
    books = compute_books(adj, edge_set)
    current_max = max_book(books)

    if current_max < best_overall_max:
        best_overall_max = current_max
        best_overall_edge_set = set(edge_set)
        best_overall_adj = [set(s) for s in adj]
        best_overall_books = dict(books)

    # Run spectral relaxation
    if time.time() < deadline - 80:
        relaxed_edge_set = spectral_relaxation(edge_set, max_iters=100)
        adj2 = build_adj_from_edges(relaxed_edge_set)
        books2 = compute_books(adj2, relaxed_edge_set)
        max2 = max_book(books2)

        if max2 < best_overall_max:
            best_overall_max = max2
            best_overall_edge_set = set(relaxed_edge_set)
            best_overall_adj = [set(s) for s in adj2]
            best_overall_books = dict(books2)

        # Local search on relaxed
        if time.time() < deadline - 30:
            ls_es, ls_adj, ls_books, ls_max = local_search(relaxed_edge_set, adj2, books2, max_rounds=200)
            if ls_max < best_overall_max:
                best_overall_max = ls_max
                best_overall_edge_set = ls_es
                best_overall_adj = ls_adj
                best_overall_books = ls_books

    # Local search on original
    if time.time() < deadline - 30:
        ls_es, ls_adj, ls_books, ls_max = local_search(edge_set, adj, books, max_rounds=300)
        if ls_max < best_overall_max:
            best_overall_max = ls_max
            best_overall_edge_set = ls_es
            best_overall_adj = ls_adj
            best_overall_books = ls_books

    # Additional random restarts if time permits
    num_extra_starts = 5
    for start in range(num_extra_starts):
        if time.time() > deadline - 25:
            break

        # Random near-regular perturbation
        es = make_near_regular()
        # Perturb: swap some random edges
        es_list = list(es)
        for _ in range(20):
            u, v = random.choice(es_list)
            c = random.randrange(n)
            d = random.randrange(n)
            if c == d:
                continue
            c, d = min(c, d), max(c, d)
            if (c, d) in es or (c, d) == (u, v):
                continue
            es.discard((u, v))
            es.add((c, d))
            es_list = list(es)

        adj = build_adj_from_edges(es)
        books = compute_books(adj, es)
        cur_max = max_book(books)

        if cur_max < best_overall_max:
            best_overall_max = cur_max
            best_overall_edge_set = set(es)
            best_overall_adj = [set(s) for s in adj]
            best_overall_books = dict(books)

        if time.time() < deadline - 20:
            ls_es, ls_adj, ls_books, ls_max = local_search(es, adj, books, max_rounds=100)
            if ls_max < best_overall_max:
                best_overall_max = ls_max
                best_overall_edge_set = ls_es
                best_overall_adj = ls_adj
                best_overall_books = ls_books

    # Final refinement
    if best_overall_edge_set is not None and time.time() < deadline - 10:
        es = set(best_overall_edge_set)
        adj = [set(s) for s in best_overall_adj]
        books = dict(best_overall_books)
        cur_max = max_book(books)

        # Intensive local search
        ls_es, ls_adj, ls_books, ls_max = local_search(es, adj, books, max_rounds=500)
        if ls_max < best_overall_max:
            best_overall_max = ls_max
            best_overall_edge_set = ls_es
            best_overall_adj = ls_adj
            best_overall_books = ls_books

    # Return
    if best_overall_edge_set is None:
        # Fallback
        es = make_near_regular()
        return [[u, v] for u, v in es]

    return [[u, v] for u, v in best_overall_edge_set]

import numpy as np
import random
import time

def construct():
    n = 40
    m = 401
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 120s kill

    # ---------- Fast book-size computation ----------
    # book[e] = number of triangles containing edge e
    # For edge (u,v): book = |N(u) ∩ N(v)|

    def compute_books(adj):
        """Compute book size for every edge. Returns dict (u,v)->count for u<v."""
        books = {}
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    # common neighbors
                    c = np.sum(adj[u] & adj[v])
                    books[(u, v)] = int(c)
        return books

    def max_book(books):
        return max(books.values()) if books else 0

    # ---------- Build initial K20,20 + 1 intra edge ----------
    def initial_graph():
        adj = np.zeros((n, n), dtype=np.uint8)
        edges = set()
        # K20,20: vertices 0..19 vs 20..39
        for u in range(20):
            for v in range(20, 40):
                adj[u, v] = 1
                adj[v, u] = 1
                edges.add((u, v))
        # Add 1 intra-partition edge to get 401
        # Choose edge within part 0: (0,1)
        adj[0, 1] = 1
        adj[1, 0] = 1
        edges.add((0, 1))
        return adj, edges

    adj, edges = initial_graph()
    books = compute_books(adj)
    cur_max = max_book(books)
    best_max = cur_max
    best_edges = set(edges)
    best_adj = adj.copy()

    # Precompute non-edges for quick sampling
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    non_edges = [p for p in all_pairs if not adj[p[0], p[1]]]

    # ---------- Phase 1: Local search to reduce max book ----------
    # Only accept swaps that reduce the max book
    def try_swap(adj, edges, books, non_edges, out_e, in_e):
        """Try swapping out_e with in_e. Return True if accepted (and update in place)."""
        u1, v1 = out_e
        u2, v2 = in_e

        # Remove out_e
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        edges.discard(out_e)
        non_edges.append(out_e)

        # Add in_e
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        edges.add(in_e)
        # Remove in_e from non_edges
        try:
            non_edges.remove(in_e)
        except ValueError:
            pass

        # Recompute books (fast enough for 40 vertices)
        new_books = compute_books(adj)
        new_max = max_book(new_books)

        if new_max <= max_book(books):
            books.clear()
            books.update(new_books)
            return True
        else:
            # Revert
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            edges.discard(in_e)
            non_edges.append(in_e)

            adj[u1, v1] = 1
            adj[v1, u1] = 1
            edges.add(out_e)
            try:
                non_edges.remove(out_e)
            except ValueError:
                pass

            books.clear()
            books.update(compute_books(adj))
            return False

    # Phase 1: Reduce max book from 6 if possible
    stagnation_counter = 0
    while time.time() < deadline - 10.0:
        if best_max <= 5:
            break

        # Pick a random edge with the highest book
        max_edges = [e for e, b in books.items() if b == max_book(books)]
        out_e = random.choice(max_edges)

        # Pick a random non-edge
        if not non_edges:
            break
        in_e = random.choice(non_edges)

        old_max = max_book(books)
        accepted = try_swap(adj, edges, books, non_edges, out_e, in_e)
        new_max = max_book(books)

        if accepted and new_max < best_max:
            best_max = new_max
            best_edges = set(edges)
            best_adj = adj.copy()
            stagnation_counter = 0
        elif accepted:
            stagnation_counter += 1
        else:
            stagnation_counter += 1

        if stagnation_counter > 5000:
            # Restart from best
            adj = best_adj.copy()
            edges = set(best_edges)
            books = compute_books(adj)
            non_edges = [p for p in all_pairs if not adj[p[0], p[1]]]
            stagnation_counter = 0
            if best_max <= 5:
                break

    # ---------- Phase 2: Try to push max book from 6 to 5 ----------
    # If we're at 6, try more aggressive search
    if best_max == 6 and time.time() < deadline - 5.0:
        # Try multiple random restarts with simulated-annealing-like acceptance
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            adj = best_adj.copy()
            edges = set(best_edges)
            books = compute_books(adj)
            non_edges = [p for p in all_pairs if not adj[p[0], p[1]]]
            cur_max = max_book(books)

            for step in range(20000):
                if time.time() > deadline - 5.0:
                    break
                if cur_max <= 5:
                    best_max = 5
                    best_edges = set(edges)
                    best_adj = adj.copy()
                    break

                # Pick edge with max book
                max_edges = [e for e, b in books.items() if b == cur_max]
                out_e = random.choice(max_edges)
                if not non_edges:
                    break
                in_e = random.choice(non_edges)

                old_max = cur_max
                accepted = try_swap(adj, edges, books, non_edges, out_e, in_e)
                cur_max = max_book(books)

                if accepted and cur_max < best_max:
                    best_max = cur_max
                    best_edges = set(edges)
                    best_adj = adj.copy()

    # ---------- Return best ----------
    result = [[u, v] for u, v in best_edges]
    return result

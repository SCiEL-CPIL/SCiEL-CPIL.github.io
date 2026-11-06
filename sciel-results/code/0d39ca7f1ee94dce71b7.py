"""Minimize the largest book in a 40-vertex, 401-edge graph.

Combines:
  - Fast book-size evaluation via matrix multiply (A @ A.T) * A  [from current program]
  - Simulated annealing with maintained edge/non-edge lists for O(1) move sampling [inspiration 1]
  - Targeted refinement on the max-book edge  [from current program]
"""
import time
import math
import numpy as np


def construct():
    deadline = time.time() + 110  # safety margin before the 120s (evaluator 150s) limit
    n = 40
    m = 401

    def compute_books(A):
        """Book size for each edge: (A @ A.T) * A, zero on non-edges and diagonal."""
        AA = A @ A.T
        books = AA * A
        np.fill_diagonal(books, 0)
        return books

    def max_book(A):
        return int(compute_books(A).max())

    def build_lists(A):
        """Return (edge_list, non_edge_list) of (i,j) with i<j."""
        edge_list, non_edge_list = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j]:
                    edge_list.append((i, j))
                else:
                    non_edge_list.append((i, j))
        return edge_list, non_edge_list

    def random_graph():
        A = np.zeros((n, n), dtype=np.int32)
        # Fisher-Yates over all pairs, take first m
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        np.random.shuffle(pairs)
        for (i, j) in pairs[:m]:
            A[i, j] = 1
            A[j, i] = 1
        return A

    def sa_run(A, T0=1.0, cool=0.9995, min_T=1e-4):
        """Simulated annealing with maintained edge/non-edge lists.
        Returns (best_A, best_score)."""
        edge_list, non_edge_list = build_lists(A)
        T = T0
        current_max = max_book(A)
        best_A = A.copy()
        best_max = current_max
        # index maps for O(1) removal from lists
        edge_idx = {e: k for k, e in enumerate(edge_list)}
        non_edge_idx = {e: k for k, e in enumerate(non_edge_list)}

        while time.time() < deadline - 8:
            if T < min_T:
                T = T0 * 0.3  # reheat to escape deep local minima
            k = np.random.randint(len(edge_list))
            u, v = edge_list[k]
            l = np.random.randint(len(non_edge_list))
            a, b = non_edge_list[l]
            if (u == a and v == b) or (u == b and v == a):
                continue

            # apply swap
            A[u, v] = 0; A[v, u] = 0
            A[a, b] = 1; A[b, a] = 1
            new_max = max_book(A)
            delta = new_max - current_max

            if delta <= 0 or (T > 0 and random_accept(delta, T)):
                # accept
                current_max = new_max
                # update lists in O(1)
                edge_list[k] = (a, b)
                edge_idx[(a, b)] = k
                non_edge_list[l] = (u, v)
                non_edge_idx[(u, v)] = l
                if current_max < best_max:
                    best_max = current_max
                    best_A = A.copy()
            else:
                # reject
                A[u, v] = 1; A[v, u] = 1
                A[a, b] = 0; A[b, a] = 0
            T *= cool
        return best_A, best_max

    def random_accept(delta, T):
        return np.random.random() < math.exp(-delta / T)

    def targeted_refine(A, max_rounds=40):
        """Repeatedly remove the max-book edge and try many candidate additions."""
        best_max = max_book(A)
        for _ in range(max_rounds):
            if time.time() > deadline - 8:
                break
            books = compute_books(A)
            # find an actual edge with the max book
            flat = np.argmax(books)
            i, j = np.unravel_index(flat, (n, n))
            if A[i, j] == 0:
                break
            max_val = int(books[i, j])
            if max_val <= 1:
                break
            # remove it
            A[i, j] = 0; A[j, i] = 0
            best_new = max_val + 1
            best_add = None
            # try many random non-edges
            for _ in range(200):
                a = np.random.randint(n)
                b = np.random.randint(n)
                if a == b or A[a, b] == 1:
                    continue
                A[a, b] = 1; A[b, a] = 1
                nb = max_book(A)
                A[a, b] = 0; A[b, a] = 0
                if nb < best_new:
                    best_new = nb
                    best_add = (a, b)
            if best_add is not None and best_new < best_max:
                a, b = best_add
                A[a, b] = 1; A[b, a] = 1
                best_max = best_new
            else:
                # revert removal
                A[i, j] = 1; A[j, i] = 1
        return A, best_max

    np.random.seed(12345)
    best_overall = float('inf')
    best_A = None

    n_restarts = 6
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
        A = random_graph()
        A, score = sa_run(A, T0=1.0, cool=0.9995, min_T=1e-4)
        if score < best_overall:
            best_overall = score
            best_A = A.copy()

    # Targeted refinement on the best solution found
    if best_A is not None and time.time() < deadline - 10:
        A = best_A.copy()
        A, score = targeted_refine(A, max_rounds=30)
        if score < best_overall:
            best_overall = score
            best_A = A.copy()

    # One more SA pass at low temperature to polish
    if best_A is not None and time.time() < deadline - 12:
        A = best_A.copy()
        A, score = sa_run(A, T0=0.3, cool=0.999, min_T=1e-5)
        if score < best_overall:
            best_overall = score
            best_A = A.copy()

    # Final targeted pass
    if best_A is not None and time.time() < deadline - 5:
        A = best_A.copy()
        A, score = targeted_refine(A, max_rounds=20)
        if score < best_overall:
            best_overall = score
            best_A = A.copy()

    # Build result edge list
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_A[i, j]:
                result.append([i, j])

    # Verify
    assert len(result) == 401, f"Expected 401 edges, got {len(result)}"
    seen = set()
    for e in result:
        u, v = e
        assert 0 <= u <= 39 and 0 <= v <= 39
        key = (min(u, v), max(u, v))
        assert key not in seen
        seen.add(key)

    return result

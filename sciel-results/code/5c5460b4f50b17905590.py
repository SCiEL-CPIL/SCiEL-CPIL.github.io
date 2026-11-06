"""Minimize the largest book in a 40-vertex, 401-edge graph.

Structural change: replace full-matrix A @ A re-evaluation with an
incremental book-size maintenance (O(n) per swap) and a targeted
"break-the-max" local search that always tries to remove an edge whose
book equals the current maximum.  This is a different optimization
family than plain simulated annealing: it is a greedy repair loop with
temperature-based acceptance of non-improving swaps to escape local
minima, but the proposal distribution is biased toward high-book edges.
"""
import time
import math
import random
import numpy as np


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120 s limit

    n = 40
    m = 401
    total_pairs = n * (n - 1) // 2
    num_non_edges = total_pairs - m  # 379

    # Precompute all possible edges once
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # ---------- Fast book-size maintenance ----------
    # books[u, v] = number of common neighbours of u and v
    # For an edge (u, v) the book size equals books[u, v].
    # When we add edge (u, v):
    #   books[u, v] += 1  (u and v become common neighbours of each other? No.)
    # Actually books[u,v] = sum_k A[u,k]*A[k,v].  Adding edge (u,v) changes
    # books for pairs that involve u or v:
    #   For any w != u, v:
    #     books[u, w] += A[w, v]   (because A[u,v] becomes 1)
    #     books[v, w] += A[w, u]   (because A[v,u] becomes 1)
    #   books[u, v] += 0? Let's check:
    #     books[u, v] = sum_k A[u,k] A[k,v].  The term k=w contributes A[u,w]A[w,v].
    #     Adding A[u,v]=1 does not change books[u,v] directly because k ranges over
    #     all vertices and A[u,v] is not part of the sum for books[u,v] (k=v gives
    #     A[u,v]A[v,v]=0, k=u gives A[u,u]A[u,v]=0).
    #   So books[u,v] is unchanged by adding (u,v).
    #   But for any w: books[u,w] gains A[w,v] and books[v,w] gains A[w,u].
    #
    # Similarly removing edge (u,v) subtracts those contributions.

    def make_books(A):
        """Compute books = A @ A (common-neighbor counts)."""
        return A @ A

    def add_edge(A, books, u, v):
        """Add edge (u,v) and update books incrementally."""
        A[u, v] = 1
        A[v, u] = 1
        # For every w != u, v:
        #   books[u, w] += A[w, v]
        #   books[v, w] += A[w, u]
        # Use vectorised update:
        w_idx = np.arange(n)
        w_idx = w_idx[(w_idx != u) & (w_idx != v)]
        books[u, w_idx] += A[w_idx, v]
        books[v, w_idx] += A[w_idx, u]
        # Symmetric: books[w_idx, u] += A[w_idx, v], books[w_idx, v] += A[w_idx, u]
        books[w_idx, u] += A[w_idx, v]
        books[w_idx, v] += A[w_idx, u]
        # books[u, v] unchanged

    def remove_edge(A, books, u, v):
        """Remove edge (u,v) and update books incrementally."""
        w_idx = np.arange(n)
        w_idx = w_idx[(w_idx != u) & (w_idx != v)]
        books[u, w_idx] -= A[w_idx, v]
        books[v, w_idx] -= A[w_idx, u]
        books[w_idx, u] -= A[w_idx, v]
        books[w_idx, v] -= A[w_idx, u]
        A[u, v] = 0
        A[v, u] = 0

    def max_book_from_books(books, A):
        """Return max book size over all edges, using maintained books matrix."""
        # books is n x n; for edge (u,v) book size = books[u,v]
        # Mask non-edges
        masked = np.where(A > 0, books, -1)
        return int(masked.max())

    # ---------- Random initial graph ----------
    def random_graph():
        A = np.zeros((n, n), dtype=np.int8)
        shuffled = all_edges.copy()
        random.shuffle(shuffled)
        for (i, j) in shuffled[:m]:
            A[i, j] = 1
            A[j, i] = 1
        return A

    # ---------- Targeted local search with temperature ----------
    def targeted_sa(A, T0, T1, time_limit):
        """
        Local search that:
        1. Finds an edge with book == current max.
        2. Proposes swapping it with a random non-edge.
        3. Accepts if max decreases, or with probability exp(-delta/T) if same/increase.
        Uses incremental book updates for O(n) per proposal.
        """
        books = make_books(A)
        current_max = max_book_from_books(books, A)
        best_max = current_max
        best_A = A.copy()
        best_books = books.copy()

        # Build edge / non-edge lists
        edge_list = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j]]
        non_edge_list = [(i, j) for i in range(n) for j in range(i + 1, n) if not A[i, j]]
        edge_set = set(edge_list)

        T = T0
        # We'll run until time limit; cooling is adaptive
        # Estimate steps: ~5000 iterations/sec with incremental update
        # Just cool linearly over expected iterations
        # Use a fixed number of "virtual steps" for cooling
        total_virtual = 100000
        virtual = 0
        cooling = (T1 / T0) ** (1.0 / total_virtual) if T1 > 0 else 0.9995

        while time.time() < time_limit:
            virtual += 1
            if virtual % 1000 == 0:
                T = max(T * cooling, 1e-6)

            if len(edge_list) == 0 or len(non_edge_list) == 0:
                break

            # Find edges with book == current_max
            # Vectorised: find all (i,j) with A[i,j]==1 and books[i,j]==current_max
            max_mask = (A > 0) & (books == current_max)
            # Get indices
            idxs = np.argwhere(max_mask)
            # Filter to i < j
            idxs = [(int(r), int(c)) for r, c in idxs if r < c]
            if not idxs:
                # Fallback: pick any edge
                idxs = [edge_list[random.randint(0, len(edge_list) - 1)]]

            u, v = random.choice(idxs)

            # Pick random non-edge
            x, y = random.choice(non_edge_list)

            # Perform swap: remove (u,v), add (x,y)
            remove_edge(A, books, u, v)
            add_edge(A, books, x, y)

            new_max = max_book_from_books(books, A)
            delta = new_max - current_max

            if delta <= 0 or (T > 1e-5 and random.random() < math.exp(-delta / T)):
                # Accept
                current_max = new_max
                # Update lists
                edge_list.remove((u, v))
                edge_list.append((x, y))
                non_edge_list.remove((x, y))
                non_edge_list.append((u, v))

                if current_max < best_max:
                    best_max = current_max
                    best_A = A.copy()
                    best_books = books.copy()
            else:
                # Reject: undo
                remove_edge(A, books, x, y)
                add_edge(A, books, u, v)

            # Early exit if we hit theoretical lower bound
            if best_max <= 7:
                break

        return best_A, best_max

    # ---------- Greedy construction phase ----------
    def greedy_construct():
        """
        Build a graph edge-by-edge, always choosing the edge that
        minimises the resulting max book.  Uses incremental books.
        """
        A = np.zeros((n, n), dtype=np.int8)
        books = np.zeros((n, n), dtype=np.int32)
        remaining = all_edges.copy()
        random.shuffle(remaining)

        for _ in range(m):
            if time.time() > deadline - 30:
                break
            best_edge = None
            best_max = float('inf')
            # Sample up to 80 candidates
            sample = random.sample(remaining, min(80, len(remaining)))
            for (u, v) in sample:
                add_edge(A, books, u, v)
                # Compute max book quickly
                masked = np.where(A > 0, books, -1)
                nm = int(masked.max())
                remove_edge(A, books, u, v)
                if nm < best_max:
                    best_max = nm
                    best_edge = (u, v)
            if best_edge is None:
                best_edge = remaining[0]
            u, v = best_edge
            add_edge(A, books, u, v)
            remaining.remove(best_edge)

        # Fill rest if time ran out
        while len(remaining) > 0 and A.sum() // 2 < m:
            u, v = remaining.pop()
            add_edge(A, books, u, v)

        return A

    # ---------- Main search ----------
    best_A_overall = None
    best_score_overall = float('inf')

    # Phase 1: Greedy construction (fast, good starting point)
    A_greedy = greedy_construct()
    books_g = make_books(A_greedy)
    score_g = max_book_from_books(books_g, A_greedy)
    if score_g < best_score_overall:
        best_score_overall = score_g
        best_A_overall = A_greedy.copy()

    # Phase 2: Targeted SA from greedy start
    if time.time() < deadline - 5:
        A_sa, score_sa = targeted_sa(A_greedy.copy(), T0=1.5, T1=0.01,
                                     time_limit=deadline - 5)
        if score_sa < best_score_overall:
            best_score_overall = score_sa
            best_A_overall = A_sa.copy()

    # Phase 3: Random restart + targeted SA
    if time.time() < deadline - 5:
        A_rand = random_graph()
        A_sa2, score_sa2 = targeted_sa(A_rand, T0=2.0, T1=0.05,
                                       time_limit=deadline - 5)
        if score_sa2 < best_score_overall:
            best_score_overall = score_sa2
            best_A_overall = A_sa2.copy()

    # Phase 4: Final greedy repair on best result
    if best_A_overall is not None and time.time() < deadline - 2:
        A_final = best_A_overall.copy()
        books_f = make_books(A_final)
        cur_max = max_book_from_books(books_f, A_final)
        edge_list = [(i, j) for i in range(n) for j in range(i + 1, n) if A_final[i, j]]
        non_edge_list = [(i, j) for i in range(n) for j in range(i + 1, n) if not A_final[i, j]]

        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for _ in range(500):
                if time.time() > deadline - 2:
                    break
                # Pick edge with max book
                max_mask = (A_final > 0) & (books_f == cur_max)
                idxs = np.argwhere(max_mask)
                idxs = [(int(r), int(c)) for r, c in idxs if r < c]
                if not idxs:
                    break
                u, v = random.choice(idxs)
                x, y = random.choice(non_edge_list)

                remove_edge(A_final, books_f, u, v)
                add_edge(A_final, books_f, x, y)
                new_max = max_book_from_books(books_f, A_final)

                if new_max < cur_max:
                    cur_max = new_max
                    edge_list.remove((u, v))
                    edge_list.append((x, y))
                    non_edge_list.remove((x, y))
                    non_edge_list.append((u, v))
                    if cur_max < best_score_overall:
                        best_score_overall = cur_max
                        best_A_overall = A_final.copy()
                    improved = True
                else:
                    remove_edge(A_final, books_f, x, y)
                    add_edge(A_final, books_f, u, v)

    # ---------- Convert to result ----------
    if best_A_overall is None:
        A_final = random_graph()
    else:
        A_final = best_A_overall

    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if A_final[i, j]:
                result.append([i, j])

    # Verify count
    if len(result) != m:
        A_fb = random_graph()
        result = []
        for i in range(n):
            for j in range(i + 1, n):
                if A_fb[i, j]:
                    result.append([i, j])

    return result

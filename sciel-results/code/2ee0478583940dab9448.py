"""Spectral + local-search design to minimize the largest book (max triangles per edge)."""
import time
import random
import numpy as np

_N = 40
_M = 401


def _init_adj():
    """K20,20 plus one intra-part edge, as adjacency matrix."""
    A = np.zeros((40, 40), dtype=np.int8)
    A[0:20, 20:40] = 1
    A[20:40, 0:20] = 1
    A[0, 1] = 1
    A[1, 0] = 1
    return A


def _edge_list(A):
    edges = []
    for u in range(40):
        for v in range(u + 1, 40):
            if A[u, v]:
                edges.append((u, v))
    return edges


def _books(A):
    """book[e] = number of triangles containing edge e (for each edge index)."""
    edges = _edge_list(A)
    eidx = {}
    for i, (u, v) in enumerate(edges):
        eidx[(u, v)] = i
        eidx[(v, u)] = i
    books = np.zeros(len(edges), dtype=np.int32)
    for i, (u, v) in enumerate(edges):
        # common neighbors of u and v
        books[i] = int(np.sum(A[u] & A[v]))
    return edges, books


def _score(books):
    return int(np.max(books)) if len(books) else 0


def _swap(A, u, v, x, y):
    """Remove edge (u,v) and add edge (x,y). Returns True if valid (no self, no dup)."""
    if u == v or x == y:
        return False
    if A[u, v] == 0:
        return False
    if A[x, y] == 1:
        return False
    A[u, v] = 0
    A[v, u] = 0
    A[x, y] = 1
    A[y, x] = 1
    return True


def _swap_back(A, u, v, x, y):
    A[u, v] = 1
    A[v, u] = 1
    A[x, y] = 0
    A[y, x] = 0


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    rng = random.Random(12345)

    A = _init_adj()
    edges, books = _books(A)
    best_score = _score(books)
    best_A = A.copy()
    best_books = books.copy()

    # Precompute current edge set for fast membership
    def edge_set(A):
        s = set()
        for u in range(40):
            for v in range(u + 1, 40):
                if A[u, v]:
                    s.add((u, v))
        return s

    # Local search: repeatedly try to reduce the max book by swapping edges.
    # Strategy: pick a "hot" edge (high book) and remove it, then add a "cold"
    # non-edge that is least likely to create new triangles.
    def max_book_edge(books, edges):
        return int(np.argmax(books))

    def best_add_edge(A, edge_set):
        # Find a non-edge whose addition creates the fewest new triangles,
        # i.e., min over non-edges of (common neighbors count).
        best = None
        best_val = 10**9
        for u in range(40):
            for v in range(u + 1, 40):
                if A[u, v]:
                    continue
                # triangles created = common neighbors of u and v
                c = int(np.sum(A[u] & A[v]))
                if c < best_val:
                    best_val = c
                    best = (u, v)
        return best, best_val

    # Iterative improvement loop
    no_improve = 0
    max_no_improve = 200000
    while time.time() < deadline and no_improve < max_no_improve:
        edges, books = _books(A)
        cur_score = _score(books)
        if cur_score < best_score:
            best_score = cur_score
            best_A = A.copy()
            best_books = books.copy()
            no_improve = 0
        else:
            no_improve += 1

        # If we've hit the theoretical lower bound, stop.
        if best_score <= 6:
            break

        # Pick a hot edge to remove.
        hi = int(np.argmax(books))
        u, v = edges[hi]
        if books[hi] <= 1:
            # nothing hot to remove; try random perturbations
            # Random swap: remove a random edge, add a random cold non-edge
            eidx = rng.randrange(len(edges))
            u, v = edges[eidx]
            ne, _ = best_add_edge(A, set(map(tuple, edges)))
            if ne is None:
                break
            x, y = ne
            if not _swap(A, u, v, x, y):
                continue
            # evaluate
            e2, b2 = _books(A)
            s2 = _score(b2)
            if s2 <= cur_score:
                # accept (keep)
                continue
            else:
                _swap_back(A, u, v, x, y)
                continue

        # Remove hot edge (u,v); find the best cold non-edge to add.
        # We want the non-edge that, when added, keeps the max book low.
        # Approximate: minimize common-neighbor count of the added edge,
        # but also account for the removal.
        # Compute A after removal temporarily.
        A[u, v] = 0
        A[v, u] = 0
        # Now find best non-edge to add
        best_add = None
        best_add_score = 10**9
        # candidate non-edges: all non-edges
        # To keep it fast, only consider a random sample plus the min-common-neighbor ones
        # First find the global min common-neighbor non-edge
        cand = []
        for uu in range(40):
            for vv in range(uu + 1, 40):
                if A[uu, vv]:
                    continue
                c = int(np.sum(A[uu] & A[vv]))
                cand.append((c, uu, vv))
        cand.sort()
        # Try top K candidates
        K = 30
        tried = 0
        for c, x, y in cand[:K]:
            A[x, y] = 1
            A[y, x] = 1
            e2, b2 = _books(A)
            s2 = _score(b2)
            A[x, y] = 0
            A[y, x] = 0
            if s2 < best_add_score:
                best_add_score = s2
                best_add = (x, y)
            tried += 1
            if time.time() > deadline - 5:
                break
        if best_add is not None:
            x, y = best_add
            A[x, y] = 1
            A[y, x] = 1
            # commit: A now has removed (u,v) and added (x,y)
            # check if it improved
            e2, b2 = _books(A)
            s2 = _score(b2)
            if s2 <= cur_score:
                # accept
                continue
            else:
                # revert: re-add (u,v), remove (x,y)
                A[u, v] = 1
                A[v, u] = 1
                A[x, y] = 0
                A[y, x] = 0
                continue
        else:
            # no good add found; restore
            A[u, v] = 1
            A[v, u] = 1
            continue

    # Final: use best found
    # Verify edge count
    Aout = best_A
    edges_out = _edge_list(Aout)
    # Ensure exactly 401 edges
    if len(edges_out) != 401:
        # fallback to initial
        Aout = _init_adj()
        edges_out = _edge_list(Aout)
    return [[u, v] for (u, v) in edges_out]

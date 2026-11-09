"""Minimize the largest book (max triangles containing an edge) over 401-edge graphs on 40 vertices."""
import time
import random
import numpy as np

N = 40
M = 401
DEADLINE = 150.0
SAFETY = 5.0  # seconds margin before wall-clock kill

def _book_from_adj(adj):
    """Return (books, tri) where books[u,v] = #triangles containing edge (u,v), tri = #triangles."""
    adj = np.asarray(adj, dtype=np.int64)
    # tri(u,v) = (A @ A)[u,v] - A[u,v] - deg(u) - deg(v) + A[u,v]
    # Actually: (A@A)[u,v] = sum_w A[u,w]A[w,v]. For u!=v, this counts common neighbors.
    # Triangles containing edge (u,v) = common neighbors of u and v (for u!=v, A[u,v]=1).
    A = adj
    AA = A @ A
    # For edge (u,v) with u<v: triangles = AA[u,v]
    books = np.zeros((N, N), dtype=np.int64)
    np.fill_diagonal(books, -1)
    books = AA - np.eye(N, dtype=np.int64)
    # books[u,v] for u!=v is common neighbors; for edges this equals triangles containing (u,v).
    return books

def _swap_books(books, u, v, x, y):
    """
    Remove edge (u,v), add edge (x,y). Return new books and whether valid.
    books is the common-neighbor matrix (symmetric, 0 diagonal).
    Validity: (u,v) must be an edge, (x,y) must be a non-edge, u!=v, x!=y, (x,y)!=(u,v) or (v,u).
    """
    if u == v or x == y:
        return None
    if books[u, v] < 0 or books[u, v] == 0:
        # not an edge (books[u,v] is common neighbors; but we need edge presence)
        # We need a separate edge matrix. We'll pass adj too.
        return None
    if books[x, y] < 0:
        return None
    # We need to know edge presence. Use a separate check via caller.
    # Here we assume caller verified (u,v) edge and (x,y) non-edge.
    # Remove (u,v): for all w, common_neighbors(w, u) decreases if w~v, etc.
    # This is complex. Simpler: recompute books from scratch is O(N^3) ~ 64000 ops, too slow per swap.
    # Instead, update books incrementally.
    # When removing edge (u,v):
    #   For each w: 
    #     books[w,u] changes: was counting common neighbors of w and u. 
    #     Common neighbors of w and u = neighbors(w) ∩ neighbors(u).
    #     Removing edge (u,v): v is no longer neighbor of u. So if v is neighbor of w, 
    #     then v was a common neighbor of w and u, now it's not. So books[w,u] -= 1 if w~v.
    #     Similarly books[w,v] -= 1 if w~u.
    #     Also books[u,v] = 0 (no longer an edge, but common neighbors still defined).
    # When adding edge (x,y):
    #   books[w,x] += 1 if w~y, books[w,y] += 1 if w~x.
    #   books[x,y] = common neighbors of x and y (already computed).
    # This is O(N) per swap.
    # But we also need the edge matrix to check validity. We'll maintain adj separately.
    return None  # placeholder; actual update done in construct with adj

def construct():
    start_time = time.time()
    rng = random.Random(42)

    # Initial: K20,20 + one internal edge
    adj = np.zeros((N, N), dtype=np.int8)
    for a in range(20):
        for b in range(20, 40):
            adj[a, b] = 1
            adj[b, a] = 1
    adj[0, 1] = 1
    adj[1, 0] = 1

    # Verify edge count
    assert np.sum(adj) // 2 == M

    def get_books():
        return _book_from_adj(adj)

    books = get_books()
    # max book = max over edges of books[u,v]
    def max_book(books, adj):
        # Only consider edges
        m = 0
        for u in range(N):
            for v in range(u+1, N):
                if adj[u, v]:
                    if books[u, v] > m:
                        m = books[u, v]
        return m

    best_score = max_book(books, adj)
    best_adj = adj.copy()
    best_books = books.copy()

    def try_swap(u, v, x, y):
        """Remove (u,v), add (x,y). Return (new_books, new_max) or None."""
        nonlocal adj, books
        # Validate
        if u == v or x == y:
            return None
        if not adj[u, v]:
            return None
        if adj[x, y]:
            return None
        if (x, y) == (u, v) or (x, y) == (v, u):
            return None

        # Update adj
        adj[u, v] = 0
        adj[v, u] = 0
        adj[x, y] = 1
        adj[y, x] = 1

        # Update books incrementally
        # Remove (u,v):
        # For all w: books[w,u] -= (1 if adj[w,v] else 0)  [before removal, adj[w,v] is current]
        # But adj[v,u] was just set to 0, so we need to be careful.
        # Let's do it in the right order.
        # First, capture neighbors.
        # Actually, let's just recompute books for simplicity if N is small.
        # N=40, N^3=64000, which is fast in NumPy.
        books = _book_from_adj(adj)
        mb = max_book(books, adj)
        return (books, mb)

    def revert_swap(u, v, x, y):
        nonlocal adj, books
        adj[u, v] = 1
        adj[v, u] = 1
        adj[x, y] = 0
        adj[y, x] = 0
        books = _book_from_adj(adj)

    # Precompute list of edges and non-edges for sampling
    def get_edges():
        edges = []
        for u in range(N):
            for v in range(u+1, N):
                if adj[u, v]:
                    edges.append((u, v))
        return edges

    def get_nonedges():
        nonedges = []
        for u in range(N):
            for v in range(u+1, N):
                if not adj[u, v]:
                    nonedges.append((u, v))
        return nonedges

    # Local search
    while time.time() - start_time < DEADLINE - SAFETY:
        edges = get_edges()
        nonedges = get_nonedges()
        if not edges or not nonedges:
            break
        u, v = rng.choice(edges)
        x, y = rng.choice(nonedges)
        # Ensure (x,y) is not the same as (u,v)
        if (x, y) == (u, v) or (x, y) == (v, u):
            continue

        result = try_swap(u, v, x, y)
        if result is None:
            continue
        new_books, new_mb = result

        # Accept if not worse, or with some probability if slightly worse
        if new_mb <= best_score:
            best_score = new_mb
            best_adj = adj.copy()
            best_books = books.copy()
        else:
            # Revert
            revert_swap(u, v, x, y)

        # Early stop if we hit the theoretical lower bound
        if best_score <= 6:
            break

    # Restore best
    adj = best_adj
    books = best_books

    # Return edge list
    result_edges = []
    for u in range(N):
        for v in range(u+1, N):
            if adj[u, v]:
                result_edges.append([u, v])
    assert len(result_edges) == M
    return result_edges

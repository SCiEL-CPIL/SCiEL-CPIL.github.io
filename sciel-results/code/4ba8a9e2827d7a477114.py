"""Rewritten program: simulated annealing to minimize the largest book."""
import time
import random
import numpy as np

N = 40
TARGET_EDGES = 401

def _init_adj():
    adj = np.zeros((N, N), dtype=bool)
    for a in range(20):
        for b in range(20, 40):
            adj[a, b] = adj[b, a] = True
    adj[0, 1] = adj[1, 0] = True
    return adj

def _count_books(adj):
    books = np.zeros((N, N), dtype=np.int32)
    for u in range(N):
        for v in range(u + 1, N):
            if adj[u, v]:
                common = int(np.count_nonzero(adj[u] & adj[v]))
                books[u, v] = books[v, u] = common
    return books

def _max_book(books):
    return int(np.max(books))

def construct():
    deadline = time.time() + 140.0
    rng = random.Random(12345)

    adj = _init_adj()
    books = _count_books(adj)
    cur_max = _max_book(books)

    best_adj = adj.copy()
    best_books = books.copy()
    best_max = cur_max

    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    cur_edges = set((u, v) for u in range(N) for v in range(u + 1, N) if adj[u, v])
    assert len(cur_edges) == TARGET_EDGES

    T = 5.0
    T_min = 0.05
    cool = 0.9995

    def apply_swap(u, v, a, b):
        """Remove (u,v), add (a,b). Update books in place. Returns nothing."""
        # Compute common neighbors before modification
        common_uv = np.where(adj[u] & adj[v])[0]
        common_ab = np.where(adj[a] & adj[b])[0]

        # Remove (u,v)
        adj[u, v] = adj[v, u] = False
        for w in common_uv:
            if w == u or w == v:
                continue
            if adj[u, w]:
                books[u, w] -= 1
                books[w, u] -= 1
            if adj[v, w]:
                books[v, w] -= 1
                books[w, v] -= 1
        books[u, v] = books[v, u] = 0

        # Add (a,b)
        adj[a, b] = adj[b, a] = True
        for w in common_ab:
            if w == a or w == b:
                continue
            if adj[a, w]:
                books[a, w] += 1
                books[w, a] += 1
            if adj[b, w]:
                books[b, w] += 1
                books[w, b] += 1
        books[a, b] = books[b, a] = int(len(common_ab))

    while time.time() < deadline:
        books_snapshot = books.copy()
        adj_snapshot = adj.copy()

        u, v = rng.choice(list(cur_edges))
        while True:
            a, b = rng.choice(all_edges)
            if (a, b) not in cur_edges:
                break

        apply_swap(u, v, a, b)
        cur_edges.discard((u, v))
        cur_edges.add((a, b))

        new_max = _max_book(books)
        delta = new_max - cur_max
        accept = (delta <= 0) or (rng.random() < np.exp(-delta / max(T, 1e-9)))
        if accept:
            cur_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_adj = adj.copy()
                best_books = books.copy()
        else:
            # Revert
            books[:] = books_snapshot
            adj[:] = adj_snapshot
            cur_edges.discard((a, b))
            cur_edges.add((u, v))

        T = max(T * cool, T_min)

    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj[u, v]:
                edges.append([u, v])
    if len(edges) != TARGET_EDGES:
        edges = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    return edges

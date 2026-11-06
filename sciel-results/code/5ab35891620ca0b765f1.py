"""Construct a 40-vertex, 401-edge graph minimizing the largest book size."""
import time
import random
import numpy as np

N = 40
TARGET_EDGES = 401
DEADLINE_SECONDS = 150
SAFETY_MARGIN = 10.0


def _compute_books_detail(adj):
    """Return (max_book, common) where common[u,v] is book size for edge (u,v)."""
    common = adj @ adj
    max_book = int(np.max(common[adj]))
    return max_book, common


def _rand_graph_balanced():
    """
    Generate a random graph close to K20,20 plus a few extra edges.
    Start with a complete bipartite graph with parts of size 20 each, then add 1 edge.
    """
    adj = np.zeros((N, N), dtype=np.uint8)
    for i in range(20):
        for j in range(20, 40):
            adj[i, j] = 1
            adj[j, i] = 1
    # That's 400 edges. Add one more random edge within a part.
    if random.random() < 0.5:
        u = random.randint(0, 19)
        v = random.randint(0, 19)
        if u != v:
            adj[u, v] = 1
            adj[v, u] = 1
        else:
            adj[0, 1] = 1
            adj[1, 0] = 1
    else:
        u = random.randint(20, 39)
        v = random.randint(20, 39)
        if u != v:
            adj[u, v] = 1
            adj[v, u] = 1
        else:
            adj[20, 21] = 1
            adj[21, 20] = 1
    return adj


def _try_swap(adj, current_max_book_val):
    """
    Try a single edge swap: remove an edge (u,v) and add an edge (x,y) that is not present.
    Return (new_adj, new_common, new_max_book_val, accepted).
    """
    # Find all edges (upper triangle)
    edges = np.argwhere(adj == 1)
    edges = edges[edges[:, 0] < edges[:, 1]]
    if len(edges) == 0:
        return adj, None, current_max_book_val, False

    # Pick a random edge to remove
    idx = random.randint(0, len(edges) - 1)
    u, v = edges[idx]

    # Find all non-edges (potential edges to add), upper triangle only
    non_edges = np.argwhere(adj == 0)
    non_edges = non_edges[non_edges[:, 0] < non_edges[:, 1]]
    if len(non_edges) == 0:
        return adj, None, current_max_book_val, False

    # Pick a random non-edge to add
    jdx = random.randint(0, len(non_edges) - 1)
    x, y = non_edges[jdx]

    # Perform the swap
    new_adj = adj.copy()
    new_adj[u, v] = 0
    new_adj[v, u] = 0
    new_adj[x, y] = 1
    new_adj[y, x] = 1

    # Compute new common and max_book
    new_common = new_adj @ new_adj
    new_max_book_val = int(np.max(new_common[new_adj]))

    return new_adj, new_common, new_max_book_val, True


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS - SAFETY_MARGIN

    # Start with a balanced random graph
    best_adj = _rand_graph_balanced()
    best_max_book_val, best_common = _compute_books_detail(best_adj)

    # Simulated annealing
    current_adj = best_adj.copy()
    current_max_book_val = best_max_book_val
    current_common = best_common

    T = 10.0  # Initial temperature
    cooling = 0.999

    while time.time() < deadline:
        # Try a swap
        new_adj, new_common, new_max_book_val, accepted = _try_swap(current_adj, current_max_book_val)
        if not accepted:
            continue

        delta = new_max_book_val - current_max_book_val

        if delta <= 0:
            # Accept
            current_adj = new_adj
            current_common = new_common
            current_max_book_val = new_max_book_val
            if current_max_book_val < best_max_book_val:
                best_adj = current_adj.copy()
                best_max_book_val = current_max_book_val
                best_common = current_common
        else:
            # Accept with probability exp(-delta / T)
            prob = np.exp(-delta / T)
            if random.random() < prob:
                current_adj = new_adj
                current_common = new_common
                current_max_book_val = new_max_book_val

        T *= cooling

    # Return the best graph found
    # Convert best_adj to list of edges
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj[u, v] == 1:
                edges.append([u, v])

    # Ensure exactly TARGET_EDGES edges
    assert len(edges) == TARGET_EDGES, f"Expected {TARGET_EDGES} edges, got {len(edges)}"

    return edges

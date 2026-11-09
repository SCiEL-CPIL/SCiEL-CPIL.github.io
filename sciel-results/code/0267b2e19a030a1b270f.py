import numpy as np
import time
import random

def construct():
    n = 40
    m = 401
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 120s kill

    def get_score(adj):
        # Vectorized computation of common neighbors for all pairs
        common = np.dot(adj, adj.T)
        # Set diagonal to 0
        np.fill_diagonal(common, 0)
        # Only consider edges where adj[i,j] == 1
        max_book = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1:
                    if common[i, j] > max_book:
                        max_book = common[i, j]
        return max_book

    def random_graph():
        edges = set()
        while len(edges) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                if u > v:
                    u, v = v, u
                edges.add((u, v))
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        return adj, list(edges)

    def greedy_construction():
        """Greedy construction that tries to minimize book size at each step."""
        adj = np.zeros((n, n), dtype=np.uint8)
        edges = []
        remaining = m

        # Start with a near-bipartite structure
        # Partition vertices into two sets of 20
        # Add cross edges first, then add intra edges carefully

        # First, add all cross edges between partitions (400 edges)
        for i in range(20):
            for j in range(20, 40):
                adj[i, j] = 1
                adj[j, i] = 1
                edges.append((i, j))
                remaining -= 1

        # Now add 1 intra edge
        # Choose an intra edge that minimizes the book size
        best_edge = None
        best_score = 100

        for i in range(20):
            for j in range(i + 1, 20):
                if adj[i, j] == 0:
                    # Tentatively add this edge
                    adj[i, j] = 1
                    adj[j, i] = 1
                    score = get_score(adj)
                    if score < best_score:
                        best_score = score
                        best_edge = (i, j)
                    # Remove it
                    adj[i, j] = 0
                    adj[j, i] = 0

        if best_edge is not None:
            adj[best_edge[0], best_edge[1]] = 1
            adj[best_edge[1], best_edge[0]] = 1
            edges.append(best_edge)

        return adj, edges

    def local_search(adj, edges, time_limit):
        """Local search to improve the graph."""
        current_score = get_score(adj)
        best_score = current_score
        best_adj = adj.copy()
        best_edges = edges[:]

        t_end = time.time() + time_limit

        while time.time() < t_end and time.time() < deadline:
            # Pick a random edge to remove
            out_idx = random.randint(0, len(edges) - 1)
            out_u, out_v = edges[out_idx]

            # Pick a random non-edge to add
            while True:
                in_u = random.randint(0, n - 1)
                in_v = random.randint(0, n - 1)
                if in_u == in_v:
                    continue
                if in_u > in_v:
                    in_u, in_v = in_v, in_u
                if (in_u, in_v) == (out_u, out_v):
                    continue
                if adj[in_u, in_v] == 0:
                    break

            # Apply swap
            adj[out_u, out_v] = 0
            adj[out_v, out_u] = 0
            edges[out_idx] = (in_u, in_v)

            adj[in_u, in_v] = 1
            adj[in_v, in_u] = 1

            new_score = get_score(adj)

            if new_score <= best_score:
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj.copy()
                    best_edges = edges[:]
                    if best_score == 7:
                        return best_adj, best_edges
            else:
                # Revert with some probability
                if random.random() < 0.3:
                    adj[in_u, in_v] = 0
                    adj[in_v, in_u] = 0
                    adj[out_u, out_v] = 1
                    adj[out_v, out_u] = 1
                    edges[out_idx] = (out_u, out_v)

        return best_adj, best_edges

    best_score = 100
    best_adj = None
    best_edges = None

    # Try greedy construction first
    adj, edges = greedy_construction()
    score = get_score(adj)
    if score < best_score:
        best_score = score
        best_adj = adj.copy()
        best_edges = edges[:]

    if best_score == 7:
        return [[u, v] for u, v in best_edges]

    # Run local search with time limit
    time_per_search = 20.0
    num_searches = 5

    for s in range(num_searches):
        if time.time() >= deadline - 5:
            break

        if s == 0:
            current_adj, current_edges = best_adj.copy(), best_edges[:]
        else:
            current_adj, current_edges = random_graph()

        current_adj, current_edges = local_search(current_adj, current_edges, time_per_search)
        current_score = get_score(current_adj)

        if current_score < best_score:
            best_score = current_score
            best_adj = current_adj.copy()
            best_edges = current_edges[:]
            if best_score == 7:
                return [[u, v] for u, v in best_edges]

    return [[u, v] for u, v in best_edges]

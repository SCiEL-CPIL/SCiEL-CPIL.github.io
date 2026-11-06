import random
import time
import math

def construct():
    deadline = time.time() + 115  # safety margin
    n = 16
    target_edges = 65

    def score_graph(adj_mask, deg):
        """Compute the max triangle degree-sum using bitmask adjacency.
        adj_mask[i] is an integer where bit j is set if edge (i,j) exists."""
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_mask[u] >> v) & 1:
                    continue
                # Common neighbors of u and v
                common = adj_mask[u] & adj_mask[v]
                # Only consider w > v to avoid double counting
                common &= ~((1 << (v + 1)) - 1)
                while common:
                    w = common.bit_length() - 1
                    common ^= (1 << w)
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
        return max_sum

    def edges_to_mask(edges):
        """Convert edge list to bitmask representation and compute degrees."""
        adj_mask = [0] * n
        deg = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj_mask, deg

    def mask_to_edges(adj_mask):
        """Convert bitmask representation back to edge list."""
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def init_edges():
        return [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]

    best_edges = init_edges()
    best_adj, best_deg = edges_to_mask(best_edges)
    best_score = score_graph(best_adj, best_deg)

    def try_swap(u, v, x, y, adj_mask, deg):
        """Try adding edge (u,v) and removing edge (x,y). Returns (new_adj, new_deg, new_score) or None."""
        if u >= v or (adj_mask[u] >> v) & 1:
            return None
        if x >= y or not (adj_mask[x] >> y) & 1:
            return None

        # Apply swap
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        adj_mask[x] &= ~(1 << y)
        adj_mask[y] &= ~(1 << x)
        deg[u] += 1; deg[v] += 1
        deg[x] -= 1; deg[y] -= 1

        new_score = score_graph(adj_mask, deg)

        # Undo
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        adj_mask[x] |= (1 << y)
        adj_mask[y] |= (1 << x)
        deg[u] -= 1; deg[v] -= 1
        deg[x] += 1; deg[y] += 1

        if new_score < best_score:
            return adj_mask, deg, new_score
        return None

    # Gradient-guided initialization: use a soft partition vector to guide initial swaps
    # The idea: treat vertices as having a continuous "side" value, and prefer swaps
    # that align with a Max-Cut-like objective to reduce triangle concentration.
    # We use a simple heuristic: vertices with high degree in triangles should be
    # "pushed apart" by adding edges between them and removing edges within clusters.

    # Local search with bitmask speedup
    while time.time() < deadline:
        adj_curr = best_adj[:]
        deg_curr = best_deg[:]
        improved = False

        # Full scan of all possible swaps (deterministic)
        for u in range(n):
            if time.time() > deadline:
                break
            for v in range(u + 1, n):
                if (adj_curr[u] >> v) & 1:
                    continue  # Only add non-edges
                for x in range(n):
                    for y in range(x + 1, n):
                        if not (adj_curr[x] >> y) & 1:
                            continue  # Only remove existing edges
                        if (u == x and v == y) or (u == y and v == x):
                            continue

                        result = try_swap(u, v, x, y, adj_curr, deg_curr)
                        if result is not None:
                            adj_curr, deg_curr, best_score = result
                            best_adj = adj_curr[:]
                            best_deg = deg_curr[:]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            # Random restart with perturbation
            if time.time() > deadline:
                break
            adj_pert = best_adj[:]
            deg_pert = best_deg[:]
            for _ in range(5):
                # Pick random non-edge to add
                while True:
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u < v and not (adj_pert[u] >> v) & 1:
                        break
                # Pick random edge to remove
                while True:
                    x = random.randint(0, n - 1)
                    y = random.randint(0, n - 1)
                    if x < y and (adj_pert[x] >> y) & 1:
                        break

                # Apply swap
                adj_pert[u] |= (1 << v)
                adj_pert[v] |= (1 << u)
                adj_pert[x] &= ~(1 << y)
                adj_pert[y] &= ~(1 << x)
                deg_pert[u] += 1; deg_pert[v] += 1
                deg_pert[x] -= 1; deg_pert[y] -= 1

            # Run local search on perturbed state
            adj_curr = adj_pert[:]
            deg_curr = deg_pert[:]
            for _ in range(10):  # Limited local search on perturbed state
                if time.time() > deadline:
                    break
                improved = False
                for u in range(n):
                    if time.time() > deadline:
                        break
                    for v in range(u + 1, n):
                        if (adj_curr[u] >> v) & 1:
                            continue
                        for x in range(n):
                            for y in range(x + 1, n):
                                if not (adj_curr[x] >> y) & 1:
                                    continue
                                if (u == x and v == y) or (u == y and v == x):
                                    continue
                                result = try_swap(u, v, x, y, adj_curr, deg_curr)
                                if result is not None:
                                    adj_curr, deg_curr, new_score = result
                                    if new_score < best_score:
                                        best_adj = adj_curr[:]
                                        best_deg = deg_curr[:]
                                        best_score = new_score
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    break

    best_edges = mask_to_edges(best_adj)
    return {'n': 16, 'edges': best_edges}

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _popcount(x):
    return bin(x).count('1')

def _ctz(x):
    if x == 0:
        return -1
    return (x & -x).bit_length() - 1

def _has_independent_set_10(n, adj_mask):
    """Return True iff there exists an independent set of size 10.
    Uses bitmask branch-and-bound."""
    all_mask = (1 << n) - 1

    def dfs(candidates, depth):
        if depth >= 10:
            return True
        if not candidates:
            return False
        if depth + _popcount(candidates) < 10:
            return False

        # Pick vertex with max degree in induced subgraph to prune include branch
        best_v = -1
        max_deg = -1
        c = candidates
        while c:
            v = _ctz(c)
            c &= c - 1
            deg = _popcount(adj_mask[v] & candidates)
            if deg > max_deg:
                max_deg = deg
                best_v = v

        v = best_v
        v_bit = 1 << v

        # Branch 1: include v
        new_cand = candidates & ~adj_mask[v] & ~v_bit
        if dfs(new_cand, depth + 1):
            return True

        # Branch 2: exclude v
        new_cand = candidates & ~v_bit
        if dfs(new_cand, depth):
            return True

        return False

    return dfs(all_mask, 0)

def _is_triangle_free(n, adj_mask):
    """Check if graph is triangle-free using adjacency bitmasks."""
    for u in range(n):
        for v in range(u + 1, n):
            if (adj_mask[u] >> v) & 1:
                if adj_mask[u] & adj_mask[v]:
                    return False
    return True

def _validate(n, edges):
    """Check both conditions."""
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    if not _is_triangle_free(n, adj_mask):
        return False
    if _has_independent_set_10(n, adj_mask):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    # Start with the known Andrasfai graph on 26 vertices
    # It is triangle-free and has alpha = 9 (no independent set of size 10)
    n0 = 26
    edges0 = [[i, j] for i in range(n0) for j in range(i + 1, n0) if (j - i) % n0 % 3 == 1]
    best_n = n0
    best_edges = edges0

    # Verify the seed
    if not _validate(best_n, best_edges):
        # Fallback to a trivial valid graph if the seed somehow fails
        best_n = 1
        best_edges = []

    # Try to improve by searching for larger graphs
    # We know R(3,10) <= 41, so max n is 40.
    # Try random triangle-free graphs with local search / simulated annealing
    # or just random generation + check.
    
    random.seed(42)

    # Strategy: For each target n from 40 down to 27, try to find a valid graph.
    # Use a greedy random edge addition with triangle avoidance.
    # Then check for independent set of size 10.
    
    for target_n in range(40, 26, -1):
        if time.time() > deadline - 10:
            break

        found = False
        max_attempts = 500
        attempts = 0

        while attempts < max_attempts and not found:
            attempts += 1
            if time.time() > deadline - 10:
                break

            n = target_n
            adj_mask = [0] * n
            edges = []

            # Generate random triangle-free graph
            # Randomly order potential edges and add with probability p
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_edges)

            # Adjust density based on n
            # For larger n, we need denser graphs to keep alpha small
            p = 0.3 + 0.2 * (n - 26) / 14.0  # density between 0.3 and 0.5

            for u, v in all_edges:
                if random.random() < p:
                    # Check if adding (u,v) creates a triangle
                    if adj_mask[u] & adj_mask[v]:
                        continue
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    edges.append([u, v])

            # Check independent set
            if not _has_independent_set_10(n, adj_mask):
                best_n = n
                best_edges = edges
                found = True
                break

        if found:
            break

    # If we found a graph, return it. Otherwise, return the best known (26).
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END

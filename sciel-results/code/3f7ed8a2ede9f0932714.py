"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    def build_adj_matrices(colors, n):
        """Build 4 adjacency matrices (numpy bool arrays) from color list."""
        num_edges = n * (n - 1) // 2
        adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i, j] = True
                adj[c][j, i] = True
                idx += 1
        return adj

    def find_c6(adj, n):
        """Find a C6 in graph given as numpy bool adjacency matrix.
        Returns the 6-tuple of vertices or None."""
        # For each pair (v0, v1) with v0 < v1, try to extend to a 6-cycle
        for v0 in range(n):
            for v1 in range(v0 + 1, n):
                if not adj[v0, v1]:
                    continue
                for v2 in range(v1 + 1, n):
                    if not adj[v1, v2]:
                        continue
                    for v3 in range(v2 + 1, n):
                        if not adj[v2, v3]:
                            continue
                        for v4 in range(v3 + 1, n):
                            if not adj[v3, v4]:
                                continue
                            for v5 in range(v4 + 1, n):
                                if not adj[v4, v5]:
                                    continue
                                if adj[v5, v0]:
                                    return (v0, v1, v2, v3, v4, v5)
        return None

    def check_no_c6(colors, n):
        """Verify no color class contains a C6."""
        adj = build_adj_matrices(colors, n)
        for c in range(4):
            if find_c6(adj[c], n) is not None:
                return False
        return True

    def count_c6s_fast(adj, n):
        """Count C6s in a single color class using numpy adjacency."""
        count = 0
        for v0 in range(n):
            for v1 in range(v0 + 1, n):
                if not adj[v0, v1]:
                    continue
                for v2 in range(v1 + 1, n):
                    if not adj[v1, v2]:
                        continue
                    for v3 in range(v2 + 1, n):
                        if not adj[v2, v3]:
                            continue
                        for v4 in range(v3 + 1, n):
                            if not adj[v3, v4]:
                                continue
                            for v5 in range(v4 + 1, n):
                                if not adj[v4, v5] or not adj[v5, v0]:
                                    continue
                                count += 1
        return count

    def total_c6_count(colors, n):
        """Count total C6s across all 4 color classes."""
        adj = build_adj_matrices(colors, n)
        total = 0
        for c in range(4):
            total += count_c6s_fast(adj[c], n)
        return total

    def targeted_local_search(n, time_limit, max_restarts=5):
        """Targeted local search: find C6s, break them by recoloring edges."""
        num_edges = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        edge_idx = {}
        for idx, (u, v) in enumerate(edges):
            edge_idx[(u, v)] = idx
            edge_idx[(v, u)] = idx

        best_colors = None
        best_cost = float('inf')

        for restart in range(max_restarts):
            if time.time() > time_limit:
                break

            # Random initial coloring
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            adj = build_adj_matrices(colors, n)
            cost = total_c6_count(colors, n)

            # Local search: repeatedly find a C6 and break it
            stuck_count = 0
            T = 5.0
            while time.time() < time_limit and cost > 0:
                if stuck_count > 500:
                    break
                stuck_count += 1

                # Find a C6 in a random color class that has one
                c6_found = None
                for c in random.sample(range(4), 4):
                    c6 = find_c6(adj[c], n)
                    if c6 is not None:
                        c6_found = (c, c6)
                        break

                if c6_found is None:
                    cost = 0
                    break

                c, cycle = c6_found
                # Get the 6 edges of the cycle
                cycle_edges = []
                for k in range(6):
                    u = cycle[k]
                    v = cycle[(k + 1) % 6]
                    cycle_edges.append(edge_idx[(u, v)])

                # Pick a random edge from the cycle and recolor it
                eidx = random.choice(cycle_edges)
                old_c = colors[eidx]
                new_c = random.choice([c for c in range(4) if c != old_c])
                u, v = edges[eidx]

                # Update adjacency
                adj[old_c][u, v] = False
                adj[old_c][v, u] = False
                adj[new_c][u, v] = True
                adj[new_c][v, u] = True
                colors[eidx] = new_c

                new_cost = total_c6_count(colors, n)
                delta = new_cost - cost

                if delta <= 0 or random.random() < np.exp(-delta / T):
                    cost = new_cost
                    stuck_count = 0
                else:
                    # Revert
                    adj[new_c][u, v] = False
                    adj[new_c][v, u] = False
                    adj[old_c][u, v] = True
                    adj[old_c][v, u] = True
                    colors[eidx] = old_c

                T = max(0.1, T * 0.995)

            if cost < best_cost:
                best_cost = cost
                best_colors = colors[:]
                if cost == 0:
                    break

        return best_colors, best_cost

    # Try n=18 first
    if time.time() < deadline - 50:
        print("Trying n=18 with targeted local search...", flush=True)
        colors, cost = targeted_local_search(18, deadline - 50, max_restarts=3)
        if cost == 0 and check_no_c6(colors, 18):
            print("Found n=18 coloring!", flush=True)
            return {'n': 18, 'colors': colors}

    # Try n=17
    if time.time() < deadline - 30:
        print("Trying n=17 with targeted local search...", flush=True)
        colors, cost = targeted_local_search(17, deadline - 30, max_restarts=10)
        if cost == 0 and check_no_c6(colors, 17):
            print("Found n=17 coloring!", flush=True)
            return {'n': 17, 'colors': colors}

    # Try n=16
    if time.time() < deadline - 15:
        print("Trying n=16 with targeted local search...", flush=True)
        colors, cost = targeted_local_search(16, deadline - 15, max_restarts=20)
        if cost == 0 and check_no_c6(colors, 16):
            print("Found n=16 coloring!", flush=True)
            return {'n': 16, 'colors': colors}

    # Fallback: small n with cyclic construction
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            c = min(d, n - d) - 1
            if c < 0:
                c = 0
            colors.append(c)
    if check_no_c6(colors, n):
        return {'n': n, 'colors': colors}

    # Ultimate fallback
    n = 5
    colors = [0] * (n * (n - 1) // 2)
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END

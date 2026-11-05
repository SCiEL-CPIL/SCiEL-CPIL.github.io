"""
4-coloring of K_n with no monochromatic triangle (R(3,3,3,3)).

Strategy:
  1. Build the known K_32 seed (two Clebsch copies, inter-copy edges color 3).
  2. Use fast bitmask-based monochromatic-triangle counting.
  3. Use greedy initialization that avoids creating triangles during construction.
  4. Run Simulated Annealing with restarts on the full edge-coloring to minimize
     triangles for target n (50, 48, ...). If we reach zero triangles, return immediately.
  5. If SA fails for a target, fall back to the best valid coloring found.

The bitmask approach gives O(1) triangle detection per edge flip, which is far
faster than the O(n^2) per-vertex check. This allows many more SA iterations
within the 120 s budget.
"""

import time
import random
import numpy as np


def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin before 120 s kill

    random.seed(42)
    np.random.seed(42)

    # ---------- GF(16) helpers for the Clebsch seed ----------
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    def build_seed_32():
        """Return list of colors for K_32 in lexicographic edge order."""
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(
                    log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3
                )
        return colors

    # ---------- Fast triangle counting with bitmasks ----------
    def count_triangles(adj, n):
        """Count total monochromatic triangles given adjacency bitmasks."""
        total = 0
        for c in range(4):
            for i in range(n):
                ai = adj[c][i]
                mask = ai & ~((1 << (i + 1)) - 1)  # only j > i
                while mask:
                    lb = mask & (-mask)
                    j = lb.bit_length() - 1
                    total += (adj[c][i] & adj[c][j]).bit_count()
                    mask ^= lb
        return total // 3

    # ---------- Greedy initialization ----------
    def greedy_init(n):
        """Greedy initialization that tries to avoid triangles."""
        adj = [[0] * n for _ in range(4)]
        colors = [-1] * (n * (n - 1) // 2)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Try each color, pick the one that creates fewest new triangles
                best_k = -1
                best_delta = float('inf')
                for k in range(4):
                    # Count triangles that would be created by adding edge (i,j) with color k
                    common = adj[k][i] & adj[k][j]
                    delta = common.bit_count()
                    if delta < best_delta or (delta == best_delta and random.random() < 0.5):
                        best_delta = delta
                        best_k = k
                
                idx = i * n - i * (i + 1) // 2 + (j - i - 1)
                colors[idx] = best_k
                adj[best_k][i] |= (1 << j)
                adj[best_k][j] |= (1 << i)
        
        tri = count_triangles(adj, n)
        return adj, colors, tri

    # ---------- Random initialization ----------
    def random_init(n):
        """Random initialization."""
        adj = [[0] * n for _ in range(4)]
        colors = [-1] * (n * (n - 1) // 2)
        
        for i in range(n):
            for j in range(i + 1, n):
                k = random.randint(0, 3)
                idx = i * n - i * (i + 1) // 2 + (j - i - 1)
                colors[idx] = k
                adj[k][i] |= (1 << j)
                adj[k][j] |= (1 << i)
        
        tri = count_triangles(adj, n)
        return adj, colors, tri

    # ---------- SA with incremental triangle counting ----------
    def sa_color(adj, colors, n, time_limit, T_init=1.5, T_min=0.05, cooling_rate=0.9997):
        """
        Run simulated annealing to minimize monochromatic triangles.
        Returns (colors_list, n_triangles) or (None, None) on timeout.
        """
        tri = count_triangles(adj, n)
        if tri == 0:
            return colors[:], 0

        T = T_init
        best_tri = tri
        best_colors = colors[:]
        best_adj = [row[:] for row in adj]

        # Precompute edge index mapping for fast lookup
        edge_to_ij = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_to_ij[idx] = (i, j)
                idx += 1

        n_edges = n * (n - 1) // 2

        while time.time() < time_limit:
            if tri == 0:
                break

            # Pick a random edge
            ei = random.randint(0, n_edges - 1)
            i, j = edge_to_ij[ei]

            c_old = colors[ei]
            c_new = (c_old + 1 + random.randint(0, 2)) % 4

            # Compute delta in triangle count
            k_old = (adj[c_old][i] & adj[c_old][j]).bit_count()
            k_new = (adj[c_new][i] & adj[c_new][j]).bit_count()
            delta = k_new - k_old

            # Acceptance
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = True  # accept neutral moves to explore
            else:
                if T > T_min:
                    p = np.exp(-delta / T)
                    if random.random() < p:
                        accept = True

            if accept:
                # Update state
                colors[ei] = c_new
                adj[c_old][i] &= ~(1 << j)
                adj[c_old][j] &= ~(1 << i)
                adj[c_new][i] |= (1 << j)
                adj[c_new][j] |= (1 << i)
                tri += delta
                if tri < best_tri:
                    best_tri = tri
                    best_colors = colors[:]
                    best_adj = [row[:] for row in adj]
                    if best_tri == 0:
                        break

            T *= cooling_rate
            if T < T_min:
                T = T_min

        return best_colors, best_tri

    # ---------- Step 1: Build seed K_32 ----------
    seed_colors = build_seed_32()
    best_n = 32
    best_result = {'n': 32, 'colors': seed_colors}

    # ---------- Step 2: Try SA for target n = 50, 49, ... ----------
    # For each target, try both greedy and random initialization.
    # If SA finds 0 triangles, return immediately.

    targets = [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40]
    for target_n in targets:
        if target_n <= best_n:
            continue
        if time.time() > DEADLINE - 10:
            break

        # Try multiple restarts with different initializations
        for restart in range(4):
            if time.time() > DEADLINE - 5:
                break

            # Alternate between greedy and random initialization
            if restart % 2 == 0:
                adj, init, tri = greedy_init(target_n)
            else:
                adj, init, tri = random_init(target_n)

            colors, tri = sa_color(adj, init, target_n, time_limit=DEADLINE - 5)
            if colors is not None and tri == 0:
                return {'n': target_n, 'colors': colors}

    # ---------- Step 3: Fallback - try to extend the seed ----------
    # If SA failed for all targets, try to incrementally extend the seed.
    # This is slower but might find a valid coloring for n > 32.

    def edge_idx(i, j, n):
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    def check_new_vertex_valid(n, colors, new_colors):
        for i in range(n):
            ci = new_colors[i]
            for j in range(i + 1, n):
                cj = new_colors[j]
                if ci == cj:
                    idx = edge_idx(i, j, n)
                    if colors[idx] == ci:
                        return False
        return True

    def try_greedy_add_vertex(n, colors, deadline):
        assignment = [None] * n

        def backtrack(v):
            if time.time() > deadline:
                return None
            if v == n:
                return assignment[:]
            for c in [0, 1, 2, 3]:
                assignment[v] = c
                ok = True
                for i in range(v):
                    if assignment[i] == c:
                        idx = edge_idx(i, v, n)
                        if colors[idx] == c:
                            ok = False
                            break
                if not ok:
                    assignment[v] = None
                    continue
                result = backtrack(v + 1)
                if result is not None:
                    return result
                assignment[v] = None
            return None

        return backtrack(0)

    # Try to extend seed
    n = 32
    colors = seed_colors[:]
    for target in range(32, 51):
        if time.time() > DEADLINE - 5:
            break
        new_colors = try_greedy_add_vertex(n, colors, DEADLINE - 5)
        if new_colors is not None:
            for i in range(n):
                colors.append(new_colors[i])
            n += 1
            if n > best_n:
                best_n = n
                best_result = {'n': n, 'colors': colors[:]}
        else:
            break

    return best_result

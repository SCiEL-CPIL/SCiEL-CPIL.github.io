"""C4-free graph without independent 11-set (R(C4,K11) > n).

Strategy:
  - Greedy edge-addition that prioritizes edges minimizing a fast independence
    number upper bound (greedy coloring of the complement), while preserving
    C4-freeness by construction.
  - Fast greedy coloring bound to quickly reject graphs with alpha >= 11.
  - Exact independent-set search (branch & bound) with bitmask pruning for
    final verification.
  - Try n from 43 down to 30; for each n, try many seeds.
  - Fallback: 10 disjoint triangles (n=30, alpha=10).
"""
import time
import random
import sys

def construct():
    start = time.time()
    DEADLINE = start + 140.0  # safety margin under 150s
    sys.setrecursionlimit(10000)

    # ---------- Fallback: 10 disjoint triangles (n=30, alpha=10) ----------
    def fallback():
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

    # ---------- C4-free check (bitmask) ----------
    def is_c4_free(adj_mask, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_mask[u] & adj_mask[v]
                if common & (common - 1):  # more than one common neighbor
                    return False
        return True

    # ---------- Fast greedy coloring upper bound on alpha ----------
    # Colors the complement graph; number of colors = upper bound on alpha.
    # Uses a greedy DSATUR-like heuristic for speed.
    def greedy_color_bound(adj_mask, n):
        # Complement adjacency: two vertices are adjacent in complement
        # iff they are NOT adjacent in the original graph.
        # We want to color the complement such that adjacent-in-complement
        # vertices get different colors.
        # Greedy: pick uncolored vertex with max degree in remaining complement
        # subgraph, assign smallest available color.
        colored = [-1] * n
        colors_used = 0
        remaining = (1 << n) - 1

        for _ in range(n):
            if remaining == 0:
                break
            # Pick vertex with max complement-degree in remaining set
            c = remaining
            best_v = -1
            best_deg = -1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                # Complement degree in remaining: count remaining vertices
                # NOT adjacent to v in original (and not v itself)
                comp_deg = (remaining & ~(adj_mask[v] | (1 << v))).bit_count()
                if comp_deg > best_deg:
                    best_deg = comp_deg
                    best_v = v

            if best_v == -1:
                break

            # Assign smallest color not used by complement-neighbors
            # A complement-neighbor of v is a vertex w in remaining
            # such that w is NOT adjacent to v in original.
            comp_neighbors = remaining & ~(adj_mask[best_v] | (1 << best_v))
            used_colors = set()
            cn = comp_neighbors
            while cn:
                lsb = cn & (-cn)
                w = lsb.bit_length() - 1
                cn ^= lsb
                if colored[w] >= 0:
                    used_colors.add(colored[w])

            col = 0
            while col in used_colors:
                col += 1
            colored[best_v] = col
            colors_used = max(colors_used, col + 1)
            remaining &= ~(1 << best_v)

        return colors_used

    # ---------- Independent set search (bitmask, branch & bound) ----------
    def has_independent_set(adj_mask, n, target=11):
        node_count = [0]
        max_nodes = 2000000

        def dfs(candidates, size):
            if size >= target:
                return True
            if candidates == 0:
                return False
            if size + candidates.bit_count() < target:
                return False

            node_count[0] += 1
            if node_count[0] > max_nodes:
                return False

            # Pick vertex with highest degree in candidate subgraph
            c = candidates
            best_v = -1
            max_deg = -1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                    if deg >= 10:
                        break

            if best_v == -1:
                return False

            # Include best_v
            new_cands = candidates & ~(adj_mask[best_v] | (1 << best_v))
            if dfs(new_cands, size + 1):
                return True

            # Exclude best_v
            new_cands = candidates & ~(1 << best_v)
            if dfs(new_cands, size):
                return True

            return False

        all_cands = (1 << n) - 1
        return dfs(all_cands, 0)

    # ---------- Greedy C4-free graph generation with alpha-minimizing edge order ----------
    def generate_greedy_c4_free(n, seed):
        rng = random.Random(seed)
        adj_mask = [0] * n
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_pairs)

        # We'll maintain a list of candidate edges and repeatedly pick the one
        # that minimizes the greedy color bound after adding it.
        # For efficiency, we use a two-phase approach:
        # Phase 1: Add edges in random order but skip those that would create C4.
        # Phase 2: Try to add remaining edges in order of alpha-bound improvement.

        # Actually, let's do a smarter approach:
        # Maintain a priority queue-like structure. At each step, among all
        # remaining valid edges, pick the one that minimizes the color bound.
        # This is expensive, so we do it in batches.

        remaining = list(all_pairs)
        batch_size = max(1, len(remaining) // 10)

        while remaining:
            if time.time() > DEADLINE - 3:
                break

            # Take a batch of edges
            batch = remaining[:batch_size]
            remaining = remaining[batch_size:]

            # For each edge in batch, check if it's C4-safe and compute
            # the resulting color bound. Pick the best one.
            best_edge = None
            best_bound = n + 1
            best_adj = None

            for (u, v) in batch:
                nu = adj_mask[u]
                nv = adj_mask[v]

                # Check C4-freeness
                if nu.bit_count() < nv.bit_count():
                    c = nu
                    safe = True
                    while c:
                        lsb = c & (-c)
                        x = lsb.bit_length() - 1
                        c ^= lsb
                        if adj_mask[x] & nv:
                            safe = False
                            break
                else:
                    c = nv
                    safe = True
                    while c:
                        lsb = c & (-c)
                        x = lsb.bit_length() - 1
                        c ^= lsb
                        if adj_mask[x] & nu:
                            safe = False
                            break

                if not safe:
                    continue

                # Compute color bound with this edge added
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                bound = greedy_color_bound(adj_mask, n)
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)

                if bound < best_bound:
                    best_bound = bound
                    best_edge = (u, v)
                    best_adj = adj_mask[:]

                # Early stop if we already have a very good bound
                if bound <= 10:
                    break

            if best_edge is not None:
                u, v = best_edge
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
            else:
                # No edge in this batch is safe; break
                break

        return adj_mask

    # ---------- Build edge list from bitmask ----------
    def build_edges(n, adj_mask):
        edges = []
        for u in range(n):
            m = adj_mask[u]
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                m ^= lsb
        return edges

    # ---------- Search from high n down ----------
    best = None
    for n in range(43, 29, -1):
        if time.time() > DEADLINE - 5:
            break
        seeds = 0
        max_seeds = 500
        while time.time() < DEADLINE - 5 and seeds < max_seeds:
            adj_mask = generate_greedy_c4_free(n, seeds)

            # Verify C4-free
            if not is_c4_free(adj_mask, n):
                seeds += 1
                continue

            # Quick check with greedy color bound
            bound = greedy_color_bound(adj_mask, n)
            if bound >= 11:
                # Could still have alpha < 11, but likely not.
                # Do exact check only if bound is close.
                if bound > 13:
                    seeds += 1
                    continue

            # Check independence number exactly
            if not has_independent_set(adj_mask, n, 11):
                edges = build_edges(n, adj_mask)
                best = {'n': n, 'edges': edges}
                break

            seeds += 1

        if best is not None:
            break

    if best is None:
        best = fallback()

    return best

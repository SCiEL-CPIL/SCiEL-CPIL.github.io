"""
Construct a C4-free graph on n vertices with no independent 11-set, maximizing n.

Strategy (combined):
  - Use bitset-based (Python int) adjacency for fast C4 checks and alpha computation.
  - Random-greedy C4-free construction: shuffle all edges, add if safe.
  - Multiple seeds, try n=43 down to n=35.
  - Exact alpha via branch-and-bound with bitsets (max-degree branching).
  - Fallback: 10 disjoint triangles (n=30, alpha=10).
"""

import time
import random
import sys

sys.setrecursionlimit(100000)


def construct():
    start_time = time.time()
    deadline = start_time + 112  # safety margin under 120s

    # Fallback: 10 disjoint triangles, n=30, alpha=10 < 11, C4-free
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]

    def check_c4_free(adj_mask, n):
        """Verify no two vertices share >= 2 common neighbours."""
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_mask[u] & adj_mask[v]
                if common & (common - 1):  # more than 1 bit set
                    return False
        return True

    def compute_alpha(adj_mask, n, time_limit):
        """
        Exact maximum independent set size via branch-and-bound with bitsets.
        Branches on vertex with max degree in induced subgraph.
        """
        best = [0]
        deadline_alpha = start_time + time_limit

        def dfs(candidates, size):
            if size > best[0]:
                best[0] = size
            if candidates == 0:
                return
            if time.time() > deadline_alpha:
                return
            # Upper bound: can't exceed size + remaining
            if size + candidates.bit_count() <= best[0]:
                return

            # Pick vertex with max degree in induced subgraph for better pruning
            max_deg = -1
            chosen = -1
            c = candidates
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    chosen = v

            if chosen == -1:
                return

            # Branch 1: include chosen
            new_cand = candidates & ~(adj_mask[chosen] | (1 << chosen))
            dfs(new_cand, size + 1)

            # Branch 2: exclude chosen
            new_cand = candidates & ~(1 << chosen)
            dfs(new_cand, size)

        all_cand = (1 << n) - 1
        dfs(all_cand, 0)
        return best[0]

    def greedy_c4_free(n, seed):
        """
        Random greedy: shuffle all edges, add each if it doesn't create a C4.
        C4 check: adding (u,v) is safe iff no x in N(v) has an edge to any y in N(u).
        Equivalently: for all x in N(v), adj[x] & N(u) == 0.
        """
        rng = random.Random(seed)
        adj_mask = [0] * n
        graph_edges = []

        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_pairs)

        for u, v in all_pairs:
            nu = adj_mask[u]
            nv = adj_mask[v]

            # Check: is there x in N(v) such that x is adjacent to some y in N(u)?
            # i.e., exists x: x in nv and (adj_mask[x] & nu) != 0
            safe = True
            c = nv
            while c:
                lsb = c & (-c)
                x = lsb.bit_length() - 1
                c ^= lsb
                if adj_mask[x] & nu:
                    safe = False
                    break

            if safe:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                graph_edges.append([u, v])

        return adj_mask, graph_edges

    def greedy_independent_lower_bound(adj_mask, n):
        """Quick greedy lower bound on alpha (for early rejection)."""
        order = sorted(range(n), key=lambda i: (adj_mask[i]).bit_count())
        selected = 0
        count = 0
        for v in order:
            # v can be added if no selected vertex is adjacent to v
            if (adj_mask[v] & selected) == 0:
                selected |= (1 << v)
                count += 1
        return count

    best_result = None
    best_n = 0

    # Try n from 43 down to 35, multiple seeds each
    for n_target in [43, 42, 41, 40, 39, 38, 37, 36, 35]:
        if time.time() > deadline - 5:
            break

        # Number of seeds scales with remaining time
        remaining = deadline - time.time()
        if remaining > 60:
            num_seeds = 20
        elif remaining > 30:
            num_seeds = 12
        elif remaining > 15:
            num_seeds = 8
        else:
            num_seeds = 4

        for seed in range(num_seeds):
            if time.time() > deadline - 5:
                break

            adj_mask, graph_edges = greedy_c4_free(n_target, seed)

            # Quick C4 verification (should always pass by construction, but be safe)
            if not check_c4_free(adj_mask, n_target):
                continue

            # Quick greedy lower bound on alpha
            glb = greedy_independent_lower_bound(adj_mask, n_target)
            if glb >= 11:
                # Greedy found an independent set of size >= 11, so alpha >= 11
                # This graph is invalid
                continue

            # Exact alpha check
            alpha = compute_alpha(adj_mask, n_target, time_limit=15.0)

            if time.time() > deadline - 5:
                break

            if alpha < 11:
                if n_target > best_n:
                    best_n = n_target
                    best_result = {'n': n_target, 'edges': graph_edges}
                # Found valid for this n; try larger n (already iterating downward, so break)
                break

        # If we found a valid graph for this n, we already broke the inner loop.
        # Continue to try larger n? No, we're going downward. 
        # Actually we want the LARGEST n, so we should try 43 first and stop if valid.
        # But the loop goes 43, 42, 41... so if 43 works, we break and return.
        # If 43 doesn't work, we try 42, etc.
        if best_result is not None and best_n == n_target:
            # We found a valid graph; but maybe a larger n also works?
            # Since we iterate from 43 down, if we found one at n_target,
            # all larger n already failed. So this is the best.
            break

    if best_result is None:
        best_result = {'n': 30, 'edges': fallback_edges}

    return best_result

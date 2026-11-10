import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph with no regular induced subgraph on 6 or more vertices.
    Strategy:
    - Fast constraint-aware local search using degree arrays and subset indices.
    - Objective: weighted count of regular induced subgraphs of size 6, 7, 8.
    - Once near-zero, perform exact verification for all sizes >= 6.
    - Deterministic seed for reproducibility; multiple restarts if time permits.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Use a fixed seed for reproducibility and consistent behavior
    random.seed(12345)
    np.random.seed(12345)

    def get_subsets(n, size):
        """Yields (mask, vertices) for all subsets of given size."""
        for comb in combinations(range(n), size):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            yield mask, comb

    def build_subset_index(n, size):
        """
        Returns:
        - subsets: list of (mask, verts)
        - edge_idx: edge_idx[u][v] -> list of subset indices containing edge (u,v)
        """
        subsets = list(get_subsets(n, size))
        edge_idx = [[[] for _ in range(n)] for _ in range(n)]
        for i, (mask, verts) in enumerate(subsets):
            for a in range(len(verts)):
                for b in range(a + 1, len(verts)):
                    u = verts[a]
                    v = verts[b]
                    edge_idx[u][v].append(i)
                    edge_idx[v][u].append(i)
        return subsets, edge_idx

    def is_regular(adj, mask, verts):
        """Check if induced subgraph on verts is regular."""
        if not verts:
            return False
        first_deg = (adj[verts[0]] & mask).bit_count()
        for u in verts[1:]:
            if (adj[u] & mask).bit_count() != first_deg:
                return False
        return True

    def weighted_cost(adj, subsets_by_size, reg_flags, weights):
        """
        Compute weighted cost:
        sum over sizes of weight_k * (# regular subsets of size k)
        reg_flags is a dict: size -> list of bools
        """
        cost = 0.0
        for k, subs in subsets_by_size.items():
            flags = reg_flags[k]
            w = weights[k]
            # Count True efficiently
            cnt = sum(1 for f in flags if f)
            cost += w * cnt
        return cost

    def update_reg_flags(adj, subsets_by_size, reg_flags, edge_idx_by_size, u, v):
        """
        Update reg_flags for all sizes after flipping edge (u,v).
        Returns delta_cost contribution for each size? We'll recompute weighted cost separately.
        """
        for k, subs in subsets_by_size.items():
            idxs = edge_idx_by_size[k][u][v]
            flags = reg_flags[k]
            for idx in idxs:
                mask, verts = subs[idx]
                flags[idx] = is_regular(adj, mask, verts)

    def try_find_graph(n, time_budget):
        """
        Attempts to find a valid graph on n vertices using weighted local search.
        Returns edges list or None.
        """
        if time.time() > time_budget:
            return None

        # Precompute subsets for sizes 6, 7, 8 (and maybe 9 for n<=18)
        sizes_to_track = [6, 7, 8]
        if n <= 18:
            sizes_to_track.append(9)
        if n <= 16:
            sizes_to_track.append(10)

        subsets_by_size = {}
        edge_idx_by_size = {}
        reg_flags = {}

        for k in sizes_to_track:
            subs, eidx = build_subset_index(n, k)
            subsets_by_size[k] = subs
            edge_idx_by_size[k] = eidx
            reg_flags[k] = [False] * len(subs)

        # Weights: emphasize smaller sizes (more numerous and critical), but also penalize larger
        weights = {}
        for k in sizes_to_track:
            # Smaller k: higher weight because many subsets; larger k: lower but still important
            weights[k] = 1.0 / (k - 5)

        # Local search state
        adj = [0] * n

        # Initial random graph with density 0.5
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

        # Initialize reg_flags
        for k, subs in subsets_by_size.items():
            flags = reg_flags[k]
            for i, (mask, verts) in enumerate(subs):
                flags[i] = is_regular(adj, mask, verts)

        # Initial cost
        cost = weighted_cost(adj, subsets_by_size, reg_flags, weights)
        best_cost = cost
        best_adj = adj[:]

        # Simulated annealing parameters
        T = 1.0
        cooling = 0.9995
        min_T = 0.05
        steps = 0
        max_steps = 200000  # Increased steps due to faster updates

        # Track no-improvement steps for restart
        no_improve = 0
        max_no_improve = 5000

        while time.time() < time_budget and T > min_T and steps < max_steps and no_improve < max_no_improve:
            steps += 1

            # Pick random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue

            # Flip edge
            if adj[u] & (1 << v):
                # Remove
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                delta_flip = -1
            else:
                # Add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                delta_flip = 1

            # Update reg_flags for affected subsets
            old_cost = cost
            update_reg_flags(adj, subsets_by_size, reg_flags, edge_idx_by_size, u, v)
            new_cost = weighted_cost(adj, subsets_by_size, reg_flags, weights)
            delta = new_cost - old_cost

            # Acceptance
            if delta < 0:
                accept = True
            else:
                if delta == 0:
                    accept = True
                else:
                    accept = random.random() < np.exp(-delta / T)

            if not accept:
                # Revert
                if delta_flip == -1:
                    # Was removed, add back
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    # Was added, remove
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                # Revert reg_flags
                update_reg_flags(adj, subsets_by_size, reg_flags, edge_idx_by_size, u, v)
                # Cost should be back to old_cost
                cost = old_cost
            else:
                cost = new_cost

            # Update best
            if cost < best_cost:
                best_cost = cost
                best_adj = adj[:]
                no_improve = 0
                # If we reach near-zero, we can stop early
                if best_cost < 1e-9:
                    break
            else:
                no_improve += 1

            T *= cooling

        # Use best_adj found
        adj = best_adj

        # Final Verification: Check all subsets of size >= 6
        # Only do this if cost is very low (likely zero)
        if best_cost < 1e-9:
            valid = True
            for k in range(6, n + 1):
                for mask, verts in get_subsets(n, k):
                    if is_regular(adj, mask, verts):
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            edges.append([u, v])
                return edges

        return None

    # Try n=20, then 19, etc.
    for n in range(20, 4, -1):
        edges = try_find_graph(n, deadline)
        if edges is not None:
            return {'n': n, 'edges': edges}

        # If we spent too much time on n, break
        if time.time() > deadline - 10:
            break

    # Fallback: Return a trivial graph
    # n=5 is always valid (no subset of size 6)
    return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END

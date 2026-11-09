import time
import random
import numpy as np
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill

    # Best known seed: Paley graph on 17 vertices
    squares = {x * x % 17 for x in range(1, 17)}
    edges_17 = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_n = 17
    best_edges = edges_17

    # Build adjacency matrix for Paley(17)
    adj17 = np.zeros((17, 17), dtype=np.int8)
    for u, v in edges_17:
        adj17[u, v] = 1
        adj17[v, u] = 1

    # Precompute neighbor lists for faster operations
    def get_neighbors(adj, n):
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        return neighbors

    def build_masks(n, adj):
        masks = [0] * n
        for u in range(n):
            for v in range(n):
                if adj[u, v]:
                    masks[u] |= (1 << v)
        return masks

    def popcount(x):
        return bin(x).count('1')

    def has_k4(n, masks):
        for u in range(n):
            for v in range(u + 1, n):
                if not (masks[u] & (1 << v)):
                    continue
                common = masks[u] & masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if masks[wi] & (1 << xi):
                            return True
        return False

    def has_ind6(n, masks):
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))

        for u in range(n):
            for v in range(u + 1, n):
                if not (comp_masks[u] & (1 << v)):
                    continue
                common = comp_masks[u] & comp_masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if comp_masks[wi] & (1 << xi):
                            common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi] & comp_masks[xi]
                            common2 &= ~((1 << (xi + 1)) - 1)
                            y_mask = common2
                            while y_mask:
                                lsb_y = y_mask & -y_mask
                                yi = lsb_y.bit_length() - 1
                                y_mask ^= lsb_y
                                z_mask = common2 & ~((1 << (yi + 1)) - 1)
                                while z_mask:
                                    lsb_z = z_mask & -z_mask
                                    zi = lsb_z.bit_length() - 1
                                    z_mask ^= lsb_z
                                    if comp_masks[yi] & (1 << zi):
                                        return True
        return False

    def is_valid(n, adj):
        masks = build_masks(n, adj)
        if has_k4(n, masks):
            return False
        if has_ind6(n, masks):
            return False
        return True

    def count_violations(n, masks):
        """Count total K4 + independent 6-sets."""
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))

        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (masks[u] & (1 << v)):
                    continue
                common = masks[u] & masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if masks[wi] & (1 << xi):
                            k4_count += 1

        ind6_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (comp_masks[u] & (1 << v)):
                    continue
                common = comp_masks[u] & comp_masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if comp_masks[wi] & (1 << xi):
                            common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi] & comp_masks[xi]
                            common2 &= ~((1 << (xi + 1)) - 1)
                            y_mask = common2
                            while y_mask:
                                lsb_y = y_mask & -y_mask
                                yi = lsb_y.bit_length() - 1
                                y_mask ^= lsb_y
                                z_mask = common2 & ~((1 << (yi + 1)) - 1)
                                while z_mask:
                                    lsb_z = z_mask & -z_mask
                                    zi = lsb_z.bit_length() - 1
                                    z_mask ^= lsb_z
                                    if comp_masks[yi] & (1 << zi):
                                        ind6_count += 1

        return k4_count + ind6_count

    def delta_k4(n, masks, u, v, adding):
        """Delta in K4 count when toggling edge (u,v).
        adding=True means we're adding the edge (it was absent).
        """
        delta = 0
        # New K4s formed: {u,v,w,x} where w,x are common neighbors of u,v
        # and w-x is an edge. Only count each K4 once by ordering.
        common = masks[u] & masks[v]
        # For each pair (w,x) in common with w<x, check if w-x is edge
        # But we need to be careful: the K4 {u,v,w,x} is only "new" if all 6 edges exist
        # Since u-v is the edge being toggled, we need u-w, u-x, v-w, v-x, w-x all present
        # u-w and v-w are guaranteed (w in common), same for x. So just need w-x.
        w_mask = common
        while w_mask:
            lsb_w = w_mask & -w_mask
            wi = lsb_w.bit_length() - 1
            w_mask ^= lsb_w
            x_mask = common & ~((1 << (wi + 1)) - 1)
            while x_mask:
                lsb_x = x_mask & -x_mask
                xi = lsb_x.bit_length() - 1
                x_mask ^= lsb_x
                if masks[wi] & (1 << xi):
                    delta += 1
        return delta if adding else -delta

    def delta_ind6(n, masks, u, v, adding):
        """Delta in independent 6-set count when toggling edge (u,v).
        adding=True means we're adding the edge (it was absent).
        Adding edge (u,v) DESTROYS independent sets containing both u and v.
        Removing edge (u,v) CREATES independent sets containing both u and v.
        """
        full_mask = (1 << n) - 1
        # Independent sets containing both u and v: need 4 more vertices from
        # common non-neighbors of u and v (excluding u,v themselves)
        # Non-neighbors of u (excluding u): full_mask & ~(masks[u] | (1 << u))
        # Common non-neighbors of u and v: intersection of their non-neighbor sets
        non_u = full_mask & ~(masks[u] | (1 << u))
        non_v = full_mask & ~(masks[v] | (1 << v))
        common_non = non_u & non_v

        # Count K4 in the subgraph induced by common_non in the complement
        # i.e., count independent 4-sets in the original graph within common_non
        # that are also non-adjacent to each other
        # This is: number of 4-cliques in complement restricted to common_non
        # = number of independent 4-sets in original within common_non

        # For efficiency, count 4-subsets of common_non that are mutually non-adjacent
        # Use bitmask iteration
        count = 0
        w_mask = common_non
        while w_mask:
            lsb_w = w_mask & -w_mask
            wi = lsb_w.bit_length() - 1
            w_mask ^= lsb_w
            x_mask = common_non & ~((1 << (wi + 1)) - 1)
            while x_mask:
                lsb_x = x_mask & -x_mask
                xi = lsb_x.bit_length() - 1
                x_mask ^= lsb_x
                # w and x must be non-adjacent
                if masks[wi] & (1 << xi):
                    continue
                y_mask = common_non & ~((1 << (xi + 1)) - 1)
                while y_mask:
                    lsb_y = y_mask & -y_mask
                    yi = lsb_y.bit_length() - 1
                    y_mask ^= lsb_y
                    # y must be non-adjacent to w and x
                    if (masks[yi] & (1 << wi)) or (masks[yi] & (1 << xi)):
                        continue
                    z_mask = common_non & ~((1 << (yi + 1)) - 1)
                    while z_mask:
                        lsb_z = z_mask & -z_mask
                        zi = lsb_z.bit_length() - 1
                        z_mask ^= lsb_z
                        # z must be non-adjacent to w, x, y
                        if (masks[zi] & (1 << wi)) or (masks[zi] & (1 << xi)) or (masks[zi] & (1 << yi)):
                            continue
                        count += 1

        # Adding edge destroys these independent 6-sets
        # Removing edge creates them
        return -count if adding else count

    def try_find_graph(n, time_budget, seed_adj=None, density=0.5):
        """Local search with incremental cost updates."""
        end_time = time.time() + time_budget

        if seed_adj is not None and seed_adj.shape == (n, n):
            adj = seed_adj.copy()
        else:
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < density:
                        adj[i, j] = 1
                        adj[j, i] = 1

        masks = build_masks(n, adj)
        current_cost = count_violations(n, masks)
        if current_cost == 0:
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges

        T = 5.0
        min_T = 0.05
        cooling = 0.9995
        no_improve = 0
        max_no_improve = 10000
        steps_since_recheck = 0
        recheck_interval = 500  # periodically verify cost to prevent drift
        best_cost = current_cost
        best_adj = adj.copy()
        best_masks = masks[:]

        # Track recent moves for diversification
        recent_moves = deque(maxlen=100)

        while time.time() < end_time and no_improve < max_no_improve:
            # Choose move strategy: mostly random, sometimes biased
            if random.random() < 0.7:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
            else:
                # Biased: pick vertices with high degree or low degree
                if random.random() < 0.5:
                    # Pick high-degree vertex
                    degrees = [popcount(masks[i]) for i in range(n)]
                    u = degrees.index(max(degrees))
                    v = random.randint(0, n - 1)
                else:
                    # Pick low-degree vertex
                    degrees = [n - 1 - popcount(masks[i]) for i in range(n)]
                    u = degrees.index(max(degrees))
                    v = random.randint(0, n - 1)
            
            if u == v:
                continue

            # Avoid repeating recent moves
            if (u, v) in recent_moves or (v, u) in recent_moves:
                continue

            old_state = adj[u, v]
            adding = (old_state == 0)  # True if we're adding the edge

            # Compute delta
            dk4 = delta_k4(n, masks, u, v, adding)
            din6 = delta_ind6(n, masks, u, v, adding)
            delta = dk4 + din6
            new_cost = current_cost + delta

            # Apply the toggle
            adj[u, v] = 1 - old_state
            adj[v, u] = 1 - old_state
            if adding:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
            else:
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)

            recent_moves.append((u, v))

            if new_cost < current_cost:
                current_cost = new_cost
                no_improve = 0
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = adj.copy()
                    best_masks = masks[:]
                if new_cost == 0:
                    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                    return edges
            else:
                if delta <= 0:
                    current_cost = new_cost
                    no_improve += 1
                else:
                    T = max(min_T, T * cooling)
                    # Simulated annealing acceptance
                    if random.random() < np.exp(-delta / T):
                        current_cost = new_cost
                        no_improve += 1
                    else:
                        # Revert
                        adj[u, v] = old_state
                        adj[v, u] = old_state
                        if adding:
                            masks[u] &= ~(1 << v)
                            masks[v] &= ~(1 << u)
                        else:
                            masks[u] |= (1 << v)
                            masks[v] |= (1 << u)
                        no_improve += 1

            # Periodic recheck to prevent floating point / logic drift
            steps_since_recheck += 1
            if steps_since_recheck >= recheck_interval:
                steps_since_recheck = 0
                actual_cost = count_violations(n, masks)
                if actual_cost != current_cost:
                    current_cost = actual_cost
                if current_cost == 0:
                    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                    return edges

        # Final check with best state
        if best_cost == 0:
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if best_adj[i, j]]
            return edges

        # Check current state
        if is_valid(n, adj):
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges

        return None

    # Strategy: try to extend from n=18 upward
    for n in range(18, 36):
        if time.time() > deadline:
            break

        remaining = deadline - time.time()
        if remaining < 3.0:
            break

        # Allocate time: more for larger n (harder but more valuable)
        if n <= 22:
            budget = min(remaining * 0.2, 5.0)
        elif n <= 28:
            budget = min(remaining * 0.3, 8.0)
        else:
            budget = min(remaining * 0.4, 12.0)

        # Try multiple restarts with different strategies
        found = None
        num_restarts = 6
        per_restart = budget / num_restarts

        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            if time.time() > start_time + deadline - 5.0:
                break

            if restart == 0:
                # First restart: extend Paley(17) by adding vertices with moderate density
                seed_adj = np.zeros((n, n), dtype=np.int8)
                seed_adj[:17, :17] = adj17
                for i in range(17, n):
                    for j in range(17):
                        if random.random() < 0.45:
                            seed_adj[i, j] = 1
                            seed_adj[j, i] = 1
                for i in range(17, n):
                    for j in range(i + 1, n):
                        if random.random() < 0.45:
                            seed_adj[i, j] = 1
                            seed_adj[j, i] = 1

                found = try_find_graph(n, per_restart, seed_adj=seed_adj, density=0.45)
            elif restart == 1:
                # Second restart: slightly denser random initialization
                seed_adj = np.zeros((n, n), dtype=np.int8)
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.48:
                            seed_adj[i, j] = 1
                            seed_adj[j, i] = 1
                found = try_find_graph(n, per_restart, seed_adj=seed_adj, density=0.48)
            elif restart == 2:
                # Third restart: slightly sparser
                seed_adj = np.zeros((n, n), dtype=np.int8)
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.42:
                            seed_adj[i, j] = 1
                            seed_adj[j, i] = 1
                found = try_find_graph(n, per_restart, seed_adj=seed_adj, density=0.42)
            elif restart == 3:
                # Fourth restart: Paley-based with different density
                seed_adj = np.zeros((n, n), dtype=np.int8)
                seed_adj[:17, :17] = adj17
                for i in range(17, n):
                    for j in range(17):
                        if random.random() < 0.52:
                            seed_adj[i, j] = 1
                            seed_adj[j, i] = 1
                for i in range(17, n):
                    for j in range(i + 1, n):
                        if random.random() < 0.52:
                            seed_adj[i, j] = 1
                            seed_adj[j, i] = 1
                found = try_find_graph(n, per_restart, seed_adj=seed_adj, density=0.52)
            else:
                # Subsequent restarts: purely random with varied densities
                dens = random.uniform(0.40, 0.55)
                found = try_find_graph(n, per_restart, density=dens)

            if found is not None:
                break

        if found is not None:
            best_n = n
            best_edges = found
            if n >= 35:
                break

    return {'n': best_n, 'edges': best_edges}

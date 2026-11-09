import time
import random
import numpy as np
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill

    best_n = 17
    best_edges = None

    # Seed: Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    edges_17 = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_edges = edges_17

    def build_circulant_adj(n, S):
        """Build adjacency matrix for circulant graph with connection set S."""
        adj = np.zeros((n, n), dtype=bool)
        for s in S:
            s_mod = s % n
            for i in range(n):
                j = (i + s_mod) % n
                adj[i, j] = True
        return adj

    def build_masks(n, adj):
        """Build bitmask neighborhoods from numpy adj matrix."""
        masks = [0] * n
        for u in range(n):
            m = 0
            for v in range(n):
                if adj[u, v]:
                    m |= (1 << v)
            masks[u] = m
        return masks

    def has_k4(n, masks):
        """Check if any K4 exists. Early exit."""
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
        """Check if any independent 6-set exists. Early exit."""
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

    def is_valid_circulant(n, S):
        """Check if circulant graph with connection set S is valid."""
        adj = build_circulant_adj(n, S)
        masks = build_masks(n, adj)
        if has_k4(n, masks):
            return False, None
        if has_ind6(n, masks):
            return False, None
        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
        return True, edges

    def count_violations(n, masks):
        """Count K4s + independent 6-sets for cost function."""
        # Count K4s
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

        # Count independent 6-sets
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))

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

    def try_circulant_search(n, time_budget):
        """Search for valid circulant graph on n vertices."""
        end_time = time.time() + time_budget
        best_cost = float('inf')
        best_S = None

        # Generate candidate connection sets using algebraic structure
        # For circulant graphs, S should be symmetric: if s in S, then n-s in S
        # We search over subsets of {1, 2, ..., n//2}

        max_s = n // 2
        candidates = list(range(1, max_s + 1))

        # Start with algebraic seeds: quadratic residues, etc.
        seeds = []

        # Paley-type: quadratic residues mod n (if n is prime)
        if n > 2:
            try:
                # Check if n is prime
                is_prime = all(n % i != 0 for i in range(2, int(n**0.5) + 1)) if n > 1 else False
                if is_prime:
                    qres = {x * x % n for x in range(1, n)}
                    qres.discard(0)
                    # Symmetrize
                    sym_qres = set()
                    for s in qres:
                        sym_qres.add(min(s, n - s))
                    if sym_qres:
                        seeds.append(sorted(sym_qres))
            except:
                pass

        # Random seeds with various densities
        for density in [0.3, 0.4, 0.5, 0.6, 0.7]:
            S_rand = [s for s in candidates if random.random() < density]
            if S_rand:
                seeds.append(S_rand)

        # Also try small structured sets
        for size in range(2, min(8, max_s + 1)):
            for combo in itertools.combinations(candidates, size):
                seeds.append(list(combo))
                if time.time() > end_time:
                    break
            if time.time() > end_time:
                break

        # Evaluate seeds
        for S in seeds:
            if time.time() > end_time:
                break
            adj = build_circulant_adj(n, S)
            masks = build_masks(n, adj)
            if has_k4(n, masks):
                continue
            if has_ind6(n, masks):
                continue
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges

        # Local search on circulant connection sets
        random.seed(int(time.time() * 1000) % 100000)
        num_restarts = min(50, max(10, int(time_budget * 5)))
        restart_budget = time_budget / num_restarts

        for restart in range(num_restarts):
            if time.time() >= end_time:
                break

            # Random initial connection set
            S = set()
            for s in candidates:
                if random.random() < 0.4:
                    S.add(s)

            if not S:
                S.add(random.choice(candidates))

            adj = build_circulant_adj(n, S)
            masks = build_masks(n, adj)
            current_cost = count_violations(n, masks)

            if current_cost == 0:
                edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                return edges

            T = 1.0
            no_improve = 0
            max_no_improve = 200

            while time.time() < end_time and no_improve < max_no_improve:
                # Choose a random element to toggle
                all_elements = list(range(1, n))
                elem = random.choice(all_elements)
                # Ensure symmetry: toggle both elem and n-elem
                sym_elem = n - elem if elem != n - elem else elem

                if elem in S:
                    S.discard(elem)
                    if sym_elem != elem:
                        S.discard(sym_elem)
                else:
                    S.add(elem)
                    if sym_elem != elem:
                        S.add(sym_elem)

                adj_new = build_circulant_adj(n, S)
                masks_new = build_masks(n, adj_new)
                new_cost = count_violations(n, masks_new)

                if new_cost < current_cost:
                    current_cost = new_cost
                    no_improve = 0
                    adj = adj_new
                    masks = masks_new
                    if new_cost == 0:
                        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                        return edges
                else:
                    delta = new_cost - current_cost
                    T = max(0.1, T * 0.998)
                    if random.random() < np.exp(-delta / T):
                        current_cost = new_cost
                        adj = adj_new
                        masks = masks_new
                        no_improve += 1
                    else:
                        # Revert
                        if elem in S:
                            S.discard(elem)
                            if sym_elem != elem:
                                S.discard(sym_elem)
                        else:
                            S.add(elem)
                            if sym_elem != elem:
                                S.add(sym_elem)
                        no_improve += 1

            if current_cost < best_cost:
                best_cost = current_cost
                best_S = S.copy()

        # Check best found
        if best_S is not None and best_cost == 0:
            adj = build_circulant_adj(n, best_S)
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges

        return None

    def try_general_search(n, time_budget):
        """General local search for non-circulant graphs (fallback)."""
        end_time = time.time() + time_budget

        best_cost = float('inf')
        best_adj = None

        num_restarts = 5
        restart_budget = time_budget / num_restarts

        for restart in range(num_restarts):
            if time.time() >= end_time:
                break

            adj = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i, j] = True
                        adj[j, i] = True

            masks = build_masks(n, adj)
            current_cost = count_violations(n, masks)
            if current_cost == 0:
                edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                return edges

            T = 1.5
            no_improve = 0
            max_no_improve = 300

            while time.time() < end_time and no_improve < max_no_improve:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue

                old_state = adj[u, v]
                adj[u, v] = not old_state
                adj[v, u] = not old_state

                masks_new = build_masks(n, adj)
                new_cost = count_violations(n, masks_new)

                if new_cost < current_cost:
                    current_cost = new_cost
                    no_improve = 0
                    masks = masks_new
                    if new_cost == 0:
                        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                        return edges
                else:
                    delta = new_cost - current_cost
                    T = max(0.1, T * 0.999)
                    if random.random() < np.exp(-delta / T):
                        current_cost = new_cost
                        masks = masks_new
                        no_improve += 1
                    else:
                        adj[u, v] = old_state
                        adj[v, u] = old_state
                        no_improve += 1

            if current_cost < best_cost:
                best_cost = current_cost
                best_adj = adj.copy()

        if best_adj is not None and best_cost == 0:
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if best_adj[i, j]]
            return edges

        return None

    # Try to find graphs from n=18 upward
    for n in range(18, 36):
        if time.time() > deadline:
            break

        remaining = deadline - time.time()
        if remaining < 3.0:
            break

        # Give more time to smaller n
        if n <= 20:
            circ_budget = min(remaining * 0.5, 8.0)
            gen_budget = min(remaining * 0.3, 5.0)
        elif n <= 25:
            circ_budget = min(remaining * 0.4, 6.0)
            gen_budget = min(remaining * 0.2, 4.0)
        else:
            circ_budget = min(remaining * 0.3, 5.0)
            gen_budget = min(remaining * 0.15, 3.0)

        # Try circulant search first (exploits symmetry)
        edges = try_circulant_search(n, circ_budget)
        if edges is not None:
            best_n = n
            best_edges = edges
            if n >= 35:
                break
            continue

        # Fallback to general search
        if time.time() < deadline - 2.0:
            edges = try_general_search(n, gen_budget)
            if edges is not None:
                best_n = n
                best_edges = edges
                if n >= 35:
                    break

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END

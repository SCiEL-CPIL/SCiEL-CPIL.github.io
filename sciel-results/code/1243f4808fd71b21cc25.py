import random
import time
import math
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def now():
        return time.time()

    def budget_left(margin=3.0):
        return DEADLINE - (now() - start_time) - margin > 0

    n = 16
    full_mask = (1 << 28) - 1

    q3_edges_local = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7),
                      (4, 5), (4, 6), (5, 7), (6, 7)]

    bit_index_map = {}
    idx = 0
    for i in range(8):
        for j in range(i + 1, 8):
            bit_index_map[(i, j)] = idx
            idx += 1

    # Build Q3 masks over the 28-bit edge space of K8
    q3_masks = set()
    for perm in itertools.permutations(range(8)):
        mask = 0
        for u, v in q3_edges_local:
            pu = perm[u]
            pv = perm[v]
            if pu > pv:
                pu, pv = pv, pu
            mask |= (1 << bit_index_map[(pu, pv)])
        q3_masks.add(mask)

    # Lookup table: lookup[mask] = 1 iff mask contains a Q3 as a subgraph
    lookup = bytearray(1 << 28)
    for qm in q3_masks:
        Z = full_mask ^ qm
        s = 0
        while True:
            lookup[qm | s] = 1
            if s == Z:
                break
            s = (s - Z) & Z

    if not budget_left(5.0):
        return []

    # All 8-subsets of 16 vertices
    subsets = list(itertools.combinations(range(n), 8))
    num_subsets = len(subsets)

    # For each edge (u,v): list of (subset_index, bit) it appears in
    edge_to_subsets = [[[] for _ in range(n)] for _ in range(n)]
    subset_edges = []
    for s_idx, combo in enumerate(subsets):
        edge_dict = {}
        for i in range(8):
            for j in range(i + 1, 8):
                u = combo[i]
                v = combo[j]
                b_idx = bit_index_map[(i, j)]
                edge_dict[(u, v)] = b_idx
                edge_to_subsets[u][v].append((s_idx, b_idx))
                edge_to_subsets[v][u].append((s_idx, b_idx))
        subset_edges.append(edge_dict)

    if not budget_left(5.0):
        return []

    def compute_masks_fast(adj):
        masks = [0] * num_subsets
        for s_idx in range(num_subsets):
            mask = 0
            edge_dict = subset_edges[s_idx]
            for (u, v), b_idx in edge_dict.items():
                if (adj[u] >> v) & 1:
                    mask |= (1 << b_idx)
            masks[s_idx] = mask
        return masks

    def compute_edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def greedy_fix(adj, masks, violation_count, max_iters=2000):
        for _ in range(max_iters):
            if not budget_left(8.0):
                break
            if violation_count == 0:
                break
            viol_subset = -1
            for s_idx in range(num_subsets):
                if lookup[masks[s_idx]]:
                    viol_subset = s_idx
                    break
            if viol_subset == -1:
                break
            combo = subsets[viol_subset]
            edge_dict = subset_edges[viol_subset]
            best_edge = None
            best_delta = 10**9
            for (u, v), b_idx in edge_dict.items():
                if not ((adj[u] >> v) & 1):
                    continue
                bit = 1 << b_idx
                old_mask = masks[viol_subset]
                new_mask = old_mask ^ bit
                delta = -lookup[old_mask] + lookup[new_mask]
                for (s_idx2, b_idx2) in edge_to_subsets[u][v]:
                    if s_idx2 == viol_subset:
                        continue
                    old_m2 = masks[s_idx2]
                    new_m2 = old_m2 ^ (1 << b_idx2)
                    delta += -lookup[old_m2] + lookup[new_m2]
                if delta < best_delta:
                    best_delta = delta
                    best_edge = (u, v, b_idx)
            if best_edge is None:
                existing = [(u, v) for (u, v) in edge_dict if (adj[u] >> v) & 1]
                if not existing:
                    break
                u, v = random.choice(existing)
                b_idx = edge_dict[(u, v)]
            else:
                u, v, b_idx = best_edge
            for (s_idx2, b_idx2) in edge_to_subsets[u][v]:
                old_m = masks[s_idx2]
                new_m = old_m ^ (1 << b_idx2)
                violation_count += -lookup[old_m] + lookup[new_m]
                masks[s_idx2] = new_m
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        return adj, masks, violation_count

    def local_search(adj, masks, violation_count, time_budget):
        current_edge_count = compute_edge_count(adj)
        best_edge_count = current_edge_count if violation_count == 0 else -1
        best_adj = adj[:] if violation_count == 0 else None

        T = 6.0
        cooling = 0.9997
        min_T = 0.005
        no_improve = 0
        max_no_improve = 4000

        deadline = now() + time_budget
        while now() < deadline and budget_left(3.0):
            if T < min_T:
                T = min_T

            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u

            affected = edge_to_subsets[u][v]
            delta_violation = 0
            changes = []
            for (s_idx, b_idx) in affected:
                bit = 1 << b_idx
                old_mask = masks[s_idx]
                new_mask = old_mask ^ bit
                changes.append((s_idx, new_mask))
                delta_violation += lookup[new_mask] - lookup[old_mask]

            edge_exists = (adj[u] >> v) & 1
            delta_edges = -1 if edge_exists else 1

            if delta_violation < 0:
                accept = True
            elif delta_violation == 0:
                if delta_edges > 0:
                    accept = True
                else:
                    accept = random.random() < (0.12 if T > 1.0 else 0.04)
            else:
                base_penalty = 40.0
                edge_bonus = 9.0
                cost = delta_violation * base_penalty - delta_edges * edge_bonus
                cost += 4.0 * math.log(1.0 + delta_violation)
                if cost <= 0:
                    accept = True
                else:
                    accept = random.random() < math.exp(-cost / T)

            if accept:
                for (s_idx, new_mask) in changes:
                    masks[s_idx] = new_mask
                violation_count += delta_violation
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                current_edge_count += delta_edges

                improved = False
                if violation_count == 0 and current_edge_count > best_edge_count:
                    best_edge_count = current_edge_count
                    best_adj = adj[:]
                    improved = True
                elif violation_count < 0:
                    # shouldn't happen, but guard
                    pass

                if improved:
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve > max_no_improve:
                    no_improve = 0
                    T = max(T, 1.5)

            T *= cooling

        if best_adj is not None:
            return best_adj, best_edge_count
        return adj, compute_edge_count(adj)

    best_overall_adj = None
    best_overall_edges = 0

    def try_start(adj, sa_budget):
        nonlocal best_overall_adj, best_overall_edges
        if not budget_left(5.0):
            return
        masks = compute_masks_fast(adj)
        viol = sum(1 for m in masks if lookup[m])
        adj, masks, viol = greedy_fix(adj, masks, viol)
        if viol == 0:
            adj, edges = local_search(adj, masks, viol, sa_budget)
            if edges > best_overall_edges:
                best_overall_edges = edges
                best_overall_adj = adj[:]

    # Strategy 1: K16 then greedy fix
    adj1 = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            adj1[i] |= (1 << j)
            adj1[j] |= (1 << i)
    try_start(adj1, 20.0)

    # Strategy 2: Random dense starts
    for restart in range(4):
        if not budget_left(15.0):
            break
        density = 0.62 + 0.07 * restart
        adj2 = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj2[i] |= (1 << j)
                    adj2[j] |= (1 << i)
        try_start(adj2, 12.0)

    # Strategy 3: K16 minus random edges
    for restart in range(3):
        if not budget_left(12.0):
            break
        adj3 = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                adj3[i] |= (1 << j)
                adj3[j] |= (1 << i)
        num_removals = 8 + random.randint(0, 6)
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges[:num_removals]:
            adj3[u] ^= (1 << v)
            adj3[v] ^= (1 << u)
        try_start(adj3, 10.0)

    # Final greedy densification
    if best_overall_adj is not None:
        final_masks = compute_masks_fast(best_overall_adj)
        for _ in range(6):
            if not budget_left(3.0):
                break
            changed = True
            while changed and budget_left(2.0):
                changed = False
                for i in range(n):
                    for j in range(i + 1, n):
                        if not ((best_overall_adj[i] >> j) & 1):
                            affected = edge_to_subsets[i][j]
                            creates_q3 = False
                            for (s_idx, b_idx) in affected:
                                bit = 1 << b_idx
                                new_mask = final_masks[s_idx] | bit
                                if lookup[new_mask]:
                                    creates_q3 = True
                                    break
                            if not creates_q3:
                                best_overall_adj[i] |= (1 << j)
                                best_overall_adj[j] |= (1 << i)
                                best_overall_edges += 1
                                for (s_idx, b_idx) in affected:
                                    final_masks[s_idx] |= (1 << b_idx)
                                changed = True

    if best_overall_adj is None:
        # Fallback valid construction
        edges = []
        for i in range(8):
            for j in range(i + 1, 8):
                edges.append([i, j])
        for i in range(8, 16):
            for j in range(i + 1, 16):
                edges.append([i, j])
        for i in range(8):
            for j in range(8, 16):
                if (i ^ j) & 1:
                    edges.append([i, j])
        return edges

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_overall_adj[i] >> j) & 1:
                edges.append([i, j])
    return edges

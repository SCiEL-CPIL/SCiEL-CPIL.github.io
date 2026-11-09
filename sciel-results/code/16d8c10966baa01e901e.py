import time
import random

def construct():
    start_time = time.time()
    TOTAL_BUDGET = 145.0

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj_mask = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        return adj_mask

    def iter_bits(x):
        while x:
            lsb = x & -x
            yield lsb.bit_length() - 1
            x ^= lsb

    def get_edges_from_mask(adj_mask, n):
        edges = set()
        for u in range(n):
            m = adj_mask[u] & ~((1 << (u + 1)) - 1)
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                edges.add((u, v))
                m ^= lsb
        return edges

    def clique_edges(C):
        """Get set of edges in clique C."""
        es = set()
        for i in range(len(C)):
            for j in range(i + 1, len(C)):
                u, v = C[i], C[j]
                es.add((min(u, v), max(u, v)))
        return es

    def solve_instance(n, p, seed, time_budget):
        adj_mask_orig = get_graph(n, p, seed)
        if not any(adj_mask_orig):
            return []

        all_edges = get_edges_from_mask(adj_mask_orig, n)
        if not all_edges:
            return []

        def is_clique(C, adj):
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v = C[i], C[j]
                    if not (adj[u] & (1 << v)):
                        return False
            return True

        def expand_clique_topk(seed_u, seed_v, adj, n, rng, top_k=3):
            """Expand edge to maximal clique using top-K strategy."""
            C = [seed_u, seed_v]
            U = adj[seed_u] & adj[seed_v]
            U &= ~(1 << seed_u) & ~(1 << seed_v)

            while U:
                candidates = list(iter_bits(U))
                if not candidates:
                    break

                # Compute degree of each candidate within candidate set
                cand_set = set(candidates)
                degs = {}
                for w in candidates:
                    d = 0
                    mw = adj[w]
                    for x in candidates:
                        if x != w and (mw >> x) & 1:
                            d += 1
                    degs[w] = d

                sorted_cands = sorted(candidates, key=lambda w: degs[w], reverse=True)
                k = min(top_k, len(sorted_cands))

                best_w = None
                best_est = -1
                best_remaining = None

                for i in range(k):
                    w = sorted_cands[i]
                    # Estimate: how many candidates remain connected to w
                    new_cands = [x for x in candidates if x != w and (adj[w] >> x) & 1]
                    est = 1 + len(new_cands)
                    if est > best_est:
                        best_est = est
                        best_w = w
                        best_remaining = new_cands

                if best_w is None:
                    break

                C.append(best_w)
                U &= adj[best_w]
                U &= ~(1 << best_w)

            return C

        def expand_clique_simple(seed_u, seed_v, adj, n, rng):
            """Simple greedy expansion."""
            C = [seed_u, seed_v]
            U = adj[seed_u] & adj[seed_v]
            U &= ~(1 << seed_u) & ~(1 << seed_v)

            while U:
                candidates = list(iter_bits(U))
                if not candidates:
                    break

                best_w = None
                best_deg = -1
                for w in candidates:
                    d = 0
                    mw = adj[w]
                    for x in candidates:
                        if x != w and (mw >> x) & 1:
                            d += 1
                    if d > best_deg:
                        best_deg = d
                        best_w = w

                if best_w is None:
                    break

                C.append(best_w)
                U &= adj[best_w]
                U &= ~(1 << best_w)

            return C

        def generate_candidate_cliques(adj_mask_orig, n, rng, deadline, strategy):
            """Generate a set of candidate cliques using a given strategy."""
            adj = list(adj_mask_orig)
            candidates = []

            while True:
                if time.time() > deadline:
                    break

                # Find edges in current graph
                current_edges = []
                for u in range(n):
                    m = adj[u] & ~((1 << (u + 1)) - 1)
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        current_edges.append((u, v))
                        m ^= lsb

                if not current_edges:
                    break

                if strategy == 0:
                    # Max common neighbors
                    best_edge = None
                    best_common = -1
                    for u, v in current_edges:
                        common = (adj[u] & adj[v]).bit_count()
                        if common > best_common:
                            best_common = common
                            best_edge = (u, v)
                elif strategy == 1:
                    # Min common neighbors
                    best_edge = None
                    best_common = float('inf')
                    for u, v in current_edges:
                        common = (adj[u] & adj[v]).bit_count()
                        if common < best_common:
                            best_common = common
                            best_edge = (u, v)
                else:
                    # Random
                    best_edge = rng.choice(current_edges)

                if best_edge is None:
                    break

                u, v = best_edge
                if strategy == 2:
                    C = expand_clique_topk(u, v, adj, n, rng, top_k=3)
                else:
                    C = expand_clique_simple(u, v, adj, n, rng)

                candidates.append(C)

                # Remove clique edges
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        a, b = C[i], C[j]
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)

            # Handle remaining edges as K2
            for u in range(n):
                m = adj[u] & ~((1 << (u + 1)) - 1)
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    candidates.append([u, v])
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    m ^= lsb

            return candidates

        def lagrangian_select(candidates, adj_mask_orig, n, rng, deadline):
            """Select non-conflicting cliques using greedy Lagrangian relaxation."""
            # Compute edge sets for each candidate
            cand_edge_sets = []
            for C in candidates:
                es = clique_edges(C)
                cand_edge_sets.append(es)

            m = len(candidates)
            if m == 0:
                return []

            # Build conflict matrix: conflict[i][j] = number of shared edges
            conflict = [[0] * m for _ in range(m)]
            for i in range(m):
                for j in range(i + 1, m):
                    shared = len(cand_edge_sets[i] & cand_edge_sets[j])
                    conflict[i][j] = shared
                    conflict[j][i] = shared

            # Weights: number of edges covered
            weights = [len(es) for es in cand_edge_sets]

            # Greedy Lagrangian: iteratively select clique with highest adjusted score
            selected = []
            selected_mask = [False] * m
            remaining_edges = set()
            for es in cand_edge_sets:
                remaining_edges.update(es)

            # Track which edges are covered
            covered_edges = set()

            # Penalty parameters
            lambda_val = 1.0

            while True:
                if time.time() > deadline:
                    break

                # Find uncovered edges
                uncovered = all_edges - covered_edges
                if not uncovered:
                    break

                # Score each unselected candidate
                best_idx = -1
                best_score = -1

                for i in range(m):
                    if selected_mask[i]:
                        continue

                    # Check if this clique covers any uncovered edges
                    new_edges = cand_edge_sets[i] - covered_edges
                    if not new_edges:
                        continue

                    # Base score: number of new edges covered
                    base_score = len(new_edges)

                    # Penalty: conflicts with already selected cliques
                    penalty = 0
                    for j in range(m):
                        if selected_mask[j] and conflict[i][j] > 0:
                            penalty += lambda_val * conflict[i][j]

                    score = base_score - penalty

                    if score > best_score:
                        best_score = score
                        best_idx = i

                if best_idx == -1 or best_score <= 0:
                    # No good candidates left, take any that covers uncovered edges
                    for i in range(m):
                        if selected_mask[i]:
                            continue
                        new_edges = cand_edge_sets[i] - covered_edges
                        if new_edges:
                            best_idx = i
                            break
                    if best_idx == -1:
                        break

                # Select this clique
                selected_mask[best_idx] = True
                selected.append(best_idx)
                covered_edges.update(cand_edge_sets[best_idx])

                # Adjust lambda: if we're stuck, increase penalty
                if len(selected) > 0 and not (all_edges - covered_edges):
                    break

            # Build result
            result = [candidates[i] for i in selected]

            # Check if all edges are covered; if not, add missing edges as K2
            if covered_edges != all_edges:
                missing = all_edges - covered_edges
                for u, v in missing:
                    result.append([u, v])

            return result

        def validate(cliques, adj_mask_orig, n):
            for C in cliques:
                if len(C) < 2:
                    return False
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        if not (adj_mask_orig[u] & (1 << v)):
                            return False
            edge_set = set()
            for C in cliques:
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        u, v = C[i], C[j]
                        key = (min(u, v), max(u, v))
                        if key in edge_set:
                            return False
                        edge_set.add(key)
            for u in range(n):
                for v in iter_bits(adj_mask_orig[u]):
                    if v > u:
                        if (u, v) not in edge_set:
                            return False
            return True

        def greedy_partition_simple(adj_mask_orig, n, rng, deadline, strategy):
            """Simple greedy partition for fallback."""
            adj = list(adj_mask_orig)
            cliques = []

            while True:
                if time.time() > deadline:
                    break

                current_edges = []
                for u in range(n):
                    m = adj[u] & ~((1 << (u + 1)) - 1)
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        current_edges.append((u, v))
                        m ^= lsb

                if not current_edges:
                    break

                if strategy == 0:
                    best_edge = None
                    best_common = -1
                    for u, v in current_edges:
                        common = (adj[u] & adj[v]).bit_count()
                        if common > best_common:
                            best_common = common
                            best_edge = (u, v)
                elif strategy == 1:
                    best_edge = None
                    best_common = float('inf')
                    for u, v in current_edges:
                        common = (adj[u] & adj[v]).bit_count()
                        if common < best_common:
                            best_common = common
                            best_edge = (u, v)
                else:
                    best_edge = rng.choice(current_edges)

                if best_edge is None:
                    break

                u, v = best_edge
                C = expand_clique_simple(u, v, adj, n, rng)
                cliques.append(C)

                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        a, b = C[i], C[j]
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)

            # Handle remaining
            for u in range(n):
                m = adj[u] & ~((1 << (u + 1)) - 1)
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    cliques.append([u, v])
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    m ^= lsb

            return cliques

        best_cliques = None
        best_count = float('inf')

        # Phase 1: Generate candidate cliques using multiple strategies
        gen_deadline = start_time + time_budget * 0.4
        all_candidates = []

        strategies = [0, 1, 2, 0, 1, 2]
        for s in strategies:
            if time.time() > gen_deadline:
                break
            rng = random.Random(seed * 1000 + s)
            cands = generate_candidate_cliques(adj_mask_orig, n, rng, gen_deadline, s)
            all_candidates.extend(cands)

        # Phase 2: Use Lagrangian relaxation to select best subset
        if all_candidates:
            ls_deadline = start_time + time_budget * 0.8
            rng_ls = random.Random(seed * 2000)
            selected = lagrangian_select(all_candidates, adj_mask_orig, n, rng_ls, ls_deadline)
            if validate(selected, adj_mask_orig, n):
                best_cliques = selected
                best_count = len(selected)

        # Phase 3: Try greedy partitions as fallback/comparison
        remaining_budget = time_budget - (time.time() - start_time)
        if remaining_budget > 2:
            num_restarts = max(2, int(remaining_budget / 1.5))
            rng_master = random.Random(seed + 5000)

            for r in range(num_restarts):
                if time.time() > start_time + time_budget - 1.0:
                    break
                rng = random.Random(rng_master.randint(0, 2**31))
                strat = r % 3
                deadline = min(start_time + time_budget - 1.0, time.time() + max(0.3, remaining_budget / (num_restarts - r)))
                try:
                    cliques = greedy_partition_simple(adj_mask_orig, n, rng, deadline, strat)
                    if validate(cliques, adj_mask_orig, n):
                        if len(cliques) < best_count:
                            best_count = len(cliques)
                            best_cliques = cliques
                except:
                    continue

        # Final fallback
        if best_cliques is None:
            best_cliques = []
            for u, v in all_edges:
                best_cliques.append([u, v])

        return best_cliques

    results = []
    # Instance 1: n=32
    t1 = min(50, TOTAL_BUDGET * 0.35)
    results.append(solve_instance(32, 0.6, 601, t1))

    # Instance 2: n=64
    elapsed = time.time() - start_time
    t2 = max(20, TOTAL_BUDGET - elapsed - 2)
    results.append(solve_instance(64, 0.55, 602, t2))

    return results

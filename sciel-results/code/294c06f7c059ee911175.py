"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit

    n = 14
    m = 35
    max_val = 10000
    all_pairs = list(combinations(range(n), 2))
    num_pairs = len(all_pairs)  # 91

    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def greedy_edges_for_values(values, rng):
        """Greedy edge selection: add edges that minimize new distinct sums, tie-break by products."""
        pair_info = []
        for idx, (u, v) in enumerate(all_pairs):
            s = values[u] + values[v]
            p = values[u] * values[v]
            pair_info.append((idx, u, v, s, p))

        sums = set()
        prods = set()
        selected = []

        # Iteratively pick the edge that introduces the fewest new distinct values
        for _ in range(m):
            best_idx = -1
            best_key = None
            for idx, u, v, s, p in pair_info:
                if any(idx == si for si, _, _, _ in selected):
                    continue
                new_sums = len(sums) if s in sums else len(sums) + 1
                new_prods = len(prods) if p in prods else len(prods) + 1
                key = (max(new_sums, new_prods), new_sums, new_prods, idx)
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx
            if best_idx < 0:
                break
            idx, u, v, s, p = pair_info[best_idx]
            selected.append((idx, u, v, s))
            sums.add(s)
            prods.add(p)

        edges = [(u, v) for _, u, v, _ in selected]
        return edges

    def clique_search(values, K, max_nodes=500000):
        """
        Find 35 edges such that #distinct sums <= K and #distinct products <= K.
        Uses branch-and-bound maximum clique on a compatibility graph.
        Two edges are compatible if adding both keeps distinct sums and products <= K.
        """
        # Precompute sum and product for each pair
        pair_sums = [values[u] + values[v] for u, v in all_pairs]
        pair_prods = [values[u] * values[v] for u, v in all_pairs]

        # Build compatibility: edge i is compatible with edge j if
        # when we have both, the total distinct sums and products are <= K.
        # This is NOT pairwise independent (it depends on the full set), so
        # we use a relaxed condition: for any two edges i,j, if their sums are
        # distinct AND products are distinct, they "use up" 2 sum-slots and 2
        # product-slots. We need a global check.
        #
        # Simpler approach: use a greedy + local search that directly tracks
        # the distinct sets, which is more practical for K around 10-20.

        # Greedy construction with backtracking-lite
        best_edges = None
        best_score = float('inf')

        rng = random.Random(hash(tuple(sorted(values))) % 1000003)

        for attempt in range(30):
            if time.time() > deadline:
                return None, float('inf')

            # Shuffle and greedily add edges
            order = list(range(num_pairs))
            rng.shuffle(order)

            edges = []
            sums = set()
            prods = set()

            for idx in order:
                if len(edges) >= m:
                    break
                u, v = all_pairs[idx]
                s = pair_sums[idx]
                p = pair_prods[idx]
                new_sums = len(sums) if s in sums else len(sums) + 1
                new_prods = len(prods) if p in prods else len(prods) + 1
                if max(new_sums, new_prods) <= K:
                    edges.append(idx)
                    sums.add(s)
                    prods.add(p)

            if len(edges) < m:
                continue

            score = max(len(sums), len(prods))
            if score < best_score:
                best_score = score
                best_edges = list(edges)

            # Local search: try to swap edges to reduce score
            for _ in range(200):
                if time.time() > deadline:
                    break
                improved = False
                idx = rng.randint(0, m - 1)
                old_idx = edges[idx]
                old_s = pair_sums[old_idx]
                old_p = pair_prods[old_idx]

                sum_counts = {}
                prod_counts = {}
                for e in edges:
                    sum_counts[pair_sums[e]] = sum_counts.get(pair_sums[e], 0) + 1
                    prod_counts[pair_prods[e]] = prod_counts.get(pair_prods[e], 0) + 1

                for _ in range(20):
                    new_idx = rng.randint(0, num_pairs - 1)
                    if new_idx == old_idx or new_idx in edges:
                        continue
                    new_s = pair_sums[new_idx]
                    new_p = pair_prods[new_idx]

                    # Check if this swap keeps us within K
                    sc = dict(sum_counts)
                    pc = dict(prod_counts)
                    sc[old_s] -= 1
                    pc[old_p] -= 1
                    if sc[old_s] == 0:
                        del sc[old_s]
                    if pc[old_p] == 0:
                        del pc[old_p]
                    sc[new_s] = sc.get(new_s, 0) + 1
                    pc[new_p] = pc.get(new_p, 0) + 1

                    new_score = max(len(sc), len(pc))
                    if new_score < score:
                        edges[idx] = new_idx
                        sum_counts = sc
                        prod_counts = pc
                        score = new_score
                        improved = True
                        break

                if not improved:
                    continue

            if score <= K:
                return edges, score

        if best_edges is not None:
            return best_edges, best_score
        return None, float('inf')

    def generate_structured_values(rng):
        """Generate values with shared structure to reduce product diversity."""
        strategy = rng.randint(0, 4)

        if strategy == 0:
            # Powers of 2 with small multipliers
            base = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
            values = base[:13] + [rng.randint(1, 100)]
            values = sorted(set(values))
            while len(values) < 14:
                values.append(rng.randint(1, max_val))
                values = sorted(set(values))
            return values[:14]

        elif strategy == 1:
            # Multiples of a small number
            base_mult = rng.choice([2, 3, 4, 5, 6])
            values = []
            for i in range(1, 20):
                v = base_mult * i
                if v <= max_val and v not in values:
                    values.append(v)
                if len(values) == 14:
                    break
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
            return values[:14]

        elif strategy == 2:
            # Values of form 2^a * 3^b
            values = set()
            for a in range(0, 8):
                for b in range(0, 4):
                    v = (2 ** a) * (3 ** b)
                    if 1 <= v <= max_val:
                        values.add(v)
            values = list(values)
            rng.shuffle(values)
            values = values[:14]
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
            return values[:14]

        elif strategy == 3:
            # Small consecutive multiples
            start = rng.randint(1, 5)
            step = rng.randint(1, 3)
            values = []
            v = start
            while len(values) < 14 and v <= max_val:
                values.append(v)
                v += step
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
            return values[:14]

        else:
            # Random but biased toward small numbers
            values = []
            for _ in range(14):
                v = rng.randint(1, 100)
                if v not in values:
                    values.append(v)
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
            return values[:14]

    def optimize_values(values, edges, rng, max_iters=200):
        """Try swapping pairs of values to improve score."""
        best_values = list(values)
        best_score = evaluate(best_values, edges)

        for _ in range(max_iters):
            if time.time() > deadline:
                break
            i, j = rng.sample(range(n), 2)
            new_values = list(best_values)
            new_values[i], new_values[j] = new_values[j], new_values[i]

            if len(set(new_values)) != 14:
                continue
            if any(v < 1 or v > max_val for v in new_values):
                continue

            new_score = evaluate(new_values, edges)
            if new_score < best_score:
                best_values = new_values
                best_score = new_score

        return best_values, best_score

    # Main search loop
    rng = random.Random(42)
    best_result = None
    best_score = float('inf')

    # Phase 1: Try K-escalation with clique/greedy search
    for K in range(10, 25):
        if time.time() > deadline:
            break

        found = False
        for restart in range(15):
            if time.time() > deadline:
                break

            values = generate_structured_values(rng)
            values = list(set(values))
            while len(values) < 14:
                v = rng.randint(1, max_val)
                if v not in values:
                    values.append(v)
            values = values[:14]

            edges_idx, score = clique_search(values, K)
            if edges_idx is not None and score < best_score:
                edges = [list(all_pairs[i]) for i in edges_idx]
                best_score = score
                best_result = (list(values), edges)
                found = True
                if best_score <= K:
                    break

        if found and best_score <= K:
            # We found a solution with score <= K, try to improve
            if time.time() > deadline:
                break

    # Phase 2: Refine best result with local search
    if best_result is not None:
        values, edges = best_result

        # Local search on edges
        edges_set = set(tuple(sorted(e)) for e in edges)
        for _ in range(300):
            if time.time() > deadline:
                break
            idx = rng.randint(0, m - 1)
            old_edge = tuple(sorted(edges[idx]))

            sum_counts = {}
            prod_counts = {}
            for u, v in edges:
                s = values[u] + values[v]
                p = values[u] * values[v]
                sum_counts[s] = sum_counts.get(s, 0) + 1
                prod_counts[p] = prod_counts.get(p, 0) + 1

            for _ in range(30):
                new_u, new_v = rng.choice(all_pairs)
                new_edge = tuple(sorted([new_u, new_v]))
                if new_edge in edges_set or new_edge == old_edge:
                    continue

                os_ = values[old_edge[0]] + values[old_edge[1]]
                op_ = values[old_edge[0]] * values[old_edge[1]]
                ns_ = values[new_u] + values[new_v]
                np_ = values[new_u] * values[new_v]

                sc = dict(sum_counts)
                pc = dict(prod_counts)
                sc[os_] -= 1
                pc[op_] -= 1
                if sc[os_] == 0:
                    del sc[os_]
                if pc[op_] == 0:
                    del pc[op_]
                sc[ns_] = sc.get(ns_, 0) + 1
                pc[np_] = pc.get(np_, 0) + 1

                new_score = max(len(sc), len(pc))
                cur_score = max(len(sum_counts), len(prod_counts))
                if new_score < cur_score:
                    edges[idx] = [new_u, new_v]
                    edges_set.discard(old_edge)
                    edges_set.add(new_edge)
                    sum_counts = sc
                    prod_counts = pc
                    break

        # Local search on values
        values, score = optimize_values(values, edges, rng, max_iters=100)

        # Re-optimize edges after value changes
        for _ in range(200):
            if time.time() > deadline:
                break
            idx = rng.randint(0, m - 1)
            old_edge = tuple(sorted(edges[idx]))
            edges_set = set(tuple(sorted(e)) for e in edges)

            sum_counts = {}
            prod_counts = {}
            for u, v in edges:
                s = values[u] + values[v]
                p = values[u] * values[v]
                sum_counts[s] = sum_counts.get(s, 0) + 1
                prod_counts[p] = prod_counts.get(p, 0) + 1

            for _ in range(30):
                new_u, new_v = rng.choice(all_pairs)
                new_edge = tuple(sorted([new_u, new_v]))
                if new_edge in edges_set or new_edge == old_edge:
                    continue

                os_ = values[old_edge[0]] + values[old_edge[1]]
                op_ = values[old_edge[0]] * values[old_edge[1]]
                ns_ = values[new_u] + values[new_v]
                np_ = values[new_u] * values[new_v]

                sc = dict(sum_counts)
                pc = dict(prod_counts)
                sc[os_] -= 1
                pc[op_] -= 1
                if sc[os_] == 0:
                    del sc[os_]
                if pc[op_] == 0:
                    del pc[op_]
                sc[ns_] = sc.get(ns_, 0) + 1
                pc[np_] = pc.get(np_, 0) + 1

                new_score = max(len(sc), len(pc))
                cur_score = max(len(sum_counts), len(prod_counts))
                if new_score < cur_score:
                    edges[idx] = [new_u, new_v]
                    edges_set.discard(old_edge)
                    edges_set.add(new_edge)
                    sum_counts = sc
                    prod_counts = pc
                    break

        final_score = evaluate(values, edges)
        if final_score < best_score:
            best_score = final_score
            best_result = (list(values), [list(e) for e in edges])

    # Phase 3: Additional random restarts with greedy + local search
    for restart in range(100):
        if time.time() > deadline:
            break

        values = generate_structured_values(rng)
        values = list(set(values))
        while len(values) < 14:
            v = rng.randint(1, max_val)
            if v not in values:
                values.append(v)
        values = values[:14]

        edges = greedy_edges_for_values(values, rng)
        score = evaluate(values, edges)

        # Local search on edges
        edges_set = set(tuple(sorted(e)) for e in edges)
        for _ in range(200):
            if time.time() > deadline:
                break
            idx = rng.randint(0, m - 1)
            old_edge = tuple(sorted(edges[idx]))

            sum_counts = {}
            prod_counts = {}
            for u, v in edges:
                s = values[u] + values[v]
                p = values[u] * values[v]
                sum_counts[s] = sum_counts.get(s, 0) + 1
                prod_counts[p] = prod_counts.get(p, 0) + 1

            for _ in range(30):
                new_u, new_v = rng.choice(all_pairs)
                new_edge = tuple(sorted([new_u, new_v]))
                if new_edge in edges_set or new_edge == old_edge:
                    continue

                os_ = values[old_edge[0]] + values[old_edge[1]]
                op_ = values[old_edge[0]] * values[old_edge[1]]
                ns_ = values[new_u] + values[new_v]
                np_ = values[new_u] * values[new_v]

                sc = dict(sum_counts)
                pc = dict(prod_counts)
                sc[os_] -= 1
                pc[op_] -= 1
                if sc[os_] == 0:
                    del sc[os_]
                if pc[op_] == 0:
                    del pc[op_]
                sc[ns_] = sc.get(ns_, 0) + 1
                pc[np_] = pc.get(np_, 0) + 1

                new_score = max(len(sc), len(pc))
                cur_score = max(len(sum_counts), len(prod_counts))
                if new_score < cur_score:
                    edges[idx] = [new_u, new_v]
                    edges_set.discard(old_edge)
                    edges_set.add(new_edge)
                    sum_counts = sc
                    prod_counts = pc
                    break

        score = evaluate(values, edges)
        if score < best_score:
            best_score = score
            best_result = (list(values), [list(e) for e in edges])
            if best_score <= 12:
                break

    if best_result is None:
        values = list(range(1, 15))
        edges = [list(e) for e in all_pairs[:m]]
    else:
        values, edges = best_result

    # Final validation
    assert len(values) == 14
    assert len(set(values)) == 14
    assert all(isinstance(v, int) and 1 <= v <= max_val for v in values)
    assert len(edges) == m
    edge_set = set()
    for e in edges:
        assert len(e) == 2
        assert e[0] != e[1]
        assert 0 <= e[0] < n and 0 <= e[1] < n
        key = tuple(sorted(e))
        assert key not in edge_set
        edge_set.add(key)

    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END

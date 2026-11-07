import time
import random
import math
from itertools import combinations
from collections import Counter

DEADLINE = 85.0
START_TIME = time.time()


def _time_left():
    return DEADLINE - (time.time() - START_TIME)


def _score_from_counts(sum_counts, prod_counts):
    ds = sum(1 for v in sum_counts.values() if v > 0)
    dp = sum(1 for v in prod_counts.values() if v > 0)
    return max(ds, dp)


def _evaluate(values, edges):
    sc = Counter()
    pc = Counter()
    for i, j in edges:
        sc[values[i] + values[j]] += 1
        pc[values[i] * values[j]] += 1
    return _score_from_counts(sc, pc)


def _edge_swap(values, edges_set, sum_counts, prod_counts, edge_sums, edge_prods,
               all_edges, all_sums, all_prods, rng):
    n = len(values)
    num_all = len(all_edges)
    in_list = list(edges_set)
    if not in_list:
        return False
    idx_in = rng.choice(in_list)
    # Pick an out edge not currently in the set
    # We need an efficient way. Since m=35 and num_all=91, we can build a list
    # But for speed in SA, let's sample
    # Actually, let's just pick a random edge from all, if it's in, pick another
    # Or maintain a bitset. 91 bits fits in an int.
    # Let's use a simple rejection sampling
    while True:
        idx_out = rng.randint(0, num_all - 1)
        if idx_out not in edges_set:
            break

    s_in = all_sums[idx_in]
    p_in = all_prods[idx_in]
    s_out = all_sums[idx_out]
    p_out = all_prods[idx_out]

    # Delta calculation
    ds_in = sum_counts[s_in]
    ds_out = sum_counts[s_out]
    dp_in = prod_counts[p_in]
    dp_out = prod_counts[p_out]

    d_sums = (1 if ds_in == 1 else 0) + (1 if ds_out == 0 else 0)
    d_prods = (1 if dp_in == 1 else 0) + (1 if dp_out == 0 else 0)

    # Apply
    sum_counts[s_in] = ds_in - 1
    if ds_in == 1:
        del sum_counts[s_in]
    sum_counts[s_out] = ds_out + 1
    prod_counts[p_in] = dp_in - 1
    if dp_in == 1:
        del prod_counts[p_in]
    prod_counts[p_out] = dp_out + 1

    edges_set.remove(idx_in)
    edges_set.add(idx_out)

    return True, s_in, ds_in, s_out, ds_out, p_in, dp_in, p_out, dp_out


def _sa_optimize_edges(values, m, deadline):
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    num_edges = len(all_edges)
    all_sums = [values[i] + values[j] for i, j in all_edges]
    all_prods = [values[i] * values[j] for i, j in all_edges]

    # Build collision groups for sums and products
    sum_groups = {}
    prod_groups = {}
    for idx in range(num_edges):
        s = all_sums[idx]
        p = all_prods[idx]
        sum_groups.setdefault(s, []).append(idx)
        prod_groups.setdefault(p, []).append(idx)

    # Sort groups by size descending to prefer dense collisions
    sum_groups_sorted = sorted(sum_groups.values(), key=len, reverse=True)
    prod_groups_sorted = sorted(prod_groups.values(), key=len, reverse=True)

    rng = random.Random(42)

    best_score = float('inf')
    best_edges = None

    # Restart with different strategies
    for restart in range(12):
        if time.time() > deadline:
            break

        edges_set = set()
        sum_counts = Counter()
        prod_counts = Counter()

        if restart == 0:
            # Greedy: pick edges from largest groups
            # We want to pick 35 edges.
            # Let's just greedily add edges that minimize the increase in distinct count
            # But that's expensive.
            # Better: Start with empty, add edges.
            # To speed up, we can pre-sort edges by "potential collision"
            # Actually, let's just use a simple greedy:
            # At each step, pick the edge that adds the least new distinct sums/prods.
            remaining = list(range(num_edges))
            rng.shuffle(remaining)
            for _ in range(m):
                best_idx = -1
                best_delta = float('inf')
                for idx in remaining:
                    s = all_sums[idx]
                    p = all_prods[idx]
                    # Cost of adding this edge
                    cost = (1 if sum_counts[s] == 0 else 0) + (1 if prod_counts[p] == 0 else 0)
                    if cost < best_delta:
                        best_delta = cost
                        best_idx = idx
                if best_idx != -1:
                    s = all_sums[best_idx]
                    p = all_prods[best_idx]
                    sum_counts[s] += 1
                    prod_counts[p] += 1
                    edges_set.add(best_idx)
                    remaining.remove(best_idx)
        elif restart == 1:
            # Random
            edges_set = set(rng.sample(range(num_edges), m))
            for idx in edges_set:
                sum_counts[all_sums[idx]] += 1
                prod_counts[all_prods[idx]] += 1
        else:
            # Perturb best
            if best_edges is None:
                continue
            edges_set = set(best_edges)
            for idx in edges_set:
                sum_counts[all_sums[idx]] += 1
                prod_counts[all_prods[idx]] += 1
            # Random swaps
            k = 3 + restart
            for _ in range(k):
                if not edges_set:
                    continue
                in_list = list(edges_set)
                idx_in = rng.choice(in_list)
                while True:
                    idx_out = rng.randint(0, num_edges - 1)
                    if idx_out not in edges_set:
                        break
                s_in = all_sums[idx_in]
                p_in = all_prods[idx_in]
                s_out = all_sums[idx_out]
                p_out = all_prods[idx_out]
                sum_counts[s_in] -= 1
                if sum_counts[s_in] == 0: del sum_counts[s_in]
                prod_counts[p_in] -= 1
                if prod_counts[p_in] == 0: del prod_counts[p_in]
                edges_set.remove(idx_in)
                sum_counts[s_out] += 1
                prod_counts[p_out] += 1
                edges_set.add(idx_out)

        current_score = _score_from_counts(sum_counts, prod_counts)
        if current_score < best_score:
            best_score = current_score
            best_edges = set(edges_set)

        # SA
        T = 3.0
        T_min = 0.01
        alpha = 0.997
        iters = 0
        max_iters = 5000
        no_improve = 0
        patience = 800

        while iters < max_iters:
            if time.time() > deadline:
                break

            # Pick an edge in, pick an edge out
            in_list = list(edges_set)
            if not in_list:
                break
            idx_in = rng.choice(in_list)
            while True:
                idx_out = rng.randint(0, num_edges - 1)
                if idx_out not in edges_set:
                    break

            s_in = all_sums[idx_in]
            p_in = all_prods[idx_in]
            s_out = all_sums[idx_out]
            p_out = all_prods[idx_out]

            ds_in = sum_counts[s_in]
            ds_out = sum_counts[s_out]
            dp_in = prod_counts[p_in]
            dp_out = prod_counts[p_out]

            # Calculate delta in distinct sums and prods
            d_sums = (1 if ds_in == 1 else 0) + (1 if ds_out == 0 else 0)
            d_prods = (1 if dp_in == 1 else 0) + (1 if dp_out == 0 else 0)

            # Apply tentatively
            sum_counts[s_in] = ds_in - 1
            if ds_in == 1: del sum_counts[s_in]
            sum_counts[s_out] = ds_out + 1
            prod_counts[p_in] = dp_in - 1
            if dp_in == 1: del prod_counts[p_in]
            prod_counts[p_out] = dp_out + 1

            new_score = _score_from_counts(sum_counts, prod_counts)
            accept = False
            if new_score < current_score:
                accept = True
            elif new_score == current_score:
                accept = True
            else:
                delta = new_score - current_score
                accept = rng.random() < math.exp(-delta / max(T, 1e-10))

            if accept:
                edges_set.remove(idx_in)
                edges_set.add(idx_out)
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(edges_set)
            else:
                # Revert
                sum_counts[s_out] = ds_out
                if ds_out == 0: del sum_counts[s_out]
                sum_counts[s_in] = ds_in
                prod_counts[p_out] = dp_out
                if dp_out == 0: del prod_counts[p_out]
                prod_counts[p_in] = dp_in
                no_improve += 1

            T *= alpha
            if T < T_min: T = T_min
            iters += 1
            if no_improve > patience:
                break

    if best_edges is None:
        best_edges = set(range(m))
        best_score = _evaluate(values, [all_edges[i] for i in best_edges])

    return [list(all_edges[idx]) for idx in best_edges], best_score


def _generate_candidates():
    candidates = []
    # Smooth numbers
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36])
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 48])
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24, 27])
    candidates.append([1, 2, 3, 4, 6, 9, 12, 16, 18, 24, 27, 32, 36, 48])
    candidates.append([1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 18, 20, 24, 27])
    candidates.append([1, 2, 3, 4, 6, 9, 12, 15, 18, 20, 24, 27, 30, 36])
    candidates.append([1, 2, 3, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30])
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 27, 36, 48, 54])
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18])
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 36, 48])
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16])
    candidates.append(list(range(1, 15)))
    candidates.append(list(range(2, 16)))
    # Random
    rng = random.Random(123)
    for _ in range(10):
        vals = set()
        while len(vals) < 14:
            if rng.random() < 0.6:
                vals.add(rng.randint(1, 20))
            else:
                vals.add(rng.randint(1, 50))
        if len(vals) == 14:
            candidates.append(sorted(vals))
    return candidates


def construct():
    start = time.time()
    n = 14
    m = 35
    candidates = _generate_candidates()
    best_score = float('inf')
    best_result = None

    for values in candidates:
        if time.time() - start > DEADLINE - 5:
            break
        if len(values) != n or len(set(values)) != n:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
        deadline_for_this = min(start + DEADLINE - 2, time.time() + 6.0)
        edges, score = _sa_optimize_edges(values, m, deadline_for_this)
        if score < best_score:
            best_score = score
            best_result = {
                'values': [int(v) for v in values],
                'edges': [[int(i), int(j)] for i, j in edges]
            }

    if best_result is None:
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
        best_result = {'values': values, 'edges': edges}

    return best_result

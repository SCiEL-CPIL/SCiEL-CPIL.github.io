import time
import random
import itertools
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    n = 14
    m = 35

    all_pairs = list(itertools.combinations(range(n), 2))
    num_pairs = len(all_pairs)  # 91

    def calc_score(vals, edges):
        """Compute max(distinct_sums, distinct_prods) for given values and edges."""
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(vals[i] + vals[j])
            prods.add(vals[i] * vals[j])
        return max(len(sums), len(prods))

    def precompute_pair_data(vals):
        """Precompute sum and product for all pairs."""
        pair_sums = {}
        pair_prods = {}
        for i, j in all_pairs:
            pair_sums[(i, j)] = vals[i] + vals[j]
            pair_prods[(i, j)] = vals[i] * vals[j]
        return pair_sums, pair_prods

    def select_best_edges(vals, pair_sums, pair_prods, m=35):
        """Select m edges that minimize max(distinct_sums, distinct_prods).
        Strategy: rank pairs by collision frequency, then greedily pick."""
        # Count how many pairs share each sum and product
        sum_count = Counter()
        prod_count = Counter()
        for (i, j) in all_pairs:
            s = pair_sums[(i, j)]
            p = pair_prods[(i, j)]
            sum_count[s] += 1
            prod_count[p] += 1

        # Score each pair by total collision potential
        pair_scores = []
        for (i, j) in all_pairs:
            s = pair_sums[(i, j)]
            p = pair_prods[(i, j)]
            score = sum_count[s] + prod_count[p]
            pair_scores.append((score, i, j))

        # Sort by score descending, break ties by smaller sum (more likely to collide)
        pair_scores.sort(key=lambda x: (-x[0], pair_sums[(x[1], x[2])]))

        # Greedy selection with collision tracking
        selected = []
        seen_sums = set()
        seen_prods = set()
        selected_set = set()

        # First pass: pick top-scoring pairs that don't add new distinct values
        for score, i, j in pair_scores:
            if len(selected) >= m:
                break
            key = frozenset((i, j))
            if key in selected_set:
                continue
            s = pair_sums[(i, j)]
            p = pair_prods[(i, j)]
            # Prefer pairs whose sum and product are already seen
            if s in seen_sums and p in seen_prods:
                selected.append([i, j])
                selected_set.add(key)
            elif s in seen_sums or p in seen_prods:
                selected.append([i, j])
                seen_sums.add(s)
                seen_prods.add(p)
                selected_set.add(key)
            elif len(selected) < m:
                # Still need to fill, add this
                selected.append([i, j])
                seen_sums.add(s)
                seen_prods.add(p)
                selected_set.add(key)

        # If we don't have enough, fill with remaining
        if len(selected) < m:
            for score, i, j in pair_scores:
                if len(selected) >= m:
                    break
                key = frozenset((i, j))
                if key not in selected_set:
                    selected.append([i, j])
                    seen_sums.add(pair_sums[(i, j)])
                    seen_prods.add(pair_prods[(i, j)])
                    selected_set.add(key)

        return selected[:m]

    def greedy_edge_optimize(vals, pair_sums, pair_prods, max_rounds=100):
        """Iteratively improve edge selection by swapping edges."""
        edges = select_best_edges(vals, pair_sums, pair_prods, m)
        current_score = calc_score(vals, edges)

        if current_score <= 2:
            return edges, current_score

        selected_set = set(frozenset(e) for e in edges)
        candidate_pairs = [(i, j) for (i, j) in all_pairs if frozenset((i, j)) not in selected_set]

        for rnd in range(max_rounds):
            if time.time() > deadline - 3:
                break

            improved = False
            random.shuffle(candidate_pairs)

            for new_i, new_j in candidate_pairs:
                if time.time() > deadline - 3:
                    break

                new_key = frozenset((new_i, new_j))
                if new_key in selected_set:
                    continue

                new_s = pair_sums[(new_i, new_j)]
                new_p = pair_prods[(new_i, new_j)]

                for k in range(m):
                    old_i, old_j = edges[k]
                    old_key = frozenset((old_i, old_j))
                    if old_key == new_key:
                        continue

                    old_s = pair_sums[(old_i, old_j)]
                    old_p = pair_prods[(old_i, old_j)]

                    # Quick check: would this swap reduce distinct count?
                    # Count how many edges use old_s and old_p
                    old_s_count = sum(1 for e in edges if pair_sums[tuple(e)] == old_s)
                    old_p_count = sum(1 for e in edges if pair_prods[tuple(e)] == old_p)
                    new_s_count = sum(1 for e in edges if pair_sums[tuple(e)] == new_s)
                    new_p_count = sum(1 for e in edges if pair_prods[tuple(e)] == new_p)

                    # If removing old edge eliminates a unique sum/prod, and new edge adds unique, skip
                    if old_s_count == 1 and new_s_count == 0:
                        continue  # would add new sum
                    if old_p_count == 1 and new_p_count == 0:
                        continue  # would add new prod

                    # Do full evaluation
                    temp_edges = edges[:]
                    temp_edges[k] = [new_i, new_j]
                    temp_set = set(frozenset(e) for e in temp_edges)
                    if len(temp_set) != m:
                        continue

                    new_score = calc_score(vals, temp_edges)
                    if new_score < current_score:
                        edges = temp_edges
                        current_score = new_score
                        selected_set = temp_set
                        candidate_pairs = [(i, j) for (i, j) in all_pairs if frozenset((i, j)) not in selected_set]
                        improved = True
                        break

                if improved:
                    break

            if not improved:
                break

        return edges, current_score

    def generate_value_sets():
        """Generate diverse value sets designed for high collision."""
        sets = []

        # 1. Powers of 2 (product collisions: 2^a * 2^b = 2^(a+b))
        sets.append([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])

        # 2. Small consecutive (sum collisions: many pairs sum to same value)
        sets.append(list(range(1, 15)))

        # 3. Multiplicative grid: products of small numbers
        sets.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72])

        # 4. Geometric with ratio 3
        sets.append([1, 3, 9, 27, 81, 243, 729, 2187, 6561])  # only 9, need 14
        # Pad with small values
        sets.append([1, 2, 3, 6, 9, 18, 27, 54, 81, 162, 243, 486, 729, 1458])

        # 5. Arithmetic progression with small step
        sets.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

        # 6. Mixed: some powers, some small
        sets.append([1, 2, 4, 8, 16, 3, 6, 12, 24, 5, 10, 20, 40, 80])

        # 7. All even small numbers (product always even, more collisions)
        sets.append([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])

        # 8. Fibonacci-like
        sets.append([1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377])
        # Fix duplicate
        sets[-1] = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

        # 9. Random small values
        for seed in range(5):
            random.seed(seed * 12345 + 678)
            v = random.sample(range(1, 30), n)
            v.sort()
            sets.append(v)

        # 10. Clustered values (many close together)
        sets.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        sets.append([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

        # 11. Powers of 2 with small multiples
        sets.append([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128])

        # 12. Square numbers
        sets.append([1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196])

        # 13. Triangular numbers
        sets.append([1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105])

        # 14. Random geometric
        for seed in range(3):
            random.seed(seed * 999 + 1)
            base = random.uniform(1.1, 1.5)
            v = [int(base ** i) for i in range(n)]
            # Ensure distinct
            seen = set()
            final = []
            for x in v:
                if x not in seen and 1 <= x <= 10000:
                    final.append(x)
                    seen.add(x)
            while len(final) < n:
                final.append(len(final) + 1)
            sets.append(sorted(final))

        return sets

    best_score = float('inf')
    best_values = None
    best_edges = None

    value_sets = generate_value_sets()

    for vals in value_sets:
        if time.time() > deadline - 5:
            break

        # Validate
        if len(set(vals)) != n or any(v < 1 or v > 10000 for v in vals):
            continue

        pair_sums, pair_prods = precompute_pair_data(vals)
        edges, score = greedy_edge_optimize(vals, pair_sums, pair_prods)

        if score < best_score:
            best_score = score
            best_values = vals[:]
            best_edges = [e[:] for e in edges]

        # Joint optimization: perturb values and re-optimize edges
        if time.time() < deadline - 8 and score <= best_score:
            current_vals = vals[:]
            current_edges = edges[:]
            current_score = score

            for _ in range(50):
                if time.time() > deadline - 5:
                    break

                idx = random.randint(0, n - 1)
                old_val = current_vals[idx]

                # Try nearby values and random small values
                candidates = set()
                for d in range(-5, 6):
                    nv = old_val + d
                    if 1 <= nv <= 10000:
                        candidates.add(nv)
                candidates.add(random.randint(1, 50))
                candidates.add(random.randint(1, 100))

                for new_val in candidates:
                    if new_val == old_val or new_val in current_vals:
                        continue

                    temp_vals = current_vals[:]
                    temp_vals[idx] = new_val
                    temp_vals.sort()

                    if len(set(temp_vals)) != n:
                        continue

                    tp_sums, tp_prods = precompute_pair_data(temp_vals)
                    te, ts = greedy_edge_optimize(temp_vals, tp_sums, tp_prods, max_rounds=30)

                    if ts < current_score:
                        current_vals = temp_vals
                        current_edges = te
                        current_score = ts
                        if current_score < best_score:
                            best_score = current_score
                            best_values = current_vals[:]
                            best_edges = [e[:] for e in current_edges]
                        break

    # Final local search on best solution
    if best_values is not None and time.time() < deadline - 3:
        pair_sums, pair_prods = precompute_pair_data(best_values)
        final_edges, final_score = greedy_edge_optimize(best_values, pair_sums, pair_prods, max_rounds=200)
        if final_score < best_score:
            best_score = final_score
            best_edges = final_edges

    # Fallback
    if best_values is None:
        best_values = list(range(1, n + 1))
        pair_sums, pair_prods = precompute_pair_data(best_values)
        best_edges, _ = select_best_edges(best_values, pair_sums, pair_prods)

    # Final type conversion
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]

    return {'values': best_values, 'edges': best_edges}

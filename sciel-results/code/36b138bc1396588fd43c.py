"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import defaultdict, Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 14
    m = 35
    maximum = 10000

    all_pairs = list(combinations(range(n), 2))
    all_pairs_set = set(all_pairs)

    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    def _pair_props(values):
        """Precompute sum and product for all pairs."""
        props = {}
        for i, j in all_pairs:
            props[(i, j)] = (values[i] + values[j], values[i] * values[j])
        return props

    def _greedy_construct(values, props, time_limit):
        """Greedily pick edges that reuse existing sums/products."""
        sum_counts = defaultdict(int)
        prod_counts = defaultdict(int)
        edges = []
        edge_set = set()

        for _ in range(m):
            if time.time() > time_limit:
                break
            best_edge = None
            best_cost = float('inf')
            best_tie = -1

            for i, j in all_pairs:
                if (i, j) in edge_set:
                    continue
                s, p = props[(i, j)]
                cost = 0
                if s not in sum_counts:
                    cost += 1
                if p not in prod_counts:
                    cost += 1
                # Tie-breaker: prefer edges with higher frequency of s and p among all pairs
                if cost < best_cost or (cost == best_cost):
                    if cost < best_cost:
                        best_cost = cost
                        best_edge = (i, j)
                        best_tie = 0
                    else:
                        # Among equal cost, prefer edges whose sum/product already appear more
                        sc = sum_counts.get(s, 0)
                        pc = prod_counts.get(p, 0)
                        tie_val = sc + pc
                        if tie_val > best_tie:
                            best_tie = tie_val
                            best_edge = (i, j)

            if best_edge is None:
                break

            i, j = best_edge
            s, p = props[(i, j)]
            edges.append([i, j])
            edge_set.add((i, j))
            sum_counts[s] += 1
            prod_counts[p] += 1

        return edges, max(len(sum_counts), len(prod_counts))

    def _local_search_sa(values, props, initial_edges, time_limit):
        """Local search with simulated annealing for edge swaps."""
        best_edges = [list(e) for e in initial_edges]
        edge_set = set(tuple(e) for e in best_edges)

        sum_counts = defaultdict(int)
        prod_counts = defaultdict(int)

        for e in best_edges:
            i, j = e
            s, p = props[(i, j)]
            sum_counts[s] += 1
            prod_counts[p] += 1

        def get_score():
            return max(len(sum_counts), len(prod_counts))

        current_score = get_score()
        best_score = current_score
        best_edges_copy = [list(e) for e in best_edges]

        temperature = 1.5
        cooling_rate = 0.998
        min_temperature = 0.05
        no_improve = 0
        max_no_improve = 3000

        while time.time() < time_limit and no_improve < max_no_improve:
            # Remove a random edge
            idx = random.randint(0, len(best_edges) - 1)
            e_remove = best_edges[idx]
            ir, jr = e_remove
            sr, pr = props[(ir, jr)]
            sum_counts[sr] -= 1
            prod_counts[pr] -= 1
            if sum_counts[sr] == 0:
                del sum_counts[sr]
            if prod_counts[pr] == 0:
                del prod_counts[pr]
            edge_set.remove((ir, jr))
            best_edges.pop(idx)

            # Add a random non-edge
            attempts = 0
            while True:
                i, j = random.choice(all_pairs)
                if (i, j) not in edge_set:
                    break
                attempts += 1
                if attempts > 50:
                    # Revert and skip
                    sum_counts[sr] += 1
                    prod_counts[pr] += 1
                    edge_set.add((ir, jr))
                    best_edges.append(e_remove)
                    temperature *= cooling_rate
                    if temperature < min_temperature:
                        temperature = min_temperature
                    break
            else:
                sa, pa = props[(i, j)]
                sum_counts[sa] += 1
                prod_counts[pa] += 1
                edge_set.add((i, j))
                best_edges.append([i, j])

                new_score = get_score()

                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_edges_copy = [list(e) for e in best_edges]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    delta = new_score - current_score
                    if delta <= 0 or random.random() < math.exp(-delta / max(temperature, 0.001)):
                        current_score = new_score
                        no_improve += 1
                    else:
                        # Revert
                        sum_counts[sa] -= 1
                        prod_counts[pa] -= 1
                        if sum_counts[sa] == 0:
                            del sum_counts[sa]
                        if prod_counts[pa] == 0:
                            del prod_counts[pa]
                        edge_set.remove((i, j))
                        best_edges.pop()
                        sum_counts[sr] += 1
                        prod_counts[pr] += 1
                        edge_set.add((ir, jr))
                        best_edges.append(e_remove)
                        no_improve += 1

            temperature *= cooling_rate
            if temperature < min_temperature:
                temperature = min_temperature

        return best_edges_copy, best_score

    def _collision_aware_values():
        """Generate value sets optimized for sum and product collisions."""
        candidates = []
        random.seed(12345)

        # 1. Arithmetic progressions (excellent sum collisions)
        for start in range(1, 20):
            for step in range(1, 10):
                vals = [start + i * step for i in range(n)]
                if max(vals) <= maximum and len(set(vals)) == n:
                    candidates.append(vals)

        # 2. Geometric-like progressions (good for product structure)
        for base in range(2, 6):
            for mult in range(1, 4):
                vals = [base * mult * (i + 1) for i in range(n)]
                if max(vals) <= maximum and len(set(vals)) == n:
                    candidates.append(sorted(vals))

        # 3. Smooth numbers (many shared factors -> product collisions)
        def is_smooth(num, primes):
            for p in primes:
                while num % p == 0:
                    num //= p
            return num == 1

        for primes in [[2, 3, 5], [2, 3, 5, 7], [2, 3, 5, 7, 11]]:
            smooth = [i for i in range(1, 2000) if is_smooth(i, primes)]
            if len(smooth) >= n:
                # Take consecutive chunks
                for offset in range(0, min(len(smooth) - n + 1, 30), 5):
                    candidates.append(sorted(smooth[offset:offset + n]))

        # 4. Structured random: pairs that sum to same target
        for _ in range(40):
            target = random.randint(20, 80)
            vals = set()
            for _ in range(n // 2):
                a = random.randint(1, target - 2)
                b = target - a
                if a != b and a not in vals and b not in vals and 0 < a < maximum and 0 < b < maximum:
                    vals.add(a)
                    vals.add(b)
            while len(vals) < n:
                v = random.randint(1, 100)
                if v not in vals:
                    vals.add(v)
            candidates.append(sorted(list(vals)[:n]))

        # 5. Small compact random sets
        for _ in range(30):
            vals = sorted(random.sample(range(1, 80), n))
            candidates.append(vals)

        # 6. Mixed: half AP, half random
        for _ in range(20):
            start = random.randint(1, 10)
            step = random.randint(1, 5)
            ap = [start + i * step for i in range(n // 2)]
            remaining = random.sample([v for v in range(1, 200) if v not in ap], n - n // 2)
            vals = sorted(ap + remaining)
            if len(set(vals)) == n and max(vals) <= maximum:
                candidates.append(vals)

        # Deduplicate
        unique = []
        seen = set()
        for v in candidates:
            key = tuple(v)
            if key not in seen and len(set(v)) == n and all(0 < x <= maximum for x in v):
                seen.add(key)
                unique.append(v)

        return unique

    def _evaluate_value_set(values, time_limit):
        """Evaluate a value set with greedy + SA local search."""
        props = _pair_props(values)

        # Greedy start
        g_edges, g_score = _greedy_construct(values, props, time.time() + 0.3)

        # SA local search from greedy
        sa_edges, sa_score = _local_search_sa(values, props, g_edges, time_limit)

        # Also try a second SA restart with different random seed
        if time.time() < time_limit - 2.0:
            random.seed(random.randint(0, 10**6))
            sa2_edges, sa2_score = _local_search_sa(values, props, g_edges, time_limit - 1.0)
            if sa2_score < sa_score:
                sa_edges, sa_score = sa2_edges, sa2_score

        return sa_edges, sa_score

    # Generate candidate value sets
    candidate_values = _collision_aware_values()

    # Sort candidates by a quick proxy: number of repeated sums among all pairs
    def _sum_collision_score(vals):
        props = _pair_props(vals)
        sum_counts = Counter(s for s, p in props.values())
        # Higher score = more collisions
        return sum(c * c for c in sum_counts.values())

    candidate_values.sort(key=_sum_collision_score, reverse=True)

    # Limit to top candidates to fit time budget
    max_candidates = 25
    candidate_values = candidate_values[:max_candidates]

    best_result = None
    best_overall_score = float('inf')

    time_per_candidate = (deadline - time.time() - 5.0) / max(1, len(candidate_values))

    for vals in candidate_values:
        if time.time() > deadline - 5.0:
            break

        t_limit = time.time() + time_per_candidate * 0.85
        edges, score = _evaluate_value_set(vals, t_limit)

        if score < best_overall_score:
            best_overall_score = score
            best_result = {'values': vals, 'edges': edges}

    # Fallback
    if best_result is None:
        vals = list(range(1, n + 1))
        edges = [list(e) for e in all_pairs[:m]]
        best_result = {'values': vals, 'edges': edges}

    result = {
        'values': [int(v) for v in best_result['values']],
        'edges': [[int(i), int(j)] for i, j in best_result['edges']]
    }

    return result

# EVOLVE-BLOCK-END

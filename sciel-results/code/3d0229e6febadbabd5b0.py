import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill

    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)

    def score_of(values, edges):
        """Compute max(distinct sums, distinct products) over edges."""
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def fast_score(values, edges):
        """Fast score computation using precomputed edge data."""
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def greedy_edge_select(values, target_k=None):
        """
        Greedy edge selection: at each step, pick the edge that minimizes
        the marginal increase in max(distinct_sums, distinct_prods).
        """
        sums = set()
        prods = set()
        selected = set()
        selected_list = []

        # Precompute edge contributions
        edge_data = []
        for (u, v) in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_data.append((u, v, s, p))

        for step in range(m):
            if time.time() > deadline - 5.0:
                break

            best_key = None
            best_edge = None
            best_sums = None
            best_prods = None

            for (u, v, s, p) in edge_data:
                if (u, v) in selected:
                    continue
                new_s = 1 if s not in sums else 0
                new_p = 1 if p not in prods else 0
                ns = len(sums) + new_s
                np_ = len(prods) + new_p
                marginal = max(ns, np_)
                key = (marginal, new_s, new_p)

                if best_key is None or key < best_key:
                    best_key = key
                    best_edge = (u, v)
                    best_sums = sums | ({s} if new_s else set())
                    best_prods = prods | ({p} if new_p else set())

            if best_edge is None:
                break

            selected.add(best_edge)
            selected_list.append(best_edge)
            sums = best_sums
            prods = best_prods

        return selected_list, sums, prods

    def joint_sa(values, edges, iterations=50000):
        """
        Joint simulated annealing on values and edges.
        """
        current_vals = list(values)
        current_edges = list(edges)
        current_score = score_of(current_vals, current_edges)

        best_vals = list(current_vals)
        best_edges = list(current_edges)
        best_score = current_score

        T = 5.0
        cooling = 0.9995

        for it in range(iterations):
            if time.time() > deadline - 10.0:
                break

            if it % 1000 == 0:
                T *= cooling

            # Choose perturbation: 70% value change, 30% edge swap
            if random.random() < 0.7:
                # Perturb values
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                # Swap values
                old_i, old_j = current_vals[i], current_vals[j]
                current_vals[i], current_vals[j] = current_vals[j], current_vals[i]

                new_score = score_of(current_vals, current_edges)
                delta = new_score - current_score

                if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_vals = list(current_vals)
                        best_edges = list(current_edges)
                else:
                    # Revert
                    current_vals[i], current_vals[j] = old_j, old_i
            else:
                # Perturb edges: swap one edge for another
                if len(current_edges) < 2:
                    continue
                e1_idx = random.randint(0, len(current_edges) - 1)
                e1 = tuple(sorted(current_edges[e1_idx]))

                # Pick a random different edge
                candidate = None
                for _ in range(50):
                    e2 = tuple(sorted(random.sample(all_edges, 1)[0]))
                    if e2 != e1:
                        candidate = e2
                        break
                if candidate is None:
                    continue

                new_edges = list(current_edges)
                new_edges[e1_idx] = list(candidate)

                # Check for duplicates
                edge_tuples = set()
                valid = True
                for e in new_edges:
                    t = tuple(sorted(e))
                    if t in edge_tuples:
                        valid = False
                        break
                    edge_tuples.add(t)
                if not valid:
                    continue

                new_score = score_of(current_vals, new_edges)
                delta = new_score - current_score

                if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    current_edges = new_edges
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_vals = list(current_vals)
                        best_edges = list(current_edges)

        return best_vals, best_edges, best_score

    def polish(values, edges, max_rounds=50):
        """
        Final polish pass: try local swaps to improve score.
        """
        vals = list(values)
        eds = list(edges)
        score = score_of(vals, eds)

        for round_num in range(max_rounds):
            if time.time() > deadline - 3.0:
                break

            improved = False

            # Try edge swaps: replace one edge with another
            for i in range(len(eds)):
                if time.time() > deadline - 3.0:
                    break
                e1 = tuple(sorted(eds[i]))
                for e2 in all_edges:
                    e2t = tuple(sorted(e2))
                    if e2t == e1:
                        continue
                    # Check e2 not already in edges
                    if e2t in set(tuple(sorted(e)) for e in eds):
                        continue

                    new_edges = list(eds)
                    new_edges[i] = list(e2)
                    new_score = score_of(vals, new_edges)
                    if new_score < score:
                        eds = new_edges
                        score = new_score
                        improved = True
                        break
                if improved:
                    break

            # Try value swaps
            if not improved:
                for i in range(n):
                    if time.time() > deadline - 3.0:
                        break
                    for j in range(i + 1, n):
                        if time.time() > deadline - 3.0:
                            break
                        old_i, old_j = vals[i], vals[j]
                        vals[i], vals[j] = vals[j], vals[i]
                        new_score = score_of(vals, eds)
                        if new_score < score:
                            score = new_score
                            improved = True
                        else:
                            vals[i], vals[j] = old_j, old_i
                        if improved:
                            break
                    if improved:
                        break

            if not improved:
                break

        return vals, eds, score

    # Candidate value sets
    value_candidates = [
        list(range(1, 15)),
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 20],
    ]

    best_score = float('inf')
    best_values = None
    best_edges = None

    # Phase 1: Greedy edge selection for each candidate
    for vals in value_candidates:
        if time.time() > deadline - 25.0:
            break

        if len(set(vals)) != n:
            continue
        if any(v < 1 or v > 10000 for v in vals):
            continue

        edges, sums, prods = greedy_edge_select(vals)

        if len(edges) < m:
            continue

        sc = max(len(sums), len(prods))

        if sc < best_score:
            best_score = sc
            best_values = list(vals)
            best_edges = list(edges)

    # Phase 2: Joint SA from best greedy result
    if best_values is not None and len(best_edges) >= m:
        if time.time() < deadline - 15.0:
            sa_vals, sa_edges, sa_score = joint_sa(best_values, best_edges[:m], iterations=30000)
            if sa_score < best_score:
                best_score = sa_score
                best_values = sa_vals
                best_edges = sa_edges

    # Phase 3: Polish
    if best_values is not None and len(best_edges) >= m:
        if time.time() < deadline - 5.0:
            p_vals, p_edges, p_score = polish(best_values, best_edges[:m], max_rounds=30)
            if p_score < best_score:
                best_score = p_score
                best_values = p_vals
                best_edges = p_edges

    # Ensure we have exactly m edges
    if best_edges is None or len(best_edges) < m:
        # Fallback
        vals = list(range(1, 15))
        edges = list(all_edges[:m])
        best_score = score_of(vals, edges)
        best_values = vals
        best_edges = edges

    if len(best_edges) > m:
        best_edges = best_edges[:m]

    # Final validation
    final_values = [int(v) for v in best_values]
    final_edges = [[int(u), int(v)] for u, v in best_edges]

    assert len(final_values) == n
    assert len(set(final_values)) == n
    assert all(isinstance(v, int) and 1 <= v <= 10000 for v in final_values)
    assert len(final_edges) == m
    assert all(len(e) == 2 for e in final_edges)
    assert all(e[0] != e[1] for e in final_edges)
    edge_set = set(tuple(sorted(e)) for e in final_edges)
    assert len(edge_set) == m

    return {
        'values': final_values,
        'edges': final_edges
    }

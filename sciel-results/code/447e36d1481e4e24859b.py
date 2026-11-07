import time
import random
import itertools
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill

    n = 14
    m = 35
    max_val = 10000

    all_pairs = list(itertools.combinations(range(n), 2))
    num_pairs = len(all_pairs)  # 91

    def solve_with_ilp(vals):
        """Use MILP to find 35 edges minimizing max(distinct_sums, distinct_prods)."""
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np

        # Precompute sums and products for all pairs
        pair_sums = np.array([vals[i] + vals[j] for i, j in all_pairs])
        pair_prods = np.array([vals[i] * vals[j] for i, j in all_pairs])

        # Get unique sums and products
        unique_sums = np.unique(pair_sums)
        unique_prods = np.unique(pair_prods)
        n_sums = len(unique_sums)
        n_prods = len(unique_prods)

        # Map each pair to its sum/prod index
        sum_idx = np.searchsorted(unique_sums, pair_sums)
        prod_idx = np.searchsorted(unique_prods, pair_prods)

        # Decision variables:
        # x[0:num_pairs] = edge selection (binary)
        # x[num_pairs:num_pairs+n_sums] = sum used (binary)
        # x[num_pairs+n_sums:num_pairs+n_sums+n_prods] = prod used (binary)
        # x[num_pairs+n_sums+n_prods] = M (continuous, the max)

        total_vars = num_pairs + n_sums + n_prods + 1

        # Objective: minimize M
        c = np.zeros(total_vars)
        c[-1] = 1.0

        # Constraints:
        # 1. Exactly m edges: sum(x[0:num_pairs]) == m
        # 2. For each sum s: sum of x_e for edges with sum s <= n_sums * y_s
        #    (if any edge with sum s is selected, y_s must be 1)
        #    Actually: sum_e_in_S(s) x_e <= |S(s)| * y_s
        # 3. Similarly for products
        # 4. M >= y_s for all s
        # 5. M >= y_p for all p

        A_rows = []
        b_lower = []
        b_upper = []

        # Constraint 1: exactly m edges
        row = np.zeros(num_pairs)
        row[:] = 1
        A_rows.append(row)
        b_lower.append(m)
        b_upper.append(m)

        # Constraint 2: For each sum, edges with that sum <= count * y_s
        for s_idx in range(n_sums):
            mask = (sum_idx == s_idx)
            row = np.zeros(num_pairs)
            row[mask] = 1
            A_rows.append(row)
            b_lower.append(0)
            b_upper.append(mask.sum())  # |S(s)| * 1, but we need <= |S(s)|*y_s
            # This linearization isn't quite right. Let me use a different approach.

        # Better approach: Use the constraint that y_s >= x_e for all e with sum s
        # And minimize sum of y_s (or use M as upper bound)

        # Let me reformulate with big-M or direct constraints:
        # y_s >= x_e for all e where sum_e = s
        # y_p >= x_e for all e where prod_e = p
        # M >= y_s for all s
        # M >= y_p for all p
        # sum x_e = m
        # x_e, y_s, y_p in {0,1}, M >= 0

        # This gives:
        # For each pair (e, s) where sum_e = s: y_s - x_e >= 0
        # For each pair (e, p) where prod_e = p: y_p - x_e >= 0
        # For each s: M - y_s >= 0
        # For each p: M - y_p >= 0
        # sum x_e = m

        A_rows = []
        b_lower = []
        b_upper = []

        # Constraint: sum x_e = m
        row = np.zeros(num_pairs)
        row[:] = 1
        A_rows.append(row)
        b_lower.append(m)
        b_upper.append(m)

        # For each edge e and its sum s: y_s >= x_e  =>  -x_e + y_s >= 0
        for e_idx in range(num_pairs):
            s = sum_idx[e_idx]
            row = np.zeros(total_vars)
            row[e_idx] = -1
            row[num_pairs + s] = 1
            A_rows.append(row)
            b_lower.append(0)
            b_upper.append(np.inf)

        # For each edge e and its prod p: y_p >= x_e  =>  -x_e + y_p >= 0
        for e_idx in range(num_pairs):
            p = prod_idx[e_idx]
            row = np.zeros(total_vars)
            row[e_idx] = -1
            row[num_pairs + n_sums + p] = 1
            A_rows.append(row)
            b_lower.append(0)
            b_upper.append(np.inf)

        # For each s: M >= y_s  =>  -y_s + M >= 0
        for s in range(n_sums):
            row = np.zeros(total_vars)
            row[num_pairs + s] = -1
            row[-1] = 1
            A_rows.append(row)
            b_lower.append(0)
            b_upper.append(np.inf)

        # For each p: M >= y_p  =>  -y_p + M >= 0
        for p in range(n_prods):
            row = np.zeros(total_vars)
            row[num_pairs + n_sums + p] = -1
            row[-1] = 1
            A_rows.append(row)
            b_lower.append(0)
            b_upper.append(np.inf)

        A = np.array(A_rows)
        b_lower_arr = np.array(b_lower)
        b_upper_arr = np.array(b_upper)

        # Integrality: x_e and y_s, y_p are binary; M is continuous
        integrality = np.zeros(total_vars)
        integrality[:num_pairs] = 1
        integrality[num_pairs:num_pairs+n_sums] = 1
        integrality[num_pairs+n_sums:num_pairs+n_sums+n_prods] = 1
        # M (last var) is continuous: integrality[-1] = 0

        # Bounds
        lb = np.zeros(total_vars)
        ub = np.ones(total_vars)
        ub[-1] = 91  # M can be at most 91 (max possible distinct values)

        constraints = LinearConstraint(A, b_lower_arr, b_upper_arr)
        bounds = Bounds(lb, ub)

        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

        if not result.success:
            return None, None

        x = result.x
        edge_vars = x[:num_pairs]
        selected_edges = []
        for e_idx in range(num_pairs):
            if edge_vars[e_idx] > 0.5:
                selected_edges.append(list(all_pairs[e_idx]))

        # Compute actual score
        sums = set()
        prods = set()
        for i, j in selected_edges:
            sums.add(vals[i] + vals[j])
            prods.add(vals[i] * vals[j])
        score = max(len(sums), len(prods))

        return selected_edges, score

    def greedy_edges(vals):
        """Fast greedy edge selection as fallback."""
        pair_data = []
        for i, j in all_pairs:
            s = vals[i] + vals[j]
            p = vals[i] * vals[j]
            pair_data.append((s, p, i, j))

        sum_freq = Counter(s for s, p, i, j in pair_data)
        prod_freq = Counter(p for s, p, i, j in pair_data)

        selected = []
        seen_sums = set()
        seen_prods = set()
        available = set(range(num_pairs))

        for _ in range(m):
            best_idx = -1
            best_cost = float('inf')
            best_tie = -1

            for idx in list(available):
                s, p, i, j = pair_data[idx]
                new_sums = len(seen_sums) + (1 if s not in seen_sums else 0)
                new_prods = len(seen_prods) + (1 if p not in seen_prods else 0)
                cost = max(new_sums, new_prods)
                tie = sum_freq[s] + prod_freq[p]

                if cost < best_cost or (cost == best_cost and tie > best_tie):
                    best_cost = cost
                    best_idx = idx
                    best_tie = tie

            if best_idx == -1:
                break
            s, p, i, j = pair_data[best_idx]
            selected.append([i, j])
            seen_sums.add(s)
            seen_prods.add(p)
            available.remove(best_idx)

        sums = set()
        prods = set()
        for i, j in selected:
            sums.add(vals[i] + vals[j])
            prods.add(vals[i] * vals[j])
        score = max(len(sums), len(prods))
        return selected, score

    def local_search_edges(vals, edges):
        """Improve edges via swaps."""
        current_edges = [e[:] for e in edges]

        def calc_score(edge_list):
            sums = set()
            prods = set()
            for i, j in edge_list:
                sums.add(vals[i] + vals[j])
                prods.add(vals[i] * vals[j])
            return max(len(sums), len(prods))

        current_score = calc_score(current_edges)
        selected_keys = set(frozenset(e) for e in current_edges)
        non_selected = [list(p) for p in all_pairs if frozenset(p) not in selected_keys]

        improved = True
        max_iters = 100
        iters = 0

        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            random.shuffle(non_selected)

            for new_edge in non_selected:
                new_key = frozenset(new_edge)
                if new_key in selected_keys:
                    continue

                for k in range(m):
                    old_key = frozenset(current_edges[k])
                    if old_key == new_key:
                        continue

                    temp = current_edges[:]
                    temp[k] = list(new_edge)

                    # Check no duplicates
                    if len(set(frozenset(e) for e in temp)) != m:
                        continue

                    new_score = calc_score(temp)
                    if new_score < current_score:
                        current_edges = temp
                        current_score = new_score
                        selected_keys = set(frozenset(e) for e in current_edges)
                        non_selected = [list(p) for p in all_pairs if frozenset(p) not in selected_keys]
                        improved = True
                        break
                if improved:
                    break
            iters += 1

        return current_edges, current_score

    best_score = float('inf')
    best_values = None
    best_edges = None

    # Try multiple value configurations
    value_sets = []

    # Set 1: 1-14
    value_sets.append(list(range(1, 15)))
    # Set 2: 1-13, 15
    value_sets.append([1,2,3,4,5,6,7,8,9,10,11,12,13,15])
    # Set 3: 1-12, 14, 16
    value_sets.append([1,2,3,4,5,6,7,8,9,10,11,12,14,16])
    # Set 4: Random small
    for _ in range(5):
        random.shuffle(list(range(1, 30)))
        v = list(range(1, 30))[:n]
        v.sort()
        value_sets.append(v)

    for vals in value_sets:
        if time.time() > deadline - 5:
            break

        # First try ILP (slower but potentially better)
        if time.time() < deadline - 30:
            try:
                edges, score = solve_with_ilp(vals)
                if edges is not None and len(edges) == m:
                    if score < best_score:
                        best_score = score
                        best_values = vals[:]
                        best_edges = [e[:] for e in edges]
            except Exception:
                pass

        # Then greedy + local search (faster)
        edges, score = greedy_edges(vals)
        if time.time() < deadline - 10:
            edges, score = local_search_edges(vals, edges)

        if score < best_score:
            best_score = score
            best_values = vals[:]
            best_edges = [e[:] for e in edges]

        # Try modifying values via local search
        if time.time() < deadline - 15:
            current_vals = vals[:]
            current_edges = best_edges[:] if best_values == vals else edges
            current_score = score

            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                idx = random.randint(0, n - 1)
                old_val = current_vals[idx]
                new_val = random.randint(1, 50)
                if new_val == old_val or new_val in current_vals:
                    continue

                temp_vals = current_vals[:]
                temp_vals[idx] = new_val
                temp_vals.sort()

                te, ts = greedy_edges(temp_vals)
                if time.time() < deadline - 10:
                    te, ts = local_search_edges(temp_vals, te)

                if ts < current_score:
                    current_vals = temp_vals
                    current_edges = te
                    current_score = ts
                    if current_score < best_score:
                        best_score = current_score
                        best_values = current_vals[:]
                        best_edges = [e[:] for e in current_edges]

    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges, _ = greedy_edges(best_values)

    # Final type conversion
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]

    return {'values': best_values, 'edges': best_edges}

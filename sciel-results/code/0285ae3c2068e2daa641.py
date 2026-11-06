import numpy as np
import random
import time
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = 128
    target_lo = 128
    target_hi = 256
    max_sum = 2 * n
    N = n + 1  # 129 possible values

    # Precompute pair contributions
    # pairs_for_sum[s] = list of (i, j) with i <= j, i+j == s, i,j in [0, n]
    pairs_for_sum = [[] for _ in range(max_sum + 1)]
    for i in range(N):
        for j in range(i, N):
            s = i + j
            if s <= max_sum:
                pairs_for_sum[s].append((i, j))

    # Precompute incidence vectors for each target sum
    # For sum s, we need a binary vector v_s of length N where v_s[k] = 1 if k is in some pair (i,j) with i+j=s
    # Actually, for the ILP, we need to express count_s = sum of x_i*x_j over pairs
    # This is quadratic. But we can linearize: for each pair (i,j), i!=j, introduce y_ij = x_i AND x_j
    # For i==j, y_ii = x_i
    # count_s = sum_{(i,j) in pairs_for_sum[s]} y_ij
    # Constraints: y_ij <= x_i, y_ij <= x_j, y_ij >= x_i + x_j - 1 (for i != j)
    # y_ii = x_i (for i == j)

    # This creates a very large MILP. Let's think of a better approach.
    # Actually, since N=129, the number of variables is manageable if we're careful.
    # x_i for i in 0..128 (129 vars)
    # y_ij for i < j in 0..128 (129*128/2 = 8256 vars)
    # y_ii for i in 0..128 (129 vars)
    # Total: 129 + 8256 + 129 = 8514 vars

    # But wait, the constraint count_s >= 1 for s in [128, 256] and count_s <= M for all s
    # count_s = sum_{(i,j) in pairs, i<j} y_ij + sum_{i: 2i=s} y_ii
    # Since y_ii = x_i, this is count_s = sum_{(i,j) in pairs, i<j, i+j=s} y_ij + sum_{i: 2i=s} x_i

    # The MILP will be:
    # Minimize M
    # Subject to:
    #   For each s in [128, 256]: sum_{(i,j) in pairs, i<j, i+j=s} y_ij + sum_{i: 2i=s} x_i >= 1
    #   For each s in [128, 256]: sum_{(i,j) in pairs, i<j, i+j=s} y_ij + sum_{i: 2i=s} x_i <= M
    #   For each i < j: y_ij <= x_i, y_ij <= x_j, y_ij >= x_i + x_j - 1
    #   For each i: y_ii = x_i (just substitute, no extra var needed)
    #   x_i in {0,1}, y_ij in {0,1}

    # Actually, we can eliminate y_ii by substitution. So variables are x_0..x_128 and y_ij for i<j.
    # That's 129 + 8256 = 8385 variables.

    # The number of constraints:
    # 256 * 2 = 512 for coverage and peak
    # 8256 * 3 = 24768 for the AND constraints
    # Total ~ 25280 constraints

    # This is a large but potentially solvable MILP. Let's try with scipy.optimize.milp.

    def solve_ilp(peak_bound):
        """Solve the MILP with peak <= peak_bound. Returns solution or None."""
        if time.time() > deadline - 5.0:
            return None

        # Variables: x_0..x_128 (indices 0..128), then y_ij for 0<=i<j<=128 (indices 129..8384)
        n_x = N
        n_y = N * (N - 1) // 2
        total_vars = n_x + n_y

        # Objective: minimize 0 (feasibility check) or we can minimize sum x_i to get a smaller set
        c = np.zeros(total_vars)
        # Actually, let's minimize the sum of x_i to prefer smaller sets
        c[:n_x] = 1.0

        # Constraints
        # We'll build the constraint matrix
        # Type 0: coverage (>= 1)
        # Type 1: peak (<= peak_bound)
        # Type 2: y_ij <= x_i
        # Type 3: y_ij <= x_j
        # Type 4: y_ij >= x_i + x_j - 1

        # Let's build sparse matrix
        rows = []
        cols = []
        vals = []
        lb = []
        ub = []

        # Helper to add a row
        def add_row(row_idx, col_idx, val, lower, upper):
            rows.append(row_idx)
            cols.append(col_idx)
            vals.append(val)
            lb.append(lower)
            ub.append(upper)

        row_count = 0

        # Coverage constraints: for s in [128, 256], count_s >= 1
        for s in range(target_lo, target_hi + 1):
            r = row_count
            row_count += 1
            for (i, j) in pairs_for_sum[s]:
                if i < j:
                    y_idx = n_x + i * N + j - (j + 1) // 2  # Need correct indexing
                    # Let's use a cleaner indexing
                    # y_ij index: n_x + i*(N-1) - (i+1)*i//2 + (j - i - 1)
                    # Simpler: precompute a mapping
                    pass
            # Let me redo this with a precomputed mapping

        # Let me precompute the y index mapping
        y_index = {}
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                y_index[(i, j)] = n_x + idx
                idx += 1
        assert idx == n_y

        # Now build constraints properly
        rows = []
        cols = []
        vals = []
        lb = []
        ub = []

        row_count = 0

        # Coverage: count_s >= 1 for s in [128, 256]
        for s in range(target_lo, target_hi + 1):
            r = row_count
            row_count += 1
            for (i, j) in pairs_for_sum[s]:
                if i < j:
                    y_idx = y_index[(i, j)]
                    rows.append(r)
                    cols.append(y_idx)
                    vals.append(1.0)
                else:  # i == j
                    rows.append(r)
                    cols.append(i)
                    vals.append(1.0)
            lb.append(1.0)
            ub.append(np.inf)

        # Peak: count_s <= peak_bound for s in [128, 256]
        for s in range(target_lo, target_hi + 1):
            r = row_count
            row_count += 1
            for (i, j) in pairs_for_sum[s]:
                if i < j:
                    y_idx = y_index[(i, j)]
                    rows.append(r)
                    cols.append(y_idx)
                    vals.append(1.0)
                else:  # i == j
                    rows.append(r)
                    cols.append(i)
                    vals.append(1.0)
            lb.append(0.0)
            ub.append(float(peak_bound))

        # AND constraints: for each i < j
        for i in range(N):
            for j in range(i + 1, N):
                y_idx = y_index[(i, j)]
                # y_ij <= x_i
                r = row_count
                row_count += 1
                rows.append(r)
                cols.append(y_idx)
                vals.append(1.0)
                rows.append(r)
                cols.append(i)
                vals.append(-1.0)
                lb.append(0.0)
                ub.append(0.0)

                # y_ij <= x_j
                r = row_count
                row_count += 1
                rows.append(r)
                cols.append(y_idx)
                vals.append(1.0)
                rows.append(r)
                cols.append(j)
                vals.append(-1.0)
                lb.append(0.0)
                ub.append(0.0)

                # y_ij >= x_i + x_j - 1
                r = row_count
                row_count += 1
                rows.append(r)
                cols.append(y_idx)
                vals.append(1.0)
                rows.append(r)
                cols.append(i)
                vals.append(-1.0)
                rows.append(r)
                cols.append(j)
                vals.append(-1.0)
                lb.append(-1.0)
                ub.append(np.inf)

        # Build sparse matrix
        A = csr_matrix((vals, (rows, cols)), shape=(row_count, total_vars))

        # Variable bounds
        var_lb = np.zeros(total_vars)
        var_ub = np.ones(total_vars)

        # Integrality: x vars are binary, y vars are binary
        integrality = np.ones(total_vars)

        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(A, lb, ub),
                integrality=integrality,
                bounds=Bounds(var_lb, var_ub),
                options={'time_limit': 30, 'presolve': True}
            )
            if res.status == 0 and res.x is not None:
                x_sol = res.x[:n_x]
                A_sol = set(int(i) for i in range(N) if x_sol[i] > 0.5)
                return A_sol
            return None
        except Exception:
            return None

    # Try ILP with decreasing peak bounds
    best_A = None
    best_peak = float('inf')

    for peak in [3, 4, 5, 6, 7]:
        if time.time() > deadline - 10.0:
            break
        A_sol = solve_ilp(peak)
        if A_sol is not None and len(A_sol) > 0:
            # Verify
            counts = [0] * (max_sum + 1)
            A_list = sorted(A_sol)
            for i in range(len(A_list)):
                for j in range(i, len(A_list)):
                    s = A_list[i] + A_list[j]
                    if s <= max_sum:
                        counts[s] += 1
            uncovered = sum(1 for s in range(target_lo, target_hi + 1) if counts[s] == 0)
            actual_peak = max(counts[target_lo:target_hi + 1])
            if uncovered == 0 and actual_peak <= peak:
                best_A = A_sol
                best_peak = actual_peak
                break

    if best_A is not None:
        return list(best_A)

    # Fallback: use the original heuristic approach if ILP fails
    # Quick greedy + local search
    def get_score(counts):
        target_counts = counts[target_lo:target_hi + 1]
        uncovered = sum(1 for c in target_counts if c == 0)
        peak = max(target_counts) if target_counts else 0
        return (uncovered, peak)

    def init_counts(A):
        counts = [0] * (max_sum + 1)
        A_list = sorted(A)
        for i in range(len(A_list)):
            for j in range(i, len(A_list)):
                s = A_list[i] + A_list[j]
                if s <= max_sum:
                    counts[s] += 1
        return counts

    # Greedy construction
    A = set()
    counts = [0] * (max_sum + 1)

    # Start with a reasonable base
    base_size = 40
    A = set(random.sample(range(N), base_size))
    counts = init_counts(A)

    # Local search
    for _ in range(5000):
        if time.time() > deadline - 2.0:
            break
        curr_score = get_score(counts)
        if curr_score[0] == 0 and curr_score[1] <= 5:
            break

        # Try a random swap
        if A:
            x = random.choice(list(A))
            candidates = [i for i in range(N) if i not in A]
            if candidates:
                y = random.choice(candidates)
                # Swap
                A.remove(x)
                for z in A:
                    s = x + z
                    if s <= max_sum:
                        counts[s] -= 1
                if 2 * x <= max_sum:
                    counts[2 * x] -= 1

                A.add(y)
                for z in A:
                    s = y + z
                    if s <= max_sum:
                        counts[s] += 1

                new_score = get_score(counts)
                if new_score > curr_score:
                    # Revert
                    A.remove(y)
                    for z in A:
                        s = y + z
                        if s <= max_sum:
                            counts[s] -= 1

                    A.add(x)
                    for z in A:
                        s = x + z
                        if s <= max_sum:
                            counts[s] += 1
                    if 2 * x <= max_sum:
                        counts[2 * x] += 1

    # Final check
    counts = init_counts(A)
    if get_score(counts)[0] == 0:
        return list(A)

    # Last resort: return a known valid set (e.g., all even numbers or a dense set)
    # A = set(range(0, N, 2)) gives sums that are even, covering [128,256] even numbers
    # But we need all numbers. Let's use a dense set.
    A = set(range(64, N))  # [64, 128], sums in [128, 256]
    counts = init_counts(A)
    if get_score(counts)[0] == 0:
        return list(A)

    return list(range(N))

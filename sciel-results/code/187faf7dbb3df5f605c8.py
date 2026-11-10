import math
import time
import random
import heapq
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85.0  # 90s wall-clock, leave 5s margin

    N = 256
    max_sum = 2 * N
    perfect_squares = set()
    for k in range(1, int(math.isqrt(max_sum)) + 2):
        sq = k * k
        if sq <= max_sum:
            perfect_squares.add(sq)

    # Self-loop check: 2a is a perfect square
    self_loop_set = set()
    for m in range(1, 13):
        a = 2 * m * m
        if 1 <= a <= N:
            self_loop_set.add(a)

    # Remaining vertices
    remaining = [a for a in range(1, N + 1) if a not in self_loop_set]
    n_rem = len(remaining)

    # Build adjacency: adj[i] contains j if remaining[i] + remaining[j] is a perfect square
    adj = [[] for _ in range(n_rem)]
    for i in range(n_rem):
        a = remaining[i]
        for j in range(i + 1, n_rem):
            b = remaining[j]
            if (a + b) in perfect_squares:
                adj[i].append(j)
                adj[j].append(i)

    # Precompute adjacency masks for fast operations
    adj_mask = [0] * n_rem
    for i in range(n_rem):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask

    degrees = [len(adj[i]) for i in range(n_rem)]

    # Edge list for LP
    edge_list = []
    for i in range(n_rem):
        for j in adj[i]:
            if i < j:
                edge_list.append((i, j))
    n_edges = len(edge_list)

    best_indices = []
    best_size = 0

    def greedy_min_degree(rs):
        available = (1 << n_rem) - 1
        result_indices = []

        while available:
            min_deg = n_rem + 1
            candidates = []
            avail_list = []
            temp = available
            while temp:
                lsb = temp & (-temp)
                i = lsb.bit_length() - 1
                avail_list.append(i)
                temp ^= lsb

            for i in avail_list:
                deg = bin(adj_mask[i] & available).count('1')
                if deg < min_deg:
                    min_deg = deg
                    candidates = [i]
                elif deg == min_deg:
                    candidates.append(i)

            v = rs.choice(candidates)
            result_indices.append(v)
            available &= ~(1 << v)
            available &= ~adj_mask[v]

        return result_indices

    def local_improvement_fast(indices, rs, max_iters=3000):
        if not indices:
            return indices

        selected_mask = 0
        for i in indices:
            selected_mask |= (1 << i)

        improved = True
        iters = 0
        all_mask = (1 << n_rem) - 1

        while improved and iters < max_iters and time.time() < DEADLINE:
            improved = False
            iters += 1

            non_selected = all_mask & ~selected_mask
            temp = non_selected
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb

                conflicts = adj_mask[v] & selected_mask
                if conflicts == 0:
                    selected_mask |= (1 << v)
                    improved = True
                else:
                    conflict_count = bin(conflicts).count('1')
                    if conflict_count == 1:
                        selected_mask = (selected_mask & ~conflicts) | (1 << v)
                        improved = True

        result = []
        temp = selected_mask
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            result.append(i)
            temp ^= lsb

        return result

    # Run initial heuristics to get a good starting solution
    for seed in range(100):
        if time.time() > DEADLINE - 5:
            break
        rs = random.Random(seed)
        s = greedy_min_degree(rs)
        s = local_improvement_fast(s, rs, max_iters=2000)
        if len(s) > best_size:
            best_indices = s
            best_size = len(s)

    # Final polish: try to add any remaining vertices
    if time.time() < DEADLINE - 3:
        final_mask = 0
        for i in best_indices:
            final_mask |= (1 << i)
        changed = True
        while changed and time.time() < DEADLINE - 3:
            changed = False
            for i in range(n_rem):
                if final_mask & (1 << i):
                    continue
                if adj_mask[i] & final_mask == 0:
                    final_mask |= (1 << i)
                    changed = True
        new_indices = []
        temp = final_mask
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            new_indices.append(i)
            temp ^= lsb
        if len(new_indices) > best_size:
            best_indices = new_indices
            best_size = len(new_indices)

    # Now use branch-and-bound with LP relaxation to try to improve
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix

        if n_edges == 0:
            best_indices = list(range(n_rem))
            best_size = n_rem
        else:
            # Build sparse constraint matrix A: each row is an edge (x_i + x_j <= 1)
            rows_data = []
            cols_data = []
            vals_data = []
            for ei, (i, j) in enumerate(edge_list):
                rows_data.extend([ei, ei])
                cols_data.extend([i, j])
                vals_data.extend([1, 1])

            A_sp = coo_matrix((vals_data, (rows_data, cols_data)), shape=(n_edges, n_rem)).tocsr()
            c_obj = -np.ones(n_rem)  # minimize -sum(x) = maximize sum(x)
            b_ub = np.ones(n_edges)

            # Sort vertices by degree descending for branching priority
            branch_order = sorted(range(n_rem), key=lambda i: -degrees[i])

            best_bb_size = best_size
            best_bb_indices = list(best_indices)

            # LP relaxation solve function
            def solve_lp(fixed_one_mask, fixed_zero_mask):
                """Solve LP relaxation with some variables fixed.
                Returns (objective_value, solution_vector) or (None, None) if infeasible."""
                # Determine free variables
                free_vars = []
                for i in range(n_rem):
                    bit = (1 << i)
                    if not (fixed_one_mask & bit) and not (fixed_zero_mask & bit):
                        free_vars.append(i)

                n_free = len(free_vars)
                if n_free == 0:
                    return (0.0, np.zeros(n_rem))

                free_idx = {v: k for k, v in enumerate(free_vars)}

                # Objective for free vars
                c_free = -np.ones(n_free)

                # Constraints: only edges where at least one endpoint is free
                A_rows = []
                b_rhs = []
                for ei, (i, j) in enumerate(edge_list):
                    i_fixed_one = bool(fixed_one_mask & (1 << i))
                    j_fixed_one = bool(fixed_zero_mask & (1 << j))
                    i_fixed_zero = bool(fixed_zero_mask & (1 << i))
                    j_fixed_zero = bool(fixed_zero_mask & (1 << j))

                    if i_fixed_one and j_fixed_one:
                        return (None, None)  # Infeasible

                    if i_fixed_one:
                        if not j_fixed_zero and not j_fixed_one:
                            # x_j <= 0
                            if j in free_idx:
                                row = np.zeros(n_free)
                                row[free_idx[j]] = 1
                                A_rows.append(row)
                                b_rhs.append(0.0)
                    elif j_fixed_one:
                        if not i_fixed_zero and not i_fixed_one:
                            # x_i <= 0
                            if i in free_idx:
                                row = np.zeros(n_free)
                                row[free_idx[i]] = 1
                                A_rows.append(row)
                                b_rhs.append(0.0)
                    else:
                        # Neither fixed to 1
                        if i_fixed_zero and j_fixed_zero:
                            pass  # 0+0 <= 1, always satisfied
                        elif i_fixed_zero:
                            # x_j <= 1, redundant
                            pass
                        elif j_fixed_zero:
                            # x_i <= 1, redundant
                            pass
                        else:
                            # Both free: x_i + x_j <= 1
                            row = np.zeros(n_free)
                            row[free_idx[i]] = 1
                            row[free_idx[j]] = 1
                            A_rows.append(row)
                            b_rhs.append(1.0)

                if A_rows:
                    A_mat = np.array(A_rows)
                    b_vec = np.array(b_rhs)
                else:
                    A_mat = None
                    b_vec = None

                # Bounds: 0 <= x_i <= 1 for all free vars
                bounds_lp = [(0, 1)] * n_free

                if A_mat is not None:
                    res = linprog(c_free, A_ub=A_mat, b_ub=b_vec, bounds=bounds_lp, method='highs')
                else:
                    # No constraints, all x_i = 1
                    return (float(n_free), np.ones(n_free))

                if res.status == 0:
                    obj_val = -res.fun  # This is sum of x_i for free vars
                    # Add the fixed-to-one count
                    total_obj = obj_val + bin(fixed_one_mask).count('1')
                    # Reconstruct full solution
                    full_sol = np.zeros(n_rem)
                    for i in range(n_rem):
                        if fixed_one_mask & (1 << i):
                            full_sol[i] = 1.0
                        elif not (fixed_zero_mask & (1 << i)):
                            full_sol[i] = res.x[free_idx[i]]
                    return (total_obj, full_sol)
                else:
                    return (None, None)

            # Branch and bound
            # Stack of (fixed_one_mask, fixed_zero_mask, upper_bound)
            # Use a priority queue approach: always explore the node with highest upper bound

            # Initial LP solve
            init_obj, init_sol = solve_lp(0, 0)
            if init_obj is None:
                pass  # Should not happen
            else:
                # Initial upper bound
                ub = init_obj

                # Get initial integer solution from LP (round down, check feasibility)
                init_int = [i for i in range(n_rem) if init_sol[i] > 0.5]
                # Validate
                valid_init = True
                init_set = set(init_int)
                for i in init_int:
                    for j in adj[i]:
                        if j in init_set:
                            valid_init = False
                            break
                    if not valid_init:
                        break
                if valid_init:
                    if len(init_int) > best_bb_size:
                        best_bb_size = len(init_int)
                        best_bb_indices = list(init_int)

                # B&B stack: (negative_upper_bound, fixed_one, fixed_zero)
                # We use negative so min-heap gives max upper bound
                stack = []
                # Root node is already solved; we need to branch on it
                # Find the most fractional variable in the LP solution
                # Prioritize by degree

                # For branching, find variables with 0 < x < 1
                frac_vars = []
                for i in range(n_rem):
                    if 0.01 < init_sol[i] < 0.99:
                        frac_vars.append(i)

                if frac_vars:
                    # Sort by degree descending (high conflict density first)
                    frac_vars.sort(key=lambda i: -degrees[i])
                    branch_var = frac_vars[0]
                    # Branch: x_i = 1 and x_i = 0
                    # x_i = 1: fix i to one, fix all neighbors to zero
                    mask_one = (1 << branch_var)
                    for nb in adj[branch_var]:
                        mask_one |= (1 << nb)  # These will be forced to zero
                    # Actually, we need to track them separately
                    # Let's use a different representation

                # Better approach: use (fixed_one, forced_zero) where forced_zero includes neighbors
                # Let me restructure

                # Actually, let me use a simpler but effective approach:
                # Iterative deepening with LP bounds

                # Let's use a recursive B&B with time limit
                nodes_explored = 0
                MAX_NODES = 50000  # Limit to avoid timeout

                def branch_and_bound(fixed_one_mask, fixed_zero_mask, remaining_ub):
                    nonlocal best_bb_size, best_bb_indices, nodes_explored

                    if time.time() > DEADLINE - 10:
                        return

                    nodes_explored += 1
                    if nodes_explored > MAX_NODES:
                        return

                    # Check if upper bound is already <= best
                    if remaining_ub <= best_bb_size:
                        return

                    # Solve LP
                    obj, sol = solve_lp(fixed_one_mask, fixed_zero_mask)
                    if obj is None:
                        return  # Infeasible

                    if obj <= best_bb_size:
                        return

                    # Check if solution is integer
                    is_integer = True
                    int_sol = []
                    for i in range(n_rem):
                        if sol[i] > 0.5:
                            int_sol.append(i)
                        elif sol[i] > 0.01:
                            is_integer = False

                    if is_integer:
                        # Validate
                        valid = True
                        int_set = set(int_sol)
                        for i in int_sol:
                            for j in adj[i]:
                                if j in int_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid and len(int_sol) > best_bb_size:
                            best_bb_size = len(int_sol)
                            best_bb_indices = list(int_sol)
                        return

                    # Find branching variable: most fractional, highest degree
                    frac_candidates = []
                    for i in range(n_rem):
                        if fixed_one_mask & (1 << i) or fixed_zero_mask & (1 << i):
                            continue
                        if 0.01 < sol[i] < 0.99:
                            frac_candidates.append((i, sol[i]))

                    if not frac_candidates:
                        # All free vars are at bounds; this shouldn't happen if not integer
                        # Try rounding
                        rounded = [i for i in range(n_rem) if sol[i] > 0.5]
                        valid = True
                        r_set = set(rounded)
                        for i in rounded:
                            for j in adj[i]:
                                if j in r_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid and len(rounded) > best_bb_size:
                            best_bb_size = len(rounded)
                            best_bb_indices = list(rounded)
                        return

                    # Sort by degree descending, then by fractionality
                    frac_candidates.sort(key=lambda x: (-degrees[x[0]], abs(x[1] - 0.5)))
                    branch_i = frac_candidates[0][0]

                    # Branch 1: x_branch_i = 1
                    # Force all neighbors to 0
                    new_fixed_one = fixed_one_mask | (1 << branch_i)
                    new_fixed_zero = fixed_zero_mask
                    for nb in adj[branch_i]:
                        if not (fixed_one_mask & (1 << nb)):  # Not already fixed to 1
                            new_fixed_zero |= (1 << nb)

                    # Check for conflict: any neighbor already fixed to 1?
                    conflict = False
                    for nb in adj[branch_i]:
                        if fixed_one_mask & (1 << nb):
                            conflict = True
                            break
                    if not conflict:
                        # Quick upper bound: obj - sol[branch_i] (setting to 0)
                        # Actually the LP will handle this
                        branch_and_bound(new_fixed_one, new_fixed_zero, obj)

                    # Branch 2: x_branch_i = 0
                    new_fixed_zero2 = fixed_zero_mask | (1 << branch_i)
                    branch_and_bound(fixed_one_mask, new_fixed_zero2, obj)

                # Start B&B
                branch_and_bound(0, 0, init_obj)

                if best_bb_size > best_size:
                    best_indices = best_bb_indices
                    best_size = best_bb_size

    except Exception:
        pass

    # Final validation and conversion
    result = sorted([remaining[i] for i in best_indices])

    valid = True
    for i in range(len(result)):
        a = result[i]
        if (2 * a) in perfect_squares:
            valid = False
            break
        for j in range(i + 1, len(result)):
            b = result[j]
            if (a + b) in perfect_squares:
                valid = False
                break
        if not valid:
            break

    if not valid or not result:
        # Fallback: return a valid single element
        for a in range(1, N + 1):
            if 2 * a not in perfect_squares:
                return [a]
        return [1]

    return result

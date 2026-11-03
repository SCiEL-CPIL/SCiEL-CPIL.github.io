"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    def calculate_cost(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    # Precompute
    a_sym = np.array_equal(a, a.T)
    b_sym = np.array_equal(b, b.T)
    a_diag = np.diag(a).copy()
    b_diag = np.diag(b).copy()

    # Helper: efficient delta for swap (i, j)
    def calc_delta(i, j, perm):
        u = perm[i]
        v = perm[j]
        delta = (a_diag[i] - a_diag[j]) * (b_diag[v] - b_diag[u])
        diff_A = a[i, :] - a[j, :]
        diff_B = b[v, perm] - b[u, perm]
        total_dot = np.dot(diff_A, diff_B)
        exclude_i = diff_A[i] * diff_B[i]
        exclude_j = diff_A[j] * diff_B[j]
        off_diag_sum = total_dot - exclude_i - exclude_j
        delta += 2 * off_diag_sum
        return int(delta)

    # Generate initial solution using Hungarian on linearized cost
    def get_initial_solution():
        # Linearized cost: C[i, k] = sum_j a[i, j] * b[k, j]
        # This is a lower-bound relaxation; solving it gives a good starting point
        C = a @ b.T
        row_ind, col_ind = linear_sum_assignment(C)
        p_init = col_ind.copy()
        return p_init

    # Tabu search on a given starting permutation
    def tabu_search(start_p, max_time):
        current_p = start_p.copy()
        current_cost = calculate_cost(current_p)
        best_p = current_p.copy()
        best_cost = current_cost

        tabu_list = {}
        tabu_tenure = 80
        iteration = 0
        no_improve_count = 0
        max_no_improve = 500

        while time.time() - start_time < max_time:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1

            # Evaluate all pairs, find best improving or best non-tabu move
            for i in range(n):
                if time.time() - start_time > max_time:
                    break
                for j in range(i + 1, n):
                    delta = calc_delta(i, j, current_p)
                    key1 = (i, j)
                    key2 = (j, i)
                    is_tabu = (key1 in tabu_list and tabu_list[key1] > iteration) or \
                              (key2 in tabu_list and tabu_list[key2] > iteration)

                    # Aspiration: accept tabu move if it improves best
                    if delta < best_delta:
                        if not is_tabu or (current_cost + delta < best_cost):
                            best_delta = delta
                            best_i = i
                            best_j = j

            if best_i >= 0:
                # Apply best move
                current_p[best_i], current_p[best_j] = current_p[best_j], current_p[best_i]
                current_cost += best_delta
                tabu_list[(best_i, best_j)] = iteration + tabu_tenure
                tabu_list[(best_j, best_i)] = iteration + tabu_tenure

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                improved = True
            else:
                # No improving move found
                no_improve_count += 1

            iteration += 1

            # Clean up old tabu entries to save memory
            if iteration % 200 == 0:
                tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}

            if no_improve_count > max_no_improve:
                break

        return best_p, best_cost

    # Hungarian kick for diversification
    def hungarian_kick(current_p):
        B_perm = b[:, current_p]
        C = a @ B_perm.T
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = col_ind.copy()
        return new_p

    # Main loop: multiple restarts with tabu search
    best_p = None
    best_cost = float('inf')

    # First run: Hungarian initial solution
    p_init = get_initial_solution()
    best_p, best_cost = tabu_search(p_init, time_limit - 5.0)

    # Additional restarts if time permits
    restart_count = 0
    while time.time() - start_time < time_limit - 10.0:
        # Generate a perturbed initial solution
        if restart_count == 0:
            # Use a different linearization or random perturbation
            p_new = np.random.permutation(n).astype(np.int64)
        else:
            # Perturb the best solution
            p_new = best_p.copy()
            for _ in range(10):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    p_new[i], p_new[j] = p_new[j], p_new[i]

        # Run tabu search with reduced time
        remaining = time_limit - 10.0 - (time.time() - start_time)
        if remaining < 5.0:
            break
        t_start = time.time()
        p_sol, cost_sol = tabu_search(p_new, time.time() - start_time + remaining)

        if cost_sol < best_cost:
            best_cost = cost_sol
            best_p = p_sol

        restart_count += 1

    # Final safety check
    if best_p is None or sorted(best_p.tolist()) != list(range(n)):
        valid_p = np.arange(n, dtype=np.int64)
        return valid_p.tolist()

    return best_p.tolist()
# EVOLVE-BLOCK-END

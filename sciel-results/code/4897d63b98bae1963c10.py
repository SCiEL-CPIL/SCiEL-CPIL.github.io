import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)

    n_cust = len(demands)
    n_fac = len(capacities)

    # --- Helper: compute cost of an assignment ---
    def _compute_cost(assignment):
        load = np.zeros(n_fac)
        for i in range(n_cust):
            load[assignment[i]] += demands[i]
        if np.any(load > capacities + 1e-9):
            return np.inf
        open_total = np.sum(open_cost[load > 0])
        assign_total = np.sum(assign_cost[np.arange(n_cust), assignment])
        return open_total + assign_total

    # --- Helper: greedy assignment with a given facility ordering strategy ---
    def _greedy_assign(strategy='nearest'):
        """
        strategy: 'nearest' - sort facilities by assign_cost[i,f]
                  'lp'      - sort by LP solution (if available)
        """
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)

        # Order customers by demand descending (largest first)
        order = np.argsort(-demands)

        for i in order:
            d_i = demands[i]
            if strategy == 'nearest':
                fac_order = np.argsort(assign_cost[i])
            else:
                fac_order = np.argsort(assign_cost[i])  # fallback to nearest

            assigned = False
            for f in fac_order:
                if load[f] + d_i <= capacities[f]:
                    load[f] += d_i
                    result[i] = f
                    assigned = True
                    break

            if not assigned:
                # Try any facility with remaining capacity
                feasible = [f for f in range(n_fac) if load[f] + d_i <= capacities[f]]
                if feasible:
                    f_best = min(feasible, key=lambda f: assign_cost[i, f])
                    load[f_best] += d_i
                    result[i] = f_best
                else:
                    # Should not happen for feasible instances
                    result[i] = 0

        return result

    # --- Helper: local search with 1-move and 2-swap ---
    def _local_search(initial, time_limit):
        current = initial.astype(int).copy()
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i in range(n_cust):
            load[current[i]] += demands[i]
            fac_count[current[i]] += 1

        current_cost = _compute_cost(current)
        if np.isinf(current_cost):
            return initial

        best_cost = current_cost
        best_result = current.copy()

        start = time.time()

        # Precompute: for each customer, sort facilities by assign_cost
        # This is used to quickly find good candidates for reassignment
        fac_sort_by_cost = np.argsort(assign_cost, axis=1)  # shape (n_cust, n_fac)

        improved = True
        while improved and (time.time() - start < time_limit):
            improved = False

            # --- 1. Single Reassignment (first improvement) ---
            for i in range(n_cust):
                if time.time() - start > time_limit:
                    break
                cur_f = current[i]
                d_i = demands[i]
                best_delta = 0.0
                best_new_f = -1

                for new_f in range(n_fac):
                    if new_f == cur_f:
                        continue
                    if load[new_f] + d_i > capacities[new_f]:
                        continue

                    delta = assign_cost[i, new_f] - assign_cost[i, cur_f]

                    # Open cost change
                    if fac_count[cur_f] == 1:
                        delta -= open_cost[cur_f]
                    if fac_count[new_f] == 0:
                        delta += open_cost[new_f]

                    if delta < best_delta - 1e-12:
                        best_delta = delta
                        best_new_f = new_f

                if best_new_f != -1:
                    # Apply move
                    load[cur_f] -= d_i
                    load[best_new_f] += d_i
                    fac_count[cur_f] -= 1
                    fac_count[best_new_f] += 1
                    current[i] = best_new_f
                    current_cost += best_delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_result = current.copy()

            # --- 2. Swap (first improvement) ---
            for i in range(n_cust):
                if time.time() - start > time_limit:
                    break
                f_i = current[i]
                d_i = demands[i]
                for j in range(i + 1, n_cust):
                    f_j = current[j]
                    if f_i == f_j:
                        continue
                    d_j = demands[j]

                    # Check capacity after swap
                    if load[f_i] - d_i + d_j > capacities[f_i]:
                        continue
                    if load[f_j] - d_j + d_i > capacities[f_j]:
                        continue

                    delta = (assign_cost[i, f_j] + assign_cost[j, f_i]) - \
                            (assign_cost[i, f_i] + assign_cost[j, f_j])

                    # Open cost: fac_count[f_i] and fac_count[f_j] unchanged
                    if delta < -1e-12:
                        # Apply swap
                        load[f_i] = load[f_i] - d_i + d_j
                        load[f_j] = load[f_j] - d_j + d_i
                        current[i] = f_j
                        current[j] = f_i
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_result = current.copy()
                        break  # Restart j loop for this i

        return best_result

    # --- Helper: simulated annealing polish ---
    def _simulated_annealing(initial, time_limit):
        current = initial.astype(int).copy()
        load = np.zeros(n_fac)
        fac_count = np.zeros(n_fac, dtype=int)
        for i in range(n_cust):
            load[current[i]] += demands[i]
            fac_count[current[i]] += 1

        current_cost = _compute_cost(current)
        if np.isinf(current_cost):
            return initial

        best_cost = current_cost
        best_result = current.copy()

        start = time.time()
        T = 50.0  # initial temperature
        cooling = 0.995

        while time.time() - start < time_limit:
            T *= cooling
            if T < 0.1:
                break

            # Pick a random customer
            i = random.randint(0, n_cust - 1)
            cur_f = current[i]
            d_i = demands[i]

            # Pick a random new facility
            new_f = random.randint(0, n_fac - 1)
            if new_f == cur_f:
                continue

            if load[new_f] + d_i > capacities[new_f]:
                continue

            delta = assign_cost[i, new_f] - assign_cost[i, cur_f]
            if fac_count[cur_f] == 1:
                delta -= open_cost[cur_f]
            if fac_count[new_f] == 0:
                delta += open_cost[new_f]

            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                # Accept move
                load[cur_f] -= d_i
                load[new_f] += d_i
                fac_count[cur_f] -= 1
                fac_count[new_f] += 1
                current[i] = new_f
                current_cost += delta

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_result = current.copy()

        return best_result

    # --- Main algorithm ---
    # Generate multiple diverse initial solutions and pick the best after local search

    # Strategy 1: Nearest-neighbor greedy (largest demand first)
    sol1 = _greedy_assign('nearest')

    # Strategy 2: Randomized greedy - shuffle customers with same demand
    sol2 = _greedy_assign('nearest')

    # Strategy 3: Try a different ordering - smallest demand first
    result_smallest = np.zeros(n_cust, dtype=int)
    load_smallest = np.zeros(n_fac)
    order_smallest = np.argsort(demands)  # smallest first
    for i in order_smallest:
        d_i = demands[i]
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load_smallest[f] + d_i <= capacities[f]:
                load_smallest[f] += d_i
                result_smallest[i] = f
                break
        else:
            feasible = [f for f in range(n_fac) if load_smallest[f] + d_i <= capacities[f]]
            if feasible:
                f_best = min(feasible, key=lambda f: assign_cost[i, f])
                load_smallest[f_best] += d_i
                result_smallest[i] = f_best
            else:
                result_smallest[i] = 0
    sol3 = result_smallest

    # Evaluate initial solutions
    candidates = [sol1, sol2, sol3]
    candidate_costs = [_compute_cost(c) for c in candidates]

    # Filter out infeasible
    feasible_candidates = [(c, cost) for c, cost in zip(candidates, candidate_costs) if not np.isinf(cost)]

    if not feasible_candidates:
        # All infeasible, return a zero-filled array (should not happen)
        return np.zeros(n_cust, dtype=int)

    # Pick the best initial solution
    best_init = min(feasible_candidates, key=lambda x: x[1])
    best_init_sol = best_init[0]
    best_init_cost = best_init[1]

    # Run local search on the best initial solution
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        ls_result = _local_search(best_init_sol, remaining_time * 0.5)
        ls_cost = _compute_cost(ls_result)

        # Also try simulated annealing on the local search result
        remaining_after_ls = deadline - time.time()
        if remaining_after_ls > 10:
            sa_result = _simulated_annealing(ls_result, remaining_after_ls * 0.6)
            sa_cost = _compute_cost(sa_result)

            # Run local search again on the SA result if time permits
            remaining_after_sa = deadline - time.time()
            if remaining_after_sa > 5:
                ls2_result = _local_search(sa_result, remaining_after_sa * 0.8)
                ls2_cost = _compute_cost(ls2_result)

                # Pick the best
                all_results = [
                    (best_init_sol, best_init_cost),
                    (ls_result, ls_cost),
                    (sa_result, sa_cost),
                    (ls2_result, ls2_cost)
                ]
                final_result = min(all_results, key=lambda x: x[1])[0]
            else:
                all_results = [
                    (best_init_sol, best_init_cost),
                    (ls_result, ls_cost),
                    (sa_result, sa_cost)
                ]
                final_result = min(all_results, key=lambda x: x[1])[0]
        else:
            final_result = ls_result if ls_cost < best_init_cost else best_init_sol
    else:
        final_result = best_init_sol

    # Final validation
    load_final = np.zeros(n_fac)
    for i in range(n_cust):
        f = final_result[i]
        if f < 0 or f >= n_fac:
            return np.zeros(n_cust, dtype=int)
        load_final[f] += demands[i]

    if np.any(load_final > capacities + 1e-9):
        # Return the initial valid solution
        return best_init_sol

    return final_result

"""Combined program: vectorized best-improvement + 2-opt swaps + restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    timeout = 145.0  # Safety margin below the 150s wall-clock limit

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)

    def calc_total_cost(result, load):
        open_cost_sum = np.sum(open_cost[load > 0])
        assign_cost_sum = np.sum(assign_cost[np.arange(n_customers), result])
        return open_cost_sum + assign_cost_sum

    def deadline_hit():
        return time.time() - start_time > timeout

    def greedy_init(seed_order):
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=float)
        for i in seed_order:
            d = demands[i]
            remaining_cap = capacities - load
            feasible_mask = remaining_cap >= d
            if not np.any(feasible_mask):
                f = int(np.argmax(remaining_cap))
            else:
                costs = np.where(feasible_mask, assign_cost[i], np.inf)
                f = int(np.argmin(costs))
            result[i] = f
            load[f] += d
        return result, load

    # ---------- Vectorized single-move best improvement ----------
    def single_move_best(result, load, current_cost, stop_time):
        """Apply best single-move improvement until local optimum or timeout."""
        improved = True
        while improved:
            if time.time() > stop_time:
                break
            improved = False
            best_delta = 0.0
            best_move = None

            for i in range(n_customers):
                if time.time() > stop_time:
                    break
                f_old = result[i]
                d = demands[i]

                # Delta for moving customer i from f_old to each f_new
                delta_assign = assign_cost[i] - assign_cost[i, f_old]
                # Open cost change: f_new was empty -> +open, f_old becomes empty -> -open
                new_empty = (load == 0)
                old_will_be_empty = (load[f_old] == d)
                delta_open = np.where(new_empty, open_cost, 0.0)
                if old_will_be_empty:
                    delta_open = delta_open - open_cost[f_old]

                # Feasibility mask
                feasible = (load + d <= capacities)
                feasible[f_old] = False  # no self-move

                total_delta = delta_assign + delta_open
                total_delta = np.where(feasible, total_delta, np.inf)

                idx = int(np.argmin(total_delta))
                val = total_delta[idx]
                if val < best_delta:
                    best_delta = val
                    best_move = (i, f_old, idx)

            if best_move is not None and best_delta < 0:
                i, f_old, f_new = best_move
                load[f_old] -= demands[i]
                load[f_new] += demands[i]
                result[i] = f_new
                current_cost += best_delta
                improved = True
            else:
                break
        return result, load, current_cost

    # ---------- 2-opt swap move ----------
    def swap_move_best(result, load, current_cost, stop_time):
        """Try swapping assignments of two customers; apply best improving swap."""
        improved = True
        while improved:
            if time.time() > stop_time:
                break
            improved = False
            best_delta = 0.0
            best_pair = None

            for i in range(n_customers):
                if time.time() > stop_time:
                    break
                fi = result[i]
                di = demands[i]
                for j in range(i + 1, n_customers):
                    if time.time() > stop_time:
                        break
                    fj = result[j]
                    if fj == fi:
                        continue
                    dj = demands[j]
                    # Feasibility after swap
                    if load[fi] - di + dj > capacities[fi]:
                        continue
                    if load[fj] - dj + di > capacities[fj]:
                        continue

                    # Assignment cost delta
                    delta_assign = (assign_cost[i, fj] + assign_cost[j, fi]
                                    - assign_cost[i, fi] - assign_cost[j, fj])

                    # Open cost delta
                    delta_open = 0.0
                    # fi: was load[fi], becomes load[fi]-di+dj
                    if load[fi] == di and (load[fi] - di + dj) == 0:
                        delta_open -= open_cost[fi]
                    elif load[fi] > 0 and (load[fi] - di + dj) == 0:
                        delta_open -= open_cost[fi]
                    # Actually handle generically:
                    # recompute
                    delta_open = 0.0
                    was_open_i = load[fi] > 0
                    will_open_i = (load[fi] - di + dj) > 0
                    if not was_open_i and will_open_i:
                        delta_open += open_cost[fi]
                    if was_open_i and not will_open_i:
                        delta_open -= open_cost[fi]
                    was_open_j = load[fj] > 0
                    will_open_j = (load[fj] - dj + di) > 0
                    if not was_open_j and will_open_j:
                        delta_open += open_cost[fj]
                    if was_open_j and not will_open_j:
                        delta_open -= open_cost[fj]

                    total_delta = delta_assign + delta_open
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_pair = (i, j, fi, fj)

            if best_pair is not None and best_delta < 0:
                i, j, fi, fj = best_pair
                di, dj = demands[i], demands[j]
                load[fi] = load[fi] - di + dj
                load[fj] = load[fj] - dj + di
                result[i] = fj
                result[j] = fi
                current_cost += best_delta
                improved = True
            else:
                break
        return result, load, current_cost

    # ---------- Main: multiple restarts ----------
    best_cost = np.inf
    best_result = None
    best_load = None

    rng = np.random.default_rng(12345)

    # Restart 0: deterministic largest-demand-first
    order0 = np.argsort(-demands)
    result, load = greedy_init(order0)
    current_cost = calc_total_cost(result, load)
    if current_cost < best_cost:
        best_cost = current_cost
        best_result = result.copy()
        best_load = load.copy()

    # Local search on this
    result, load, current_cost = single_move_best(result, load, current_cost, timeout - 15)
    if current_cost < best_cost:
        best_cost = current_cost
        best_result = result.copy()
        best_load = load.copy()

    # 2-opt on this
    result, load, current_cost = swap_move_best(result, load, current_cost, timeout - 15)
    if current_cost < best_cost:
        best_cost = current_cost
        best_result = result.copy()
        best_load = load.copy()

    # Additional random restarts if time permits
    restart_budget = 6
    for r in range(restart_budget):
        if time.time() > timeout - 20:
            break
        # Randomized greedy: shuffle with a bias toward large demand
        perm = np.arange(n_customers)
        rng.shuffle(perm)
        # Blend: sort by (random + small demand weight) to diversify
        order_r = np.argsort(rng.random(n_customers) - 0.1 * demands)
        result, load = greedy_init(order_r)
        current_cost = calc_total_cost(result, load)
        if current_cost < best_cost:
            best_cost = current_cost
            best_result = result.copy()
            best_load = load.copy()

        # Local search
        result, load, current_cost = single_move_best(result, load, current_cost, timeout - 15)
        if current_cost < best_cost:
            best_cost = current_cost
            best_result = result.copy()
            best_load = load.copy()

        # 2-opt
        result, load, current_cost = swap_move_best(result, load, current_cost, timeout - 15)
        if current_cost < best_cost:
            best_cost = current_cost
            best_result = result.copy()
            best_load = load.copy()

        # Final polish: alternate single + swap a few times
        for _ in range(3):
            if time.time() > timeout - 15:
                break
            result, load, current_cost = single_move_best(result, load, current_cost, timeout - 15)
            result, load, current_cost = swap_move_best(result, load, current_cost, timeout - 15)
            if current_cost < best_cost:
                best_cost = current_cost
                best_result = result.copy()
                best_load = load.copy()

    # Safety: ensure a feasible result exists (fallback to greedy if none recorded)
    if best_result is None:
        result, load = greedy_init(np.argsort(-demands))
        best_result = result
        best_load = load

    # Final feasibility verification
    final_load = np.zeros(n_facilities, dtype=float)
    for i in range(n_customers):
        f = best_result[i]
        if f < 0 or f >= n_facilities:
            best_result[i] = 0
            f = 0
        final_load[f] += demands[i]
    # If any capacity violated, rebuild with greedy (should not happen)
    if np.any(final_load > capacities + 1e-9):
        result, load = greedy_init(np.argsort(-demands))
        best_result = result

    return best_result.astype(int)
# EVOLVE-BLOCK-END

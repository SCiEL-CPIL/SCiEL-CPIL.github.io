"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each of n customers to one of m facilities.
    Returns an integer array of length n with facility indices.
    """
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n = len(demands)
    m = len(capacities)

    # Total demand
    total_demand = demands.sum()

    # If total demand exceeds total capacity, we need to be careful
    # But the problem states it's feasible, so assume total_demand <= sum(capacities)

    # Strategy: Try to solve as a capacitated facility location problem
    # Using a combination of:
    # 1. LP relaxation via scipy (if available) or manual approach
    # 2. Local search / swap improvements
    # 3. Greedy initialization with multiple restarts

    # First, let's try a smart greedy approach with look-ahead
    # Then improve with local search

    # Greedy initialization: largest demand first, cheapest feasible
    def greedy_init():
        result = np.zeros(n, dtype=int)
        load = np.zeros(m)
        order = np.argsort(-demands)
        for i in order:
            # Find cheapest feasible facility
            feasible = []
            for f in range(m):
                if load[f] + demands[i] <= capacities[f]:
                    feasible.append(f)
            if not feasible:
                # Should not happen if problem is feasible
                # Assign to facility with most remaining capacity
                remaining = capacities - load
                f = np.argmax(remaining)
            else:
                # Sort by cost, pick cheapest
                costs = assign_cost[i, feasible]
                f = feasible[np.argmin(costs)]
            load[f] += demands[i]
            result[i] = f
        return result, load

    def compute_cost(result, load=None):
        """Compute total cost: open costs + assignment costs."""
        if load is None:
            load = np.zeros(m)
            for i in range(n):
                load[result[i]] += demands[i]
        # Open cost: facilities with at least one customer
        open_facilities = set(result.tolist())
        total_open = sum(open_cost[f] for f in open_facilities)
        # Assignment cost
        total_assign = sum(assign_cost[i, result[i]] for i in range(n))
        return total_open + total_assign, load

    # Local search: try moving customers to different facilities
    def local_search(result, max_iters=5000):
        load = np.zeros(m)
        for i in range(n):
            load[result[i]] += demands[i]

        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1

            # Try moving each customer to a better facility
            # Process in random-ish order to escape local optima
            for i in range(n):
                current_f = result[i]
                current_cost = assign_cost[i, current_f]
                current_open_penalty = 0.0

                # Check if moving i away from current_f would close the facility
                # and thus save open_cost
                would_close = (load[current_f] == demands[i])

                best_f = current_f
                best_delta = 0.0  # negative is improvement

                for f in range(m):
                    if f == current_f:
                        continue
                    # Check capacity
                    if load[f] + demands[i] > capacities[f]:
                        continue

                    # Cost change:
                    # - Remove i from current_f: save assign_cost[i, current_f]
                    # - Add i to f: pay assign_cost[i, f]
                    # - If current_f becomes empty, save open_cost[current_f]
                    # - If f was empty and now has i, pay open_cost[f]

                    delta_assign = assign_cost[i, f] - assign_cost[i, current_f]

                    delta_open = 0.0
                    if would_close:
                        delta_open -= open_cost[current_f]
                    if load[f] == 0:
                        delta_open += open_cost[f]

                    total_delta = delta_assign + delta_open

                    if total_delta < best_delta - 1e-9:
                        best_delta = total_delta
                        best_f = f

                if best_f != current_f:
                    # Move customer i from current_f to best_f
                    load[current_f] -= demands[i]
                    load[best_f] += demands[i]
                    result[i] = best_f
                    improved = True

                    # Check deadline
                    if time.time() > deadline:
                        return result, load

        return result, load

    # Multi-start local search
    best_result = None
    best_cost = float('inf')

    # Start 1: greedy
    result1, load1 = greedy_init()
    result1, load1 = local_search(result1)
    cost1, _ = compute_cost(result1, load1)
    if cost1 < best_cost:
        best_cost = cost1
        best_result = result1.copy()

    # Start 2: cheapest facility assignment (ignoring capacity, then fix)
    result2 = np.zeros(n, dtype=int)
    for i in range(n):
        result2[i] = np.argmin(assign_cost[i])

    # Fix capacity violations: move customers from overloaded facilities
    load2 = np.zeros(m)
    for i in range(n):
        load2[result2[i]] += demands[i]

    # Iteratively fix overloads
    for _ in range(100):
        overloads = []
        for f in range(m):
            if load2[f] > capacities[f] + 1e-9:
                overloads.append(f)
        if not overloads:
            break

        for f in overloads:
            # Find customers in f that can be moved
            customers_in_f = [i for i in range(n) if result2[i] == f]
            # Sort by how much they "prefer" f (lower cost difference to alternatives)
            for i in customers_in_f:
                if load2[f] <= capacities[f] + 1e-9:
                    break
                # Find best alternative
                best_alt = -1
                best_cost_alt = float('inf')
                for f2 in range(m):
                    if f2 == f:
                        continue
                    if load2[f2] + demands[i] <= capacities[f2]:
                        c = assign_cost[i, f2]
                        if c < best_cost_alt:
                            best_cost_alt = c
                            best_alt = f2
                if best_alt >= 0:
                    load2[f] -= demands[i]
                    load2[best_alt] += demands[i]
                    result2[i] = best_alt

    result2, load2 = local_search(result2)
    cost2, _ = compute_cost(result2, load2)
    if cost2 < best_cost:
        best_cost = cost2
        best_result = result2.copy()

    # Start 3: Random perturbation of greedy
    if time.time() < deadline - 10:
        rng = np.random.RandomState(42)
        for restart in range(5):
            result3 = np.zeros(n, dtype=int)
            load3 = np.zeros(m)
            order = np.argsort(-demands)
            for i in order:
                feasible = [f for f in range(m) if load3[f] + demands[i] <= capacities[f]]
                if not feasible:
                    remaining = capacities - load3
                    f = np.argmax(remaining)
                else:
                    # With some probability, pick a non-cheapest feasible
                    costs = assign_cost[i, feasible]
                    idx = np.argsort(costs)
                    if rng.random() < 0.3 and len(idx) > 1:
                        # Pick from top 3
                        k = min(3, len(idx))
                        f = idx[rng.randint(k)]
                    else:
                        f = idx[0]
                load3[f] += demands[i]
                result3[i] = f

            result3, load3 = local_search(result3)
            cost3, _ = compute_cost(result3, load3)
            if cost3 < best_cost:
                best_cost = cost3
                best_result = result3.copy()

            if time.time() > deadline - 5:
                break

    # Final check: ensure validity
    if best_result is None:
        best_result = result1

    # Verify capacities
    final_load = np.zeros(m)
    for i in range(n):
        final_load[best_result[i]] += demands[i]

    # If any capacity violation, fix by moving
    for f in range(m):
        while final_load[f] > capacities[f] + 1e-9:
            # Find a customer in f that can be moved
            moved = False
            for i in range(n):
                if best_result[i] == f:
                    for f2 in range(m):
                        if f2 != f and final_load[f2] + demands[i] <= capacities[f2]:
                            final_load[f] -= demands[i]
                            final_load[f2] += demands[i]
                            best_result[i] = f2
                            moved = True
                            break
                    if moved:
                        break
            if not moved:
                break  # Should not happen

    return best_result.astype(int)
# EVOLVE-BLOCK-END

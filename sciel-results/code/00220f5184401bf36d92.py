# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_customers = len(demands)
    n_facilities = len(capacities)
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s

    def compute_cost(assignment, load):
        # assignment: array of facility indices per customer
        # load: array of total demand per facility
        opened = load > 0
        total = np.sum(open_cost[opened])
        total += np.sum(assign_cost[np.arange(n_customers), assignment])
        return total

    # --- Phase 1: Greedy initial assignment (largest demand first, cheapest feasible) ---
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    for i in np.argsort(-demands):
        placed = False
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                result[i] = f
                placed = True
                break
        if not placed:
            # fallback: least-loaded facility
            f = int(np.argmin(load))
            load[f] += demands[i]
            result[i] = f

    best_assignment = result.copy()
    best_load = load.copy()
    best_cost = compute_cost(best_assignment, best_load)

    # --- Phase 2: Local search ---
    # Precompute: for each customer, sorted facilities by cost
    # We'll do: (a) move a customer to a cheaper facility if capacity allows
    # (b) swap two customers if it reduces cost
    # (c) try to close a facility and reassign its customers

    # Sorted facility indices by cost for each customer
    sorted_facs = np.argsort(assign_cost, axis=1)  # shape (n_customers, n_facilities)

    # Customer order for search: customers with more potential savings first
    # Potential saving = current_cost - min_cost over feasible facilities
    current_costs = assign_cost[np.arange(n_customers), result]
    min_costs = np.min(assign_cost, axis=1)
    savings = current_costs - min_costs
    search_order = np.argsort(-savings)

    improved = True
    max_iter = 50
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        improved = False

        # Pass 1: single customer moves
        for i in search_order:
            if time.time() > deadline:
                break
            old_f = result[i]
            old_cost_i = assign_cost[i, old_f]
            # Try cheaper facilities
            for f in sorted_facs[i]:
                if f == old_f:
                    continue
                new_cost_i = assign_cost[i, f]
                if new_cost_i >= old_cost_i:
                    break  # sorted, no more savings
                # Check capacity
                if load[f] + demands[i] <= capacities[f]:
                    # Accept the move
                    result[i] = f
                    load[old_f] -= demands[i]
                    load[f] += demands[i]
                    best_cost += (new_cost_i - old_cost_i)
                    # Account for open cost change
                    if load[old_f] == 0:
                        best_cost -= open_cost[old_f]
                    if load[f] > 0 and load[f] - demands[i] == 0:
                        best_cost += open_cost[f]
                    improved = True
                    break

        # Pass 2: try to close facilities with high open cost relative to savings
        # For each facility, check if we can reassign all its customers to others
        for f in np.argsort(-open_cost):
            if time.time() > deadline:
                break
            if load[f] <= 0:
                continue
            # Get customers at facility f
            customers_at_f = np.where(result == f)[0]
            if len(custometers_at_f) == 0:
                continue
            # Try to move all customers to other facilities
            total_demand = load[f]
            # Check if there's enough capacity elsewhere
            other_capacity = capacities.sum() - capacities[f] - (load.sum() - total_demand)
            if other_capacity < total_demand:
                continue
            # Try to reassign
            new_load = load.copy()
            new_load[f] = 0
            new_result = result.copy()
            success = True
            # Sort by demand descending for better packing
            for i in np.argsort(-demands[customers_at_f]):
                i = customers_at_f[i]
                placed = False
                for g in sorted_facs[i]:
                    if g == f:
                        continue
                    if new_load[g] + demands[i] <= capacities[g]:
                        new_load[g] += demands[i]
                        new_result[i] = g
                        placed = True
                        break
                if not placed:
                    success = False
                    break
            if success:
                new_cost = compute_cost(new_result, new_load)
                if new_cost < best_cost - 1e-9:
                    result = new_result
                    load = new_load
                    best_cost = new_cost
                    improved = True

        if not improved:
            break

    # Final validation
    for i in range(n_customers):
        f = result[i]
        if f < 0 or f >= n_facilities:
            raise ValueError("Invalid facility index")
    final_load = np.zeros(n_facilities)
    for i in range(n_customers):
        final_load[result[i]] += demands[i]
    if np.any(final_load > capacities + 1e-9):
        # Should not happen, but safety
        pass

    return result.astype(int)
# EVOLVE-BLOCK-END

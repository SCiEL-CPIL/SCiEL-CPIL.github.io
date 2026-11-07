"""Stronger solver for capacitated facility location."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    Strategy: diverse initial solutions + fast vectorized local search with swaps.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n_customers = len(demands)
    n_facilities = len(capacities)

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    def compute_cost(assignment):
        """Compute total cost. Returns inf if infeasible."""
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            f = int(assignment[c])
            if f < 0 or f >= n_facilities:
                return np.inf
            load[f] += demands[c]
        for f in range(n_facilities):
            if load[f] > capacities[f] + 1e-9:
                return np.inf
        cost = 0.0
        used = np.zeros(n_facilities, dtype=bool)
        for c in range(n_customers):
            f = int(assignment[c])
            cost += assign_cost[c, f]
            used[f] = True
        for f in range(n_facilities):
            if used[f]:
                cost += open_cost[f]
        return cost

    def greedy_assignment(order, rng=None):
        """Greedy: assign in given order to cheapest feasible facility."""
        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        for c in order:
            costs = assign_cost[c].copy()
            feasible = load + demands[c] <= capacities
            costs[~feasible] = np.inf
            if rng is not None:
                feasible_mask = feasible
                if feasible_mask.any():
                    pert = rng.uniform(0, 0.05 * np.mean(costs[feasible_mask]), size=n_facilities)
                    pert[~feasible_mask] = 0
                    costs += pert
            f = int(np.argmin(costs))
            if f >= 0 and feasible[f]:
                result[c] = f
                load[f] += demands[c]
            else:
                return None
        return result

    def local_search(assignment, max_passes=50):
        """
        Fast local search: single reassignments + pairwise swaps.
        Uses vectorized evaluation where possible.
        """
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]

        current_cost = compute_cost(assignment)
        if current_cost == np.inf:
            return assignment

        best_cost = current_cost
        best_assignment = assignment.copy()

        for _pass in range(max_passes):
            if time.time() > deadline:
                break

            improved = False

            # --- Single reassignment pass ---
            for c in range(n_customers):
                if time.time() > deadline:
                    break
                old_f = int(assignment[c])
                load[old_f] -= demands[c]

                # Vectorized: compute delta for all facilities
                deltas = assign_cost[c] - assign_cost[c, old_f]  # shape (n_facilities,)
                # Add open cost if facility becomes newly used
                newly_used = (load == 0)
                deltas[newly_used] += open_cost[newly_used]
                # Subtract open cost if old facility becomes unused
                if load[old_f] == 0:
                    deltas -= open_cost[old_f]

                # Feasibility
                feasible = load + demands[c] <= capacities + 1e-9
                feasible[old_f] = False  # don't stay
                deltas[~feasible] = np.inf

                best_idx = int(np.argmin(deltas))
                best_delta = deltas[best_idx]

                if best_delta < -1e-9 and feasible[best_idx]:
                    assignment[c] = best_idx
                    load[best_idx] += demands[c]
                    current_cost += best_delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_assignment = assignment.copy()
                else:
                    load[old_f] += demands[c]

            # --- Pairwise swap pass (limited) ---
            if time.time() > deadline - 5:
                break

            # Try swapping pairs of customers
            n_swaps_checked = 0
            max_swaps = 2000
            for c1 in range(0, n_customers):
                if time.time() > deadline - 2:
                    break
                f1 = int(assignment[c1])
                for c2 in range(c1 + 1, n_customers):
                    if time.time() > deadline - 2:
                        break
                    if n_swaps_checked >= max_swaps:
                        break
                    n_swaps_checked += 1
                    f2 = int(assignment[c2])
                    if f1 == f2:
                        continue

                    # Check feasibility after swap
                    load_f1 = load[f1] - demands[c1] + demands[c2]
                    load_f2 = load[f2] - demands[c2] + demands[c1]
                    if load_f1 > capacities[f1] + 1e-9 or load_f2 > capacities[f2] + 1e-9:
                        continue

                    # Cost delta
                    delta = (assign_cost[c1, f2] + assign_cost[c2, f1]) - \
                            (assign_cost[c1, f1] + assign_cost[c2, f2])

                    # Open cost changes
                    # f1: was used (load[f1] > 0), after swap load_f1
                    # f2: was used (load[f2] > 0), after swap load_f2
                    # Only change if a facility goes from 0 to >0 or vice versa
                    old_load_f1 = load[f1]
                    old_load_f2 = load[f2]

                    if old_load_f1 > 0 and load_f1 == 0:
                        delta -= open_cost[f1]
                    elif old_load_f1 == 0 and load_f1 > 0:
                        delta += open_cost[f1]

                    if old_load_f2 > 0 and load_f2 == 0:
                        delta -= open_cost[f2]
                    elif old_load_f2 == 0 and load_f2 > 0:
                        delta += open_cost[f2]

                    if delta < -1e-9:
                        # Apply swap
                        assignment[c1] = f2
                        assignment[c2] = f1
                        load[f1] = load_f1
                        load[f2] = load_f2
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_assignment = assignment.copy()

            if not improved:
                break

        return best_assignment

    # Phase 1: Generate diverse initial solutions
    best_solution = None
    best_cost = np.inf

    orderings = [
        np.arange(n_customers),
        np.argsort(-demands),
        np.argsort(demands),
        np.argsort(assign_cost.sum(axis=1)),
        np.argsort(assign_cost.sum(axis=1) / demands),
        np.argsort(-demands / (assign_cost.sum(axis=1) + 1)),
    ]

    for order in orderings:
        if time.time() > deadline:
            break
        sol = greedy_assignment(order)
        if sol is not None:
            cost = compute_cost(sol)
            if cost < best_cost:
                best_cost = cost
                best_solution = sol.copy()

    # Randomized greedy
    rng = np.random.RandomState(42)
    n_random = 40
    for i in range(n_random):
        if time.time() > deadline - 30:
            break
        order = rng.permutation(n_customers)
        sol = greedy_assignment(order, rng)
        if sol is not None:
            cost = compute_cost(sol)
            if cost < best_cost:
                best_cost = cost
                best_solution = sol.copy()

        # Biased ordering
        bias = rng.random()
        score = bias * (-demands) + (1 - bias) * assign_cost.sum(axis=1)
        order = np.argsort(score)
        sol = greedy_assignment(order, rng)
        if sol is not None:
            cost = compute_cost(sol)
            if cost < best_cost:
                best_cost = cost
                best_solution = sol.copy()

    # Phase 2: Local search on best
    if best_solution is not None and time.time() < deadline:
        improved = local_search(best_solution, max_passes=30)
        cost = compute_cost(improved)
        if cost < best_cost:
            best_cost = cost
            best_solution = improved

    # Phase 3: More diverse candidates + local search
    n_more = 15
    for i in range(n_more):
        if time.time() > deadline - 15:
            break

        if i % 2 == 0:
            order = rng.permutation(n_customers)
            sol = greedy_assignment(order, rng)
        else:
            if best_solution is None:
                sol = greedy_assignment(np.argsort(-demands))
            else:
                sol = best_solution.copy()
                # Perturb
                for _ in range(rng.randint(5, 15)):
                    c = rng.randint(n_customers)
                    old_f = int(sol[c])
                    load_tmp = np.zeros(n_facilities)
                    for cc in range(n_customers):
                        load_tmp[int(sol[cc])] += demands[cc]
                    load_tmp[old_f] -= demands[c]
                    feasible = np.where(load_tmp + demands[c] <= capacities)[0]
                    if len(feasible) > 1:
                        new_f = int(feasible[rng.randint(len(feasible))])
                        sol[c] = new_f

        if sol is None:
            continue

        ls_sol = local_search(sol, max_passes=20)
        cost = compute_cost(ls_sol)
        if cost < best_cost:
            best_cost = cost
            best_solution = ls_sol.copy()

    # Final local search
    if best_solution is not None and time.time() < deadline:
        final = local_search(best_solution, max_passes=40)
        cost = compute_cost(final)
        if cost < best_cost:
            best_cost = cost
            best_solution = final

    if best_solution is None:
        best_solution = greedy_assignment(np.argsort(-demands))
        if best_solution is None:
            best_solution = np.zeros(n_customers, dtype=int)

    return best_solution.astype(int)
# EVOLVE-BLOCK-END

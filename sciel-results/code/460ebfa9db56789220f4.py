"""
Combined CFLP solver: Benders-style facility opening/closing + diverse greedy starts + vectorized local search + simulated annealing polish.

Strategy:
1. Generate diverse initial solutions via multiple greedy orderings (deterministic + randomized).
2. Use a Benders-like loop to try closing expensive, under-utilized facilities and reopening if beneficial.
3. Polish with a fast local search (single moves + pairwise swaps) and a short simulated-annealing pass.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to one facility. Returns integer array of facility indices.
    """
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s kill

    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    n_customers = len(demands)
    n_facilities = len(capacities)

    def compute_cost(assignment):
        """Compute total cost; return inf if infeasible."""
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

    def greedy_assignment(order=None, random_seed=None, noise_scale=0.0):
        """
        Greedy assignment: assign customers in given order to cheapest feasible facility.
        Optional random noise on costs for diversity.
        """
        if order is None:
            order = np.arange(n_customers)

        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)

        for c in order:
            costs = assign_cost[c].copy()
            feasible = load + demands[c] <= capacities
            costs[~feasible] = np.inf

            if random_seed is not None and noise_scale > 0:
                rng = np.random.RandomState(random_seed)
                feasible_costs = costs[feasible]
                if np.any(feasible):
                    mean_feas = np.mean(feasible_costs)
                    costs[feasible] += rng.uniform(0, noise_scale * mean_feas, size=np.sum(feasible))

            f = int(np.argmin(costs))
            if f >= 0 and feasible[f]:
                result[c] = f
                load[f] += demands[c]
            else:
                return None

        return result

    def local_search(assignment, max_time=10.0):
        """
        Local search: single reassignments and pairwise swaps.
        Returns improved assignment.
        """
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]

        current_cost = compute_cost(assignment)
        if np.isinf(current_cost):
            return assignment

        start_ls = time.time()
        improved = True
        while improved and (time.time() - start_ls) < max_time:
            improved = False

            # Single moves
            for c in range(n_customers):
                if time.time() > deadline:
                    return assignment
                old_f = int(assignment[c])
                load[old_f] -= demands[c]

                best_new_f = old_f
                best_delta = 0.0

                for new_f in range(n_facilities):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[c] > capacities[new_f]:
                        continue

                    delta = assign_cost[c, new_f] - assign_cost[c, old_f]
                    if load[new_f] == 0:
                        delta += open_cost[new_f]
                    if load[old_f] == 0:
                        delta -= open_cost[old_f]

                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_new_f = new_f

                if best_new_f != old_f:
                    assignment[c] = best_new_f
                    load[best_new_f] += demands[c]
                    current_cost += best_delta
                    improved = True
                else:
                    load[old_f] += demands[c]

            # Pairwise swaps (only if time permits)
            if not improved and (time.time() - start_ls) < max_time * 0.4:
                for c1 in range(n_customers):
                    if time.time() > deadline:
                        return assignment
                    f1 = int(assignment[c1])
                    for c2 in range(c1 + 1, n_customers):
                        f2 = int(assignment[c2])
                        if f1 == f2:
                            continue
                        if load[f1] - demands[c1] + demands[c2] > capacities[f1]:
                            continue
                        if load[f2] - demands[c2] + demands[c1] > capacities[f2]:
                            continue

                        delta = (assign_cost[c1, f2] + assign_cost[c2, f1]) - \
                                (assign_cost[c1, f1] + assign_cost[c2, f2])

                        if delta < -1e-9:
                            assignment[c1] = f2
                            assignment[c2] = f1
                            load[f1] = load[f1] - demands[c1] + demands[c2]
                            load[f2] = load[f2] - demands[c2] + demands[c1]
                            current_cost += delta
                            improved = True
                            break
                    if improved:
                        break

        return assignment

    def simulated_annealing(assignment, max_iter=1500, T_start=5.0, T_end=0.05):
        """
        Simulated annealing polish: accept worse moves with decreasing probability.
        """
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]

        current_cost = compute_cost(assignment)
        if np.isinf(current_cost):
            return assignment

        best_assignment = assignment.copy()
        best_cost = current_cost

        for it in range(max_iter):
            if time.time() > deadline:
                break

            T = T_start * (T_end / T_start) ** (it / max(1, max_iter - 1))
            c = np.random.randint(n_customers)
            old_f = int(assignment[c])

            load[old_f] -= demands[c]
            feasible = np.where(load + demands[c] <= capacities)[0]

            if len(feasible) == 0:
                load[old_f] += demands[c]
                continue

            new_f = int(feasible[np.random.randint(len(feasible))])

            if new_f == old_f:
                load[old_f] += demands[c]
                continue

            delta = assign_cost[c, new_f] - assign_cost[c, old_f]
            if load[new_f] == 0:
                delta += open_cost[new_f]
            if load[old_f] == 0:
                delta -= open_cost[old_f]

            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                assignment[c] = new_f
                load[new_f] += demands[c]
                current_cost += delta

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assignment = assignment.copy()
            else:
                load[old_f] += demands[c]

        return best_assignment

    # --- Main Logic ---

    best_solution = None
    best_cost = np.inf

    # Phase 1: Diverse initial solutions
    # Deterministic orderings
    orderings = [
        np.arange(n_customers),
        np.argsort(-demands),
        np.argsort(demands),
        np.argsort(assign_cost.sum(axis=1)),
        np.argsort(assign_cost.sum(axis=1) / np.maximum(demands, 1e-9)),
        np.argsort(assign_cost.min(axis=1)),
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

    # Randomized greedy variants
    rng = np.random.RandomState(42)
    n_random_greedy = 25
    for i in range(n_random_greedy):
        if time.time() > deadline:
            break

        order = rng.permutation(n_customers)
        noise = rng.uniform(0.0, 0.15)
        sol = greedy_assignment(order, random_seed=rng.randint(100000), noise_scale=noise)
        if sol is not None:
            cost = compute_cost(sol)
            if cost < best_cost:
                best_cost = cost
                best_solution = sol.copy()

    # Phase 2: Local search on best solution
    if best_solution is not None and time.time() < deadline - 10:
        improved = local_search(best_solution, max_time=min(15.0, deadline - time.time() - 5))
        improved_cost = compute_cost(improved)
        if improved_cost < best_cost:
            best_cost = improved_cost
            best_solution = improved.copy()

    # Phase 3: Benders-like facility closing/opening
    if best_solution is not None and time.time() < deadline - 15:
        open_set = set(int(f) for f in best_solution)

        for benders_iter in range(20):
            if time.time() > deadline - 10:
                break

            # Try to close the most expensive under-utilized facility
            load = np.zeros(n_facilities)
            for c in range(n_customers):
                load[int(best_solution[c])] += demands[c]

            # Sort open facilities by open_cost descending
            sorted_open = sorted(open_set, key=lambda f: -open_cost[f])

            closed_any = False
            for f_to_close in sorted_open:
                if time.time() > deadline - 10:
                    break

                # Check if closing is feasible: can all customers at f_to_close be reassigned?
                trial_sol = best_solution.copy()
                load_trial = load.copy()
                customers_to_move = [c for c in range(n_customers) if int(trial_sol[c]) == f_to_close]
                customers_to_move.sort(key=lambda c: -demands[c])

                feasible = True
                for c in customers_to_move:
                    best_f = -1
                    best_move_cost = np.inf
                    for new_f in open_set:
                        if new_f == f_to_close:
                            continue
                        if load_trial[new_f] + demands[c] <= capacities[new_f]:
                            mc = assign_cost[c, new_f]
                            if mc < best_move_cost:
                                best_move_cost = mc
                                best_f = new_f
                    if best_f == -1:
                        feasible = False
                        break
                    trial_sol[c] = best_f
                    load_trial[best_f] += demands[c]

                if feasible:
                    load_trial[f_to_close] = 0.0
                    trial_cost = compute_cost(trial_sol)
                    if trial_cost < best_cost - 1e-9:
                        best_cost = trial_cost
                        best_solution = trial_sol.copy()
                        open_set.discard(f_to_close)
                        closed_any = True
                        break

            if not closed_any:
                # Try opening a closed facility if it helps
                closed_facilities = [f for f in range(n_facilities) if f not in open_set]
                if closed_facilities:
                    # Try opening the one with lowest open_cost
                    f_to_open = min(closed_facilities, key=lambda f: open_cost[f])
                    new_open_set = open_set | {f_to_open}

                    # Re-solve with all facilities in new_open_set via greedy
                    sol, _ = None, None
                    sol = greedy_assignment(np.argsort(-demands))
                    if sol is not None:
                        # Restrict to new_open_set if needed (greedy may use all)
                        # Actually greedy uses all facilities, so just compute cost
                        cost = compute_cost(sol)
                        if cost < best_cost:
                            best_cost = cost
                            best_solution = sol.copy()
                            open_set = new_open_set
                    break  # Don't loop too much on opening

    # Phase 4: Final polish with local search + SA
    if best_solution is not None and time.time() < deadline - 3:
        improved = local_search(best_solution, max_time=min(8.0, deadline - time.time() - 2))
        improved_cost = compute_cost(improved)
        if improved_cost < best_cost:
            best_cost = improved_cost
            best_solution = improved.copy()

        if time.time() < deadline - 2:
            sa_sol = simulated_annealing(best_solution, max_iter=1000, T_start=3.0, T_end=0.02)
            sa_cost = compute_cost(sa_sol)
            if sa_cost < best_cost:
                best_cost = sa_cost
                best_solution = sa_sol.copy()

    if best_solution is None:
        # Fallback
        best_solution = greedy_assignment(np.argsort(-demands))
        if best_solution is None:
            best_solution = np.zeros(n_customers, dtype=int)

    return best_solution.astype(int)

# EVOLVE-BLOCK-END

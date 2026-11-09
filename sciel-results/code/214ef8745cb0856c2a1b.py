"""Capacitated facility location: MILP with tight linking + multi-start local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    n_cust = len(demands)
    n_fac = len(capacities)

    def compute_score(res):
        load = np.zeros(n_fac)
        np.add.at(load, res, demands)
        if np.any(load > capacities + 1e-9):
            return np.inf
        used = load > 0
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n_cust), res])

    def greedy_init(seed=None):
        """Greedy: assign customers in order of demand (largest first) to cheapest feasible facility."""
        rng = np.random.RandomState(seed)
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            placed = False
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    res[i] = f
                    load[f] += demands[i]
                    placed = True
                    break
            if not placed:
                space = capacities - load
                best_f = np.argmax(space)
                if space[best_f] >= demands[i]:
                    res[i] = best_f
                    load[best_f] += demands[i]
                else:
                    res[i] = fac_order[0]
        return res

    def local_search(res, load, max_iter=5000, seed=0):
        """Local search with relocate and swap moves."""
        rng = np.random.RandomState(seed)
        improved = True
        it = 0
        while improved and it < max_iter:
            if time.time() > deadline:
                break
            improved = False
            it += 1
            # Relocate moves
            for i in rng.permutation(n_cust):
                current_f = res[i]
                best_f = -1
                best_delta = 1e-9
                for f_new in range(n_fac):
                    if f_new == current_f:
                        continue
                    if load[f_new] + demands[i] > capacities[f_new] + 1e-9:
                        continue
                    delta = assign_cost[i, f_new] - assign_cost[i, current_f]
                    if load[f_new] == 0:
                        delta += open_cost[f_new]
                    if load[current_f] == demands[i]:
                        delta -= open_cost[current_f]
                    if delta < best_delta:
                        best_delta = delta
                        best_f = f_new
                if best_f != -1:
                    load[current_f] -= demands[i]
                    load[best_f] += demands[i]
                    res[i] = best_f
                    improved = True
            # Swap moves: try swapping two customers
            if time.time() > deadline:
                break
            swap_pairs = rng.randint(0, n_cust, size=200)
            swap_pairs2 = rng.randint(0, n_cust, size=200)
            for idx in range(200):
                i1, i2 = swap_pairs[idx], swap_pairs2[idx]
                if i1 == i2:
                    continue
                f1, f2 = res[i1], res[i2]
                if f1 == f2:
                    continue
                # Check capacity after swap
                if load[f2] - demands[i2] + demands[i1] > capacities[f2] + 1e-9:
                    continue
                if load[f1] - demands[i1] + demands[i2] > capacities[f1] + 1e-9:
                    continue
                delta = (assign_cost[i1, f2] + assign_cost[i2, f1]) - \
                        (assign_cost[i1, f1] + assign_cost[i2, f2])
                # Open cost changes
                if load[f1] == demands[i1] and load[f2] - demands[i2] + demands[i1] > 0:
                    # f1 closes, f2 stays open (or was already open)
                    delta -= open_cost[f1]
                elif load[f1] - demands[i1] + demands[i2] == 0 and load[f2] - demands[i2] + demands[i1] > 0:
                    delta -= open_cost[f1]
                if load[f2] == demands[i2] and load[f1] - demands[i1] + demands[i2] > 0:
                    delta -= open_cost[f2]
                elif load[f2] - demands[i2] + demands[i1] == 0 and load[f1] - demands[i1] + demands[i2] > 0:
                    delta -= open_cost[f2]
                # If a facility was open and becomes empty
                if load[f1] - demands[i1] + demands[i2] == 0 and load[f1] - demands[i1] > 0:
                    # f1 loses i1 but gains i2, stays open
                    pass
                if load[f2] - demands[i2] + demands[i1] == 0 and load[f2] - demands[i2] > 0:
                    pass
                if delta < -1e-9:
                    load[f1] -= demands[i1]
                    load[f1] += demands[i2]
                    load[f2] -= demands[i2]
                    load[f2] += demands[i1]
                    res[i1] = f2
                    res[i2] = f1
                    improved = True

    def solve_milp(time_limit):
        """Solve MILP with tight capacity linking: sum_i d_i*x_ij <= cap_j * y_j."""
        n_x = n_cust * n_fac
        n_vars = n_x + n_fac

        c = np.zeros(n_vars)
        c[:n_x] = assign_cost.flatten()
        c[n_x:] = open_cost

        # Constraint 1: sum_j x[i,j] = 1 for each i
        A1 = np.zeros((n_cust, n_vars))
        for i in range(n_cust):
            A1[i, i * n_fac:(i + 1) * n_fac] = 1.0
        lb1 = np.ones(n_cust)
        ub1 = np.ones(n_cust)

        # Constraint 2: sum_i demands[i] * x[i,j] <= capacities[j] * y[j]
        # => sum_i demands[i] * x[i,j] - capacities[j] * y[j] <= 0
        A2 = np.zeros((n_fac, n_vars))
        for j in range(n_fac):
            for i in range(n_cust):
                A2[j, i * n_fac + j] = demands[i]
            A2[j, n_x + j] = -capacities[j]
        lb2 = np.full(n_fac, -np.inf)
        ub2 = np.zeros(n_fac)

        constraints = [
            LinearConstraint(A1, lb1, ub1),
            LinearConstraint(A2, lb2, ub2)
        ]

        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(n_vars)

        try:
            res = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                options={"time_limit": time_limit}
            )
            if res.status == 0 and res.x is not None:
                x_sol = res.x[:n_x].reshape(n_cust, n_fac)
                result = np.argmax(x_sol, axis=1).astype(int)
                load = np.zeros(n_fac)
                np.add.at(load, result, demands)
                if np.all(load <= capacities + 1e-9):
                    return result
        except Exception:
            pass
        return None

    def lagrangian_relaxation(max_iter=500, seed=0):
        """
        Lagrangian Relaxation for capacitated facility location.
        Dualizes capacity constraints: sum_i d_i * x[i,j] <= cap_j
        Multipliers lambda_j >= 0 for each facility j.
        
        At each iteration:
        1. Solve decomposed subproblem: for each customer i, choose j minimizing
           assign_cost[i,j] + lambda_j * d_i, subject to facility open cost.
        2. Update multipliers via subgradient method.
        3. Use the solution as a heuristic starting point.
        """
        rng = np.random.RandomState(seed)
        lam = np.zeros(n_fac)  # Lagrange multipliers
        best_solution = None
        best_score = np.inf
        
        # Track best feasible solution found during iterations
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Step 1: Solve the Lagrangian subproblem
            # For each customer i, the effective cost of assigning to facility j is:
            #   assign_cost[i,j] + lam[j] * demands[i]
            # We also need to account for open costs.
            # 
            # Approach: For each facility j, if we open it, we pay open_cost[j].
            # The Lagrangian relaxation gives us a lower bound.
            # For the heuristic, we assign each customer to the facility with
            # minimum (assign_cost[i,j] + lam[j] * demands[i]), then check feasibility.
            
            # Effective costs including Lagrange penalty
            eff_cost = assign_cost + np.outer(demands, lam)  # shape (n_cust, n_fac)
            
            # For each customer, find the best facility by effective cost
            prelim_assign = np.argmin(eff_cost, axis=1)
            
            # Now we need to handle open costs. The Lagrangian subproblem decomposes
            # but open costs couple facilities. A common approach:
            # - For each facility, determine if it's "worth" opening based on assigned customers
            # - Use a greedy/iterative approach to determine which facilities to open
            
            # Determine which facilities have customers assigned
            load = np.zeros(n_fac)
            np.add.at(load, prelim_assign, demands)
            
            # Check capacity feasibility
            feasible = np.all(load <= capacities + 1e-9)
            
            if feasible:
                # This is a feasible solution; compute its true score
                score = compute_score(prelim_assign)
                if score < best_score:
                    best_score = score
                    best_solution = prelim_assign.copy()
            
            # Step 2: Subgradient update
            # Subgradient for lambda_j = load_j - cap_j (violation of capacity constraint)
            subgrad = load - capacities
            subgrad = np.maximum(subgrad, 0)  # lambda must be non-negative
            
            # Compute step size using Polyak-type or diminishing step
            # Use diminishing step: t_k = alpha / (1 + k*beta)
            alpha = 10.0
            beta = 0.01
            step_size = alpha / (1.0 + it * beta)
            
            # Cap the step to avoid divergence
            max_step = 1.0
            step_size = min(step_size, max_step)
            
            # Update multipliers (project onto non-negative orthant)
            lam = np.maximum(lam + step_size * subgrad, 0)
            
            # Decay multipliers slightly to allow exploration
            if it > 50:
                lam = lam * 0.99
            
            # Periodically try to repair infeasible solutions
            if it % 10 == 0 and not feasible:
                # Try to repair by reassigning customers from overloaded facilities
                repaired = prelim_assign.copy()
                load_rep = np.zeros(n_fac)
                np.add.at(load_rep, repaired, demands)
                
                # Find overloaded facilities
                overloaded = np.where(load_rep > capacities + 1e-9)[0]
                
                for f in overloaded:
                    # Find customers assigned to f and reassign them
                    custs_at_f = np.where(repaired == f)[0]
                    for i in custs_at_f:
                        # Find the best feasible facility for customer i
                        best_f = -1
                        best_cost = np.inf
                        for f_new in range(n_fac):
                            if load_rep[f_new] + demands[i] <= capacities[f_new] + 1e-9:
                                cost = assign_cost[i, f_new]
                                if load_rep[f_new] == 0:
                                    cost += open_cost[f_new]
                                if cost < best_cost:
                                    best_cost = cost
                                    best_f = f_new
                        if best_f != -1:
                            load_rep[repaired[i]] -= demands[i]
                            load_rep[best_f] += demands[i]
                            repaired[i] = best_f
                
                # Check if repair was successful
                if np.all(load_rep <= capacities + 1e-9):
                    score_rep = compute_score(repaired)
                    if score_rep < best_score:
                        best_score = score_rep
                        best_solution = repaired.copy()
        
        return best_solution

    # Phase 1: Greedy initialization
    res_greedy = greedy_init(seed=42)
    load_greedy = np.zeros(n_fac)
    np.add.at(load_greedy, res_greedy, demands)
    local_search(res_greedy, load_greedy, max_iter=2000, seed=42)
    best_assign = res_greedy.copy()
    best_score = compute_score(best_assign)

    # Phase 2: MILP solve
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        milp_time = min(deadline - elapsed - 5, 100.0)
        res_milp = solve_milp(milp_time)
        if res_milp is not None:
            load_milp = np.zeros(n_fac)
            np.add.at(load_milp, res_milp, demands)
            score_milp = compute_score(res_milp)
            if score_milp < best_score:
                best_score = score_milp
                best_assign = res_milp.copy()

    # Phase 3: Lagrangian Relaxation
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        lag_time = min(deadline - elapsed - 5, 60.0)
        max_iter_lag = min(500, int(lag_time * 50))
        res_lag = lagrangian_relaxation(max_iter=max_iter_lag, seed=77)
        if res_lag is not None:
            load_lag = np.zeros(n_fac)
            np.add.at(load_lag, res_lag, demands)
            score_lag = compute_score(res_lag)
            if score_lag < best_score:
                best_score = score_lag
                best_assign = res_lag.copy()

    # Phase 4: Local search on best solution with more iterations
    load_best = np.zeros(n_fac)
    np.add.at(load_best, best_assign, demands)
    remaining = deadline - time.time() - 2
    if remaining > 1:
        max_iter = min(10000, int(remaining * 200))
        local_search(best_assign, load_best, max_iter=max_iter, seed=123)

    # Phase 5: Try a few more local search restarts from perturbed solutions
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        for restart_seed in [7, 99, 2024]:
            if time.time() > deadline - 3:
                break
            res_pert = best_assign.copy()
            load_pert = np.zeros(n_fac)
            np.add.at(load_pert, res_pert, demands)
            # Perturb: randomly relocate 10% of customers
            n_perturb = max(1, n_cust // 10)
            pert_idx = np.random.RandomState(restart_seed).choice(n_cust, size=n_perturb, replace=False)
            for i in pert_idx:
                old_f = res_pert[i]
                load_pert[old_f] -= demands[i]
                # Assign to random feasible facility
                feasible = np.where(load_pert + demands[i] <= capacities + 1e-9)[0]
                if len(feasible) > 0:
                    new_f = feasible[np.random.RandomState(restart_seed + i).randint(len(feasible))]
                    res_pert[i] = new_f
                    load_pert[new_f] += demands[i]
            local_search(res_pert, load_pert, max_iter=1000, seed=restart_seed)
            score_pert = compute_score(res_pert)
            if score_pert < best_score:
                best_score = score_pert
                best_assign = res_pert.copy()

    # Final verification
    load_verify = np.zeros(n_fac)
    np.add.at(load_verify, best_assign, demands)
    if np.any(load_verify > capacities + 1e-9):
        best_assign = res_greedy

    return best_assign.astype(int)

# EVOLVE-BLOCK-END

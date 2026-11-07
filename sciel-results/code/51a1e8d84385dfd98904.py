"""
Facility location solver using MIP (scipy.optimize.milp) with heuristic fallback.
Strategy: Formulate as an integer linear program and solve with HiGHS via scipy.
If MIP is unavailable or times out, fall back to greedy + local search.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility.
    Primary: MIP via scipy.optimize.milp.
    Fallback: greedy + local search.
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
        """Compute total cost for an assignment, return inf if infeasible."""
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

    def greedy_assignment(order=None):
        """Greedy: assign in given order to cheapest feasible facility."""
        if order is None:
            order = np.arange(n_customers)
        result = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        for c in order:
            costs = assign_cost[c].copy()
            feasible = load + demands[c] <= capacities
            costs[~feasible] = np.inf
            f = int(np.argmin(costs))
            if feasible[f]:
                result[c] = f
                load[f] += demands[c]
            else:
                return None
        return result

    def local_search(assignment, max_iter=200):
        """Improve assignment via single reassignments."""
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            load[int(assignment[c])] += demands[c]
        current_cost = compute_cost(assignment)
        if current_cost == np.inf:
            return assignment
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            for c in range(n_customers):
                old_f = int(assignment[c])
                load[old_f] -= demands[c]
                best_new_f = old_f
                best_delta = 0.0
                for new_f in range(n_facilities):
                    if new_f == old_f:
                        continue
                    if load[new_f] + demands[c] > capacities[new_f] + 1e-9:
                        continue
                    delta = assign_cost[c, new_f] - assign_cost[c, old_f]
                    if load[new_f] == 0:
                        delta += open_cost[new_f]
                    if load[old_f] == 0:
                        delta -= open_cost[old_f]
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_new_f = new_f
                if best_new_f != old_f and best_delta < -1e-9:
                    assignment[c] = best_new_f
                    load[best_new_f] += demands[c]
                    current_cost += best_delta
                    improved = True
                else:
                    load[old_f] += demands[c]
            if not improved:
                break
        return assignment

    # Try MIP first
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix

        # Variables:
        # x[c, f] = 1 if customer c assigned to facility f (n_customers * n_facilities)
        # y[f] = 1 if facility f is open (n_facilities)
        # Total vars: n_customers * n_facilities + n_facilities

        n_vars = n_customers * n_facilities + n_facilities
        x_vars = n_customers * n_facilities
        y_vars = n_facilities

        # Objective: minimize sum_c,f assign_cost[c,f]*x[c,f] + sum_f open_cost[f]*y[f]
        c_obj = np.zeros(n_vars)
        for c in range(n_customers):
            for f in range(n_facilities):
                c_obj[c * n_facilities + f] = assign_cost[c, f]
        for f in range(n_facilities):
            c_obj[x_vars + f] = open_cost[f]

        # Constraints:
        # 1. Each customer assigned to exactly one facility:
        #    sum_f x[c,f] = 1 for each c
        # 2. Capacity: sum_c demands[c]*x[c,f] <= capacities[f]*y[f] for each f
        #    => sum_c demands[c]*x[c,f] - capacities[f]*y[f] <= 0
        # 3. y[f] >= x[c,f] for each c,f (if customer assigned to f, f must be open)
        #    => x[c,f] - y[f] <= 0

        # Build constraint matrix
        # Constraint 1: n_customers rows, each has 1s in the n_facilities positions for that customer
        A1 = np.zeros((n_customers, n_vars))
        for c in range(n_customers):
            A1[c, c * n_facilities:(c + 1) * n_facilities] = 1.0

        # Constraint 2: n_facilities rows
        A2 = np.zeros((n_facilities, n_vars))
        for f in range(n_facilities):
            for c in range(n_customers):
                A2[f, c * n_facilities + f] = demands[c]
            A2[f, x_vars + f] = -capacities[f]

        # Constraint 3: n_customers * n_facilities rows
        A3 = np.zeros((n_customers * n_facilities, n_vars))
        for c in range(n_customers):
            for f in range(n_facilities):
                idx = c * n_facilities + f
                A3[idx, c * n_facilities + f] = 1.0
                A3[idx, x_vars + f] = -1.0

        # Combine constraints
        # Use LinearConstraint with separate bounds for equality and inequality
        # scipy milp supports mixed constraints via bounds

        # Let's use a single sparse matrix approach
        # Equality constraints (type 1): n_customers
        # Inequality constraints (type 2): n_facilities (capacity)
        # Inequality constraints (type 3): n_customers * n_facilities (openness)

        n_cons = n_customers + n_facilities + n_customers * n_facilities
        A = np.zeros((n_cons, n_vars))
        lb = np.zeros(n_cons)
        ub = np.zeros(n_cons)

        # Type 1: equality
        for c in range(n_customers):
            A[c, c * n_facilities:(c + 1) * n_facilities] = 1.0
            lb[c] = 1.0
            ub[c] = 1.0

        # Type 2: capacity (<= 0)
        for f in range(n_facilities):
            row = n_customers + f
            for c in range(n_customers):
                A[row, c * n_facilities + f] = demands[c]
            A[row, x_vars + f] = -capacities[f]
            lb[row] = -np.inf
            ub[row] = 0.0

        # Type 3: openness (x[c,f] - y[f] <= 0)
        for c in range(n_customers):
            for f in range(n_facilities):
                row = n_customers + n_facilities + c * n_facilities + f
                A[row, c * n_facilities + f] = 1.0
                A[row, x_vars + f] = -1.0
                lb[row] = -np.inf
                ub[row] = 0.0

        constraints = LinearConstraint(A, lb, ub)

        # Variable bounds: all in [0, 1]
        var_bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))

        # Integrality: all binary
        integrality = np.ones(n_vars)

        result = milp(
            c=c_obj,
            constraints=constraints,
            integrality=integrality,
            bounds=var_bounds,
            options={'time_limit': 100}
        )

        if result.status == 0 and result.x is not None:
            assignment = np.zeros(n_customers, dtype=int)
            for c in range(n_customers):
                for f in range(n_facilities):
                    if result.x[c * n_facilities + f] > 0.5:
                        assignment[c] = f
                        break
            # Verify
            cost = compute_cost(assignment)
            if cost < np.inf:
                return assignment.astype(int)

    except Exception as e:
        pass

    # Fallback: greedy + local search
    best_assignment = None
    best_cost = np.inf

    # Multiple greedy orderings
    orderings = [
        np.arange(n_customers),
        np.argsort(-demands),
        np.argsort(demands),
        np.argsort(assign_cost.sum(axis=1)),
    ]

    for order in orderings:
        if time.time() > deadline - 10:
            break
        sol = greedy_assignment(order)
        if sol is not None:
            sol = local_search(sol)
            cost = compute_cost(sol)
            if cost < best_cost:
                best_cost = cost
                best_assignment = sol.copy()

    # Randomized greedy
    rng = np.random.RandomState(42)
    for i in range(20):
        if time.time() > deadline - 5:
            break
        order = rng.permutation(n_customers)
        sol = greedy_assignment(order)
        if sol is not None:
            sol = local_search(sol)
            cost = compute_cost(sol)
            if cost < best_cost:
                best_cost = cost
                best_assignment = sol.copy()

    if best_assignment is None:
        # Last resort fallback
        best_assignment = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        for c in range(n_customers):
            f = int(np.argmax(capacities - load))
            best_assignment[c] = f
            load[f] += demands[c]

    return best_assignment.astype(int)
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility using a min-cost flow formulation.
    We model this as a transportation problem with facility opening costs.
    
    Approach:
    1. Use a min-cost flow to find optimal assignment given a set of open facilities.
    2. Use local search to determine which facilities to open/close.
    
    The problem is:
    min sum_f open_cost[f] * y[f] + sum_{i,f} assign_cost[i,f] * x[i,f]
    s.t. sum_f x[i,f] = 1 for all i (each customer assigned to one facility)
         sum_i demands[i] * x[i,f] <= capacities[f] * y[f] for all f
         y[f] in {0,1}
         x[i,f] in {0,1}
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Helper: solve min-cost flow for a given set of open facilities
    # Returns assignment array and total assignment cost (excluding opening costs)
    def solve_min_cost_flow(open_mask):
        """
        Solve min-cost flow where only facilities with open_mask[f]=True can be used.
        Uses successive shortest path with potentials.
        """
        # Build cost matrix for min-cost flow
        # We need to assign each customer to a facility, respecting capacities
        
        # Use linear programming via scipy
        from scipy.optimize import linprog
        
        # Decision variables: x[i,f] for each customer i and facility f
        # Only facilities in open_mask are allowed
        allowed_facilities = [f for f in range(n_facilities) if open_mask[f]]
        n_allowed = len(allowed_facilities)
        
        if n_allowed == 0:
            return None, np.inf
        
        # Create cost vector
        cost_vec = []
        for i in range(n_customers):
            for f in allowed_facilities:
                cost_vec.append(assign_cost[i, f])
        cost_vec = np.array(cost_vec)
        
        # Equality constraints: each customer assigned to exactly one facility
        # A_eq shape: (n_customers, n_customers * n_allowed)
        A_eq = np.zeros((n_customers, n_customers * n_allowed))
        for i in range(n_customers):
            for j, f in enumerate(allowed_facilities):
                A_eq[i, i * n_allowed + j] = 1.0
        
        # Inequality constraints: capacity constraints for each facility
        # sum_i demands[i] * x[i,f] <= capacities[f]
        A_ub = []
        b_ub = []
        for f_idx, f in enumerate(allowed_facilities):
            row = np.zeros(n_customers * n_allowed)
            for i in range(n_customers):
                row[i * n_allowed + f_idx] = demands[i]
            A_ub.append(row)
            b_ub.append(capacities[f])
        
        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None
        
        # Bounds: 0 <= x[i,f] <= 1
        bounds = [(0, 1)] * (n_customers * n_allowed)
        
        # Solve LP relaxation
        res = linprog(cost_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.ones(n_customers),
                      bounds=bounds, method='highs')
        
        if not res.success:
            return None, np.inf
        
        # Extract solution
        x_sol = res.x.reshape(n_customers, n_allowed)
        
        # Round to integer solution (for transportation problem with integer data, LP is integral)
        # But due to numerical issues, we might need to round
        assignment = np.zeros(n_customers, dtype=int)
        for i in range(n_customers):
            # Find the facility with highest x value
            best_j = np.argmax(x_sol[i])
            assignment[i] = allowed_facilities[best_j]
        
        # Verify capacity constraints
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
        
        for f in allowed_facilities:
            if load[f] > capacities[f] + 1e-6:
                # Infeasible - should not happen with correct LP
                return None, np.inf
        
        total_assign_cost = sum(assign_cost[i, assignment[i]] for i in range(n_customers))
        return assignment, total_assign_cost
    
    # Initial solution: use all facilities
    open_mask = np.ones(n_facilities, dtype=bool)
    result, assign_cost_val = solve_min_cost_flow(open_mask)
    
    if result is None:
        # Fallback to greedy
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        order = np.argsort(-demands)
        for i in order:
            best_f = -1
            best_cost = np.inf
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    cost = assign_cost[i, f]
                    if cost < best_cost:
                        best_cost = cost
                        best_f = f
            if best_f >= 0:
                load[best_f] += demands[i]
                result[i] = best_f
            else:
                result[i] = 0
        assign_cost_val = sum(assign_cost[i, result[i]] for i in range(n_customers))
    
    def total_cost_with_opening(assignment):
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
        open_cost_val = sum(open_cost[f] for f in range(n_facilities) if load[f] > 0)
        assign_val = sum(assign_cost[i, assignment[i]] for i in range(n_customers))
        return open_cost_val + assign_val
    
    current_cost = total_cost_with_opening(result)
    
    # Local search: try closing facilities
    for f in range(n_facilities):
        # Try closing facility f
        new_open_mask = open_mask.copy()
        new_open_mask[f] = False
        
        new_result, new_assign_cost = solve_min_cost_flow(new_open_mask)
        if new_result is None:
            continue
        
        new_total = total_cost_with_opening(new_result)
        if new_total < current_cost - 1e-9:
            result = new_result
            open_mask = new_open_mask
            current_cost = new_total
    
    # Local search: try swapping assignments
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[result[i]] += demands[i]
    
    improved = True
    max_iterations = 30
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(n_customers):
            for j in range(i + 1, n_customers):
                fi = result[i]
                fj = result[j]
                if fi == fj:
                    continue
                # Try swapping: i -> fj, j -> fi
                if load[fi] - demands[i] + demands[j] <= capacities[fi] and \
                   load[fj] - demands[j] + demands[i] <= capacities[fj]:
                    new_cost = current_cost
                    new_cost -= assign_cost[i, fi] + assign_cost[j, fj]
                    new_cost += assign_cost[i, fj] + assign_cost[j, fi]
                    
                    old_load_fi = load[fi]
                    old_load_fj = load[fj]
                    new_load_fi = old_load_fi - demands[i] + demands[j]
                    new_load_fj = old_load_fj - demands[j] + demands[i]
                    
                    if old_load_fi > 0 and new_load_fi == 0:
                        new_cost -= open_cost[fi]
                    elif old_load_fi == 0 and new_load_fi > 0:
                        new_cost += open_cost[fi]
                    if old_load_fj > 0 and new_load_fj == 0:
                        new_cost -= open_cost[fj]
                    elif old_load_fj == 0 and new_load_fj > 0:
                        new_cost += open_cost[fj]
                    
                    if new_cost < current_cost - 1e-9:
                        result[i] = fj
                        result[j] = fi
                        load[fi] = new_load_fi
                        load[fj] = new_load_fj
                        current_cost = new_cost
                        improved = True
    
    return result
# EVOLVE-BLOCK-END

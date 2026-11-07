import numpy as np
import time
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # 1. Greedy Initialization
    # Sort customers by demand descending to place large demands first
    order = np.argsort(-demands)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    for i in order:
        # Find feasible facilities
        feasible = np.where(load + demands[i] <= capacities + 1e-9)[0]
        if len(feasible) == 0:
            # If no feasible, pick the one with most remaining capacity (or just first)
            # But this would be invalid. We assume instances are feasible.
            # If truly stuck, pick min cost among all, but risk invalidity.
            # Better: pick min cost among those with max remaining capacity?
            # Let's just pick the one with min assign_cost among all, hoping it fits?
            # No, that violates capacity.
            # If we can't fit, we have a problem.
            # Let's assume feasibility. If not, fallback to random?
            # For robustness, pick the facility with max remaining capacity.
            max_cap_idx = np.argmax(capacities - load)
            result[i] = max_cap_idx
            load[max_cap_idx] += demands[i]
        else:
            # Pick min assign_cost among feasible
            costs = assign_cost[i, feasible]
            best_idx = np.argmin(costs)
            f = feasible[best_idx]
            result[i] = f
            load[f] += demands[i]

    # 2. LP Relaxation with Fixed Open Costs
    # We fix the set of open facilities to those used in the greedy solution.
    # This makes the problem a standard transportation problem (min-cost flow).
    # The constraint matrix is totally unimodular, so the LP solution is integer.
    
    open_mask = np.zeros(n_facilities, dtype=bool)
    open_mask[result] = True
    
    # If we want to potentially open/close facilities, we could iterate, 
    # but for now, let's solve the assignment for the current open set.
    
    # Variables: x[i, j] for i in range(n_customers), j in range(n_facilities)
    # Only allow j where open_mask[j] is True.
    # To simplify linprog, we can just set cost to infinity for closed facilities.
    
    c = assign_cost.copy()
    # Set cost to large value for closed facilities
    closed_mask = ~open_mask
    c[:, closed_mask] = 1e10
    
    # Constraints:
    # 1. Sum_j x[i, j] = 1 for each i (each customer assigned to exactly one facility)
    # 2. Sum_i demands[i] * x[i, j] <= capacities[j] for each j (capacity)
    
    # Build A_ub and b_ub
    # We need to handle the equality constraints for assignment.
    # linprog standard form: A_ub @ x <= b_ub, A_eq @ x == b_eq
    
    # Equality constraints:
    # For each customer i: sum_j x[i,j] = 1
    A_eq = np.zeros((n_customers, n_customers * n_facilities))
    for i in range(n_customers):
        for j in range(n_facilities):
            A_eq[i, i * n_facilities + j] = 1.0
    b_eq = np.ones(n_customers)
    
    # Inequality constraints:
    # For each facility j: sum_i demands[i] * x[i,j] <= capacities[j]
    A_ub = np.zeros((n_facilities, n_customers * n_facilities))
    for j in range(n_facilities):
        for i in range(n_customers):
            A_ub[j, i * n_facilities + j] = demands[i]
    b_ub = capacities
    
    # Bounds: 0 <= x[i,j] <= 1
    bounds = [(0, 1)] * (n_customers * n_facilities)
    
    # Solve LP
    try:
        res = linprog(c.reshape(-1), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.status == 0:
            x = res.x.reshape(n_customers, n_facilities)
            # Extract assignment
            # Due to total unimodularity, x should be integer (0 or 1)
            # But due to floating point, we round.
            new_result = np.argmax(x, axis=1)
            
            # Verify feasibility
            new_load = np.zeros(n_facilities)
            for i in range(n_customers):
                new_load[new_result[i]] += demands[i]
            
            if np.all(new_load <= capacities + 1e-6):
                result = new_result
            else:
                # If infeasible due to numerical issues, stick with greedy
                pass
    except Exception:
        pass

    # 3. Local Search Polish
    # Single-move local search: try moving each customer to a different facility.
    # If it improves cost and maintains feasibility, accept.
    
    def compute_cost(assignment):
        cost = 0.0
        for i in range(n_customers):
            cost += assign_cost[i, assignment[i]]
        # Add open costs for facilities with at least one customer
        used = set(assignment)
        for j in used:
            cost += open_cost[j]
        return cost

    current_cost = compute_cost(result)
    load = np.zeros(n_facilities)
    counts = np.zeros(n_facilities)
    for i in range(n_customers):
        load[result[i]] += demands[i]
        counts[result[i]] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_customers):
            if time.time() > deadline:
                break
            
            cur_f = result[i]
            d_i = demands[i]
            
            # Try moving to each other facility
            for f in range(n_facilities):
                if f == cur_f:
                    continue
                
                # Check capacity
                if load[f] + d_i > capacities[f] + 1e-9:
                    continue
                
                # Calculate cost delta
                delta_assign = assign_cost[i, f] - assign_cost[i, cur_f]
                
                # Open cost delta
                delta_open = 0.0
                # If cur_f becomes empty, we save open_cost[cur_f]
                if counts[cur_f] == 1:
                    delta_open -= open_cost[cur_f]
                # If f was empty, we pay open_cost[f]
                if counts[f] == 0:
                    delta_open += open_cost[f]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < -1e-6:
                    # Accept move
                    result[i] = f
                    load[cur_f] -= d_i
                    load[f] += d_i
                    counts[cur_f] -= 1
                    counts[f] += 1
                    current_cost += total_delta
                    improved = True
                    break # Restart loop? Or continue? 
                    # Breaking the inner loop and continuing the outer is standard.
                    # But we might want to continue checking other customers immediately.
                    # Let's just break the inner loop.

    # Final verification
    final_load = np.zeros(n_facilities)
    for i in range(n_customers):
        final_load[result[i]] += demands[i]
    
    if np.any(final_load > capacities + 1e-6):
        # Fallback to greedy if something went wrong
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        for i in np.argsort(-demands):
            feasible = np.where(load + demands[i] <= capacities + 1e-9)[0]
            if len(feasible) == 0:
                max_cap_idx = np.argmax(capacities - load)
                result[i] = max_cap_idx
                load[max_cap_idx] += demands[i]
            else:
                costs = assign_cost[i, feasible]
                best_idx = np.argmin(costs)
                f = feasible[best_idx]
                result[i] = f
                load[f] += demands[i]

    return result.astype(int)

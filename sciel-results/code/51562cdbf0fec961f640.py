"""Improved capacitated facility location solver."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0  # seconds before deadline to stop
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Convert to numpy arrays if needed
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    # Check if any customer can't be assigned (capacity too small)
    # Fallback: greedy assignment if LP fails or times out
    
    def greedy_assignment():
        """Greedy: largest demand first, cheapest feasible facility."""
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        # Sort customers by demand descending
        order = np.argsort(-demands)
        for i in order:
            # Find cheapest facility with enough remaining capacity
            feasible = load + demands[i] <= capacities
            if not np.any(feasible):
                # Fallback: assign to facility with most remaining capacity
                best_f = np.argmax(capacities - load)
                result[i] = best_f
                load[best_f] += demands[i]
            else:
                # Among feasible, pick cheapest
                costs = np.where(feasible, assign_cost[i], np.inf)
                best_f = np.argmin(costs)
                result[i] = best_f
                load[best_f] += demands[i]
        return result
    
    # First, get a feasible solution via greedy
    best_solution = greedy_assignment()
    best_cost = compute_total_cost(best_solution, demands, capacities, open_cost, assign_cost)
    
    # Try LP relaxation to get a better starting point
    try:
        lp_solution = solve_lp(demands, capacities, open_cost, assign_cost)
        if lp_solution is not None:
            # Round the LP solution
            rounded = round_lp_solution(lp_solution, demands, capacities, open_cost, assign_cost, best_solution)
            if rounded is not None:
                rounded_cost = compute_total_cost(rounded, demands, capacities, open_cost, assign_cost)
                if rounded_cost < best_cost:
                    best_solution = rounded
                    best_cost = rounded_cost
    except Exception:
        pass
    
    # Local search: 1-swap and 2-swap improvements
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        best_solution, best_cost = local_search_1swap(
            best_solution, demands, capacities, open_cost, assign_cost, best_cost
        )
    
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        best_solution, best_cost = local_search_2swap(
            best_solution, demands, capacities, open_cost, assign_cost, best_cost
        )
    
    # Final polish: more aggressive 1-swap with multiple passes
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
        best_solution, best_cost = local_search_1swap_polish(
            best_solution, demands, capacities, open_cost, assign_cost, best_cost
        )
    
    return best_solution.astype(int)


def compute_total_cost(solution, demands, capacities, open_cost, assign_cost):
    """Compute total cost for a given assignment."""
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Check validity
    for i in range(n_customers):
        f = int(solution[i])
        if f < 0 or f >= n_facilities:
            return np.inf
    
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[int(solution[i])] += demands[i]
    
    for f in range(n_facilities):
        if load[f] > capacities[f] + 1e-9:
            return np.inf
    
    # Compute cost
    assignment_cost = sum(assign_cost[i, int(solution[i])] for i in range(n_customers))
    open_facilities = set(int(solution[i]) for i in range(n_customers))
    opening_cost = sum(open_cost[f] for f in open_facilities)
    
    return assignment_cost + opening_cost


def solve_lp(demands, capacities, open_cost, assign_cost):
    """
    Solve LP relaxation of the facility location problem.
    Variables: x[i,f] = fraction of customer i assigned to facility f
               y[f] = whether facility f is open (0 to 1)
    
    Minimize: sum_{i,f} assign_cost[i,f]*x[i,f] + sum_f open_cost[f]*y[f]
    Subject to:
        sum_f x[i,f] = 1 for all i
        sum_i demands[i]*x[i,f] <= capacities[f]*y[f] for all f
        0 <= x[i,f] <= 1
        0 <= y[f] <= 1
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Number of variables: n_customers * n_facilities (x) + n_facilities (y)
    n_x = n_customers * n_facilities
    n_y = n_facilities
    n_vars = n_x + n_y
    
    # Objective
    c = np.zeros(n_vars)
    for i in range(n_customers):
        for f in range(n_facilities):
            c[i * n_facilities + f] = assign_cost[i, f]
    for f in range(n_facilities):
        c[n_x + f] = open_cost[f]
    
    # Equality constraints: sum_f x[i,f] = 1 for each i
    A_eq = np.zeros((n_customers, n_vars))
    b_eq = np.ones(n_customers)
    for i in range(n_customers):
        for f in range(n_facilities):
            A_eq[i, i * n_facilities + f] = 1.0
    
    # Inequality constraints: sum_i demands[i]*x[i,f] - capacities[f]*y[f] <= 0
    A_ub = np.zeros((n_facilities, n_vars))
    b_ub = np.zeros(n_facilities)
    for f in range(n_facilities):
        for i in range(n_customers):
            A_ub[f, i * n_facilities + f] = demands[i]
        A_ub[f, n_x + f] = -capacities[f]
    
    # Bounds: 0 <= x <= 1, 0 <= y <= 1
    bounds = [(0, 1)] * n_x + [(0, 1)] * n_y
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if result.status != 0:
        return None
    
    x = result.x[:n_x].reshape(n_customers, n_facilities)
    y = result.x[n_x:]
    
    return x, y


def round_lp_solution(lp_solution, demands, capacities, open_cost, assign_cost, fallback):
    """
    Round the LP solution to a feasible integer solution.
    Uses a deterministic rounding strategy.
    """
    x, y = lp_solution
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Determine which facilities to open based on y values
    # Open facilities with y > threshold, ensure capacity feasibility
    # Use a greedy approach: sort by y descending, open until all customers can be served
    
    # First, assign each customer to their best facility based on x values
    # Then adjust for capacity constraints
    
    # Initialize with the facility that has highest x value for each customer
    assignment = np.zeros(n_customers, dtype=int)
    for i in range(n_customers):
        assignment[i] = np.argmax(x[i])
    
    # Check capacity constraints and fix violations
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[assignment[i]] += demands[i]
    
    # Fix capacity violations
    for f in range(n_facilities):
        if load[f] > capacities[f]:
            # Move customers from this facility to others
            # Sort customers at this facility by how much they prefer this facility
            customers_at_f = [i for i in range(n_customers) if assignment[i] == f]
            # Sort by x[i,f] ascending (least preferred first)
            customers_at_f.sort(key=lambda i: x[i, f])
            
            for i in customers_at_f:
                if load[f] <= capacities[f]:
                    break
                # Find best alternative facility
                best_alt = -1
                best_cost = np.inf
                for alt_f in range(n_facilities):
                    if alt_f == f:
                        continue
                    if load[alt_f] + demands[i] <= capacities[alt_f]:
                        # Cost of moving: change in assignment cost + potential change in open cost
                        # Simplified: just use assignment cost
                        if assign_cost[i, alt_f] < best_cost:
                            best_cost = assign_cost[i, alt_f]
                            best_alt = alt_f
                if best_alt >= 0:
                    load[f] -= demands[i]
                    load[best_alt] += demands[i]
                    assignment[i] = best_alt
                else:
                    # Can't fix, return fallback
                    return fallback
    
    # Verify feasibility
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[assignment[i]] += demands[i]
    for f in range(n_facilities):
        if load[f] > capacities[f] + 1e-9:
            return fallback
    
    return assignment


def local_search_1swap(solution, demands, capacities, open_cost, assign_cost, current_cost):
    """
    1-swap local search: try moving each customer to a different facility.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0
    
    solution = solution.copy()
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[solution[i]] += demands[i]
    
    improved = True
    while improved:
        improved = False
        for i in range(n_customers):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            current_f = solution[i]
            # Try moving to other facilities
            for new_f in range(n_facilities):
                if new_f == current_f:
                    continue
                if load[new_f] + demands[i] > capacities[new_f]:
                    continue
                
                # Compute cost change
                # Remove from current_f, add to new_f
                cost_change = assign_cost[i, new_f] - assign_cost[i, current_f]
                
                # Check if this changes open cost
                # If current_f becomes empty, we save open_cost[current_f]
                # If new_f was empty, we pay open_cost[new_f]
                if load[current_f] == demands[i]:
                    cost_change -= open_cost[current_f]
                if load[new_f] == 0:
                    cost_change += open_cost[new_f]
                
                if cost_change < -1e-9:
                    # Make the move
                    load[current_f] -= demands[i]
                    load[new_f] += demands[i]
                    solution[i] = new_f
                    current_cost += cost_change
                    improved = True
                    break
    
    return solution, current_cost


def local_search_2swap(solution, demands, capacities, open_cost, assign_cost, current_cost):
    """
    2-swap local search: try swapping two customers' facilities.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0
    
    solution = solution.copy()
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[solution[i]] += demands[i]
    
    improved = True
    while improved:
        improved = False
        for i in range(n_customers):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            for j in range(i + 1, n_customers):
                if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                    break
                fi = solution[i]
                fj = solution[j]
                if fi == fj:
                    continue
                
                # Try swapping: i goes to fj, j goes to fi
                # Check capacity
                if load[fi] - demands[i] + demands[j] > capacities[fi]:
                    continue
                if load[fj] - demands[j] + demands[i] > capacities[fj]:
                    continue
                
                # Compute cost change
                cost_change = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
                
                # Open cost changes
                # fi: load changes from load[fi] to load[fi] - demands[i] + demands[j]
                # fj: load changes from load[fj] to load[fj] - demands[j] + demands[i]
                old_load_fi = load[fi]
                old_load_fj = load[fj]
                new_load_fi = old_load_fi - demands[i] + demands[j]
                new_load_fj = old_load_fj - demands[j] + demands[i]
                
                # Open cost change for fi
                if old_load_fi == 0 and new_load_fi > 0:
                    cost_change += open_cost[fi]
                elif old_load_fi > 0 and new_load_fi == 0:
                    cost_change -= open_cost[fi]
                
                # Open cost change for fj
                if old_load_fj == 0 and new_load_fj > 0:
                    cost_change += open_cost[fj]
                elif old_load_fj > 0 and new_load_fj == 0:
                    cost_change -= open_cost[fj]
                
                if cost_change < -1e-9:
                    # Make the swap
                    load[fi] = new_load_fi
                    load[fj] = new_load_fj
                    solution[i] = fj
                    solution[j] = fi
                    current_cost += cost_change
                    improved = True
                    break
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
    
    return solution, current_cost


def local_search_1swap_polish(solution, demands, capacities, open_cost, assign_cost, current_cost):
    """
    More aggressive 1-swap with multiple passes and tighter tolerance.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 3.0  # tighter margin for polish
    
    solution = solution.copy()
    load = np.zeros(n_facilities)
    for i in range(n_customers):
        load[solution[i]] += demands[i]
    
    # Multiple passes
    for _ in range(5):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        improved = True
        while improved:
            improved = False
            for i in range(n_customers):
                if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                    break
                current_f = solution[i]
                for new_f in range(n_facilities):
                    if new_f == current_f:
                        continue
                    if load[new_f] + demands[i] > capacities[new_f]:
                        continue
                    
                    cost_change = assign_cost[i, new_f] - assign_cost[i, current_f]
                    
                    if load[current_f] == demands[i]:
                        cost_change -= open_cost[current_f]
                    if load[new_f] == 0:
                        cost_change += open_cost[new_f]
                    
                    if cost_change < -1e-9:
                        load[current_f] -= demands[i]
                        load[new_f] += demands[i]
                        solution[i] = new_f
                        current_cost += cost_change
                        improved = True
                        break
    
    return solution, current_cost
# EVOLVE-BLOCK-END

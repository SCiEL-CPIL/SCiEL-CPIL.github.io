import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # 1. Greedy Initialization
    # Sort customers by demand descending to place large demands first
    order = np.argsort(-demands)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    for i in order:
        # Find feasible facilities
        feasible = np.where(load + demands[i] <= capacities + 1e-9)[0]
        if len(feasible) == 0:
            # If no feasible, pick the one with most remaining capacity
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

    # 2. Local Search Polish
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

    # 3. Pairwise Swap Local Search
    # Try swapping two customers assigned to different facilities.
    # This can unlock improvements that single moves cannot.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_customers):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_customers):
                if time.time() > deadline:
                    break
                
                f_i = result[i]
                f_j = result[j]
                
                if f_i == f_j:
                    continue
                
                d_i = demands[i]
                d_j = demands[j]
                
                # Check capacity after swap
                if load[f_i] - d_i + d_j > capacities[f_i] + 1e-9:
                    continue
                if load[f_j] - d_j + d_i > capacities[f_j] + 1e-9:
                    continue
                
                # Calculate cost delta
                delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                
                # Open cost delta
                delta_open = 0.0
                # If f_i becomes empty after removing i (and before adding j), but we add j so it won't be empty if j was elsewhere
                # Actually, if f_i had only i, and we move i to f_j, f_i becomes empty.
                # If f_j had only j, and we move j to f_i, f_j becomes empty.
                # But we are swapping, so:
                # f_i loses i, gains j.
                # f_j loses j, gains i.
                
                # Count before swap
                count_f_i = counts[f_i]
                count_f_j = counts[f_j]
                
                # After swap:
                # f_i count is still count_f_i (lost i, gained j)
                # f_j count is still count_f_j (lost j, gained i)
                # So open costs don't change unless a facility becomes empty, which can't happen with a swap between two non-empty facilities.
                # Wait, if f_i == f_j, we skip. So f_i != f_j.
                # If f_i had only i, count_f_i = 1. After swap, f_i has j, so count is 1. Not empty.
                # If f_j had only j, count_f_j = 1. After swap, f_j has i, so count is 1. Not empty.
                # So delta_open is always 0 for swaps.
                
                total_delta = delta_assign + delta_open
                
                if total_delta < -1e-6:
                    # Accept swap
                    result[i] = f_j
                    result[j] = f_i
                    load[f_i] = load[f_i] - d_i + d_j
                    load[f_j] = load[f_j] - d_j + d_i
                    # counts don't change
                    current_cost += total_delta
                    improved = True
                    break
            if improved:
                break # Restart outer loop

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

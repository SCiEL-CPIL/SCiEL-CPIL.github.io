"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Initial greedy assignment: largest demand first, cheapest feasible facility
    result = np.full(n_customers, -1, dtype=int)
    load = np.zeros(n_facilities, dtype=float)
    
    # Sort customers by demand descending
    order = np.argsort(-demands)
    
    for i in order:
        # Find cheapest feasible facility
        best_f = -1
        best_cost = float('inf')
        for f in range(n_facilities):
            if load[f] + demands[i] <= capacities[f]:
                cost = assign_cost[i, f]
                if cost < best_cost:
                    best_cost = cost
                    best_f = f
        if best_f >= 0:
            result[i] = best_f
            load[best_f] += demands[i]
    
    # If any customer unassigned, assign to any feasible facility (shouldn't happen)
    for i in range(n_customers):
        if result[i] < 0:
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    result[i] = f
                    load[f] += demands[i]
                    break
    
    # Compute total cost
    def total_cost(assignment):
        open_mask = np.zeros(n_facilities, dtype=bool)
        for f in assignment:
            if f >= 0:
                open_mask[f] = True
        open_cost_sum = np.sum(open_cost[open_mask])
        assign_cost_sum = np.sum(assign_cost[np.arange(n_customers), assignment])
        return open_cost_sum + assign_cost_sum
    
    best_assignment = result.copy()
    best_score = total_cost(best_assignment)
    
    # Local search: try moving customers to better facilities
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle order for diversity
        search_order = np.random.permutation(n_customers)
        
        for i in search_order:
            if time.time() > deadline:
                break
            current_f = best_assignment[i]
            if current_f < 0:
                continue
            
            # Try moving to each other facility
            for new_f in range(n_facilities):
                if new_f == current_f:
                    continue
                if load[new_f] + demands[i] > capacities[new_f]:
                    continue
                
                # Check if this move improves total cost
                # Remove from current, add to new
                old_open_mask = np.zeros(n_facilities, dtype=bool)
                for f in best_assignment:
                    if f >= 0:
                        old_open_mask[f] = True
                
                # Simulate move
                load_sim = load.copy()
                load_sim[current_f] -= demands[i]
                load_sim[new_f] += demands[i]
                
                new_open_mask = old_open_mask.copy()
                # Check if current_f becomes empty
                remaining_at_current = np.sum(best_assignment == current_f) - 1
                if remaining_at_current == 0:
                    new_open_mask[current_f] = False
                # Check if new_f was empty
                if load_sim[new_f] == demands[i]:
                    new_open_mask[new_f] = True
                
                old_open_cost = np.sum(open_cost[old_open_mask])
                new_open_cost = np.sum(open_cost[new_open_mask])
                
                old_assign_cost = assign_cost[i, current_f]
                new_assign_cost = assign_cost[i, new_f]
                
                delta = (new_open_cost - old_open_cost) + (new_assign_cost - old_assign_cost)
                
                if delta < -1e-9:
                    # Apply move
                    load[current_f] -= demands[i]
                    load[new_f] += demands[i]
                    best_assignment[i] = new_f
                    best_score += delta
                    improved = True
                    break  # Move to next customer
    
    # Second pass: try swapping pairs of customers
    if time.time() < deadline - 5.0:
        for i in range(n_customers):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_customers):
                if time.time() > deadline:
                    break
                fi = best_assignment[i]
                fj = best_assignment[j]
                if fi == fj:
                    continue
                
                # Check if swap is feasible
                if load[fi] - demands[i] + demands[j] > capacities[fi]:
                    continue
                if load[fj] - demands[j] + demands[i] > capacities[fj]:
                    continue
                
                # Check cost improvement
                old_assign = assign_cost[i, fi] + assign_cost[j, fj]
                new_assign = assign_cost[i, fj] + assign_cost[j, fi]
                
                # Open cost change: only if one facility becomes empty
                delta_open = 0.0
                count_fi = np.sum(best_assignment == fi)
                count_fj = np.sum(best_assignment == fj)
                
                if count_fi == 1:  # fi becomes empty
                    delta_open -= open_cost[fi]
                if count_fj == 1:  # fj becomes empty
                    delta_open -= open_cost[fj]
                
                # After swap
                new_count_fi = count_fi - 1 + (1 if count_fj == 1 else 0)
                new_count_fj = count_fj - 1 + (1 if count_fi == 1 else 0)
                
                # Recalculate open cost change properly
                old_open_mask = np.zeros(n_facilities, dtype=bool)
                for f in best_assignment:
                    if f >= 0:
                        old_open_mask[f] = True
                old_open_cost = np.sum(open_cost[old_open_mask])
                
                # Simulate swap
                best_assignment[i] = fj
                best_assignment[j] = fi
                new_open_mask = np.zeros(n_facilities, dtype=bool)
                for f in best_assignment:
                    if f >= 0:
                        new_open_mask[f] = True
                new_open_cost = np.sum(open_cost[new_open_mask])
                
                delta = (new_open_cost - old_open_cost) + (new_assign - old_assign)
                
                if delta < -1e-9:
                    # Apply swap
                    load[fi] = load[fi] - demands[i] + demands[j]
                    load[fj] = load[fj] - demands[j] + demands[i]
                    best_score += delta
                    improved = True
                else:
                    # Revert
                    best_assignment[i] = fi
                    best_assignment[j] = fj
    
    return best_assignment.astype(int)
# EVOLVE-BLOCK-END

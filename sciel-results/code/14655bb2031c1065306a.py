"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    def calculate_cost(assignment):
        # assignment: array of facility indices
        total_assign_cost = np.sum(assign_cost[np.arange(n_cust), assignment])
        # Open cost: sum of open_cost for unique facilities used
        used_facilities = np.unique(assignment)
        total_open_cost = np.sum(open_cost[used_facilities])
        return total_assign_cost + total_open_cost

    def is_valid(assignment):
        # Check capacity
        loads = np.zeros(n_fac)
        for i in range(n_cust):
            loads[assignment[i]] += demands[i]
        if np.any(loads > capacities):
            return False
        # Check indices
        if np.any(assignment < 0) or np.any(assignment >= n_fac):
            return False
        return True

    def greedy_assign(order):
        """
        Assign customers in given order to cheapest feasible facility.
        """
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        for i in order:
            # Sort facilities by cost for this customer
            fac_costs = assign_cost[i]
            fac_indices = np.argsort(fac_costs)
            for f in fac_indices:
                if load[f] + demands[i] <= capacities[f]:
                    result[i] = f
                    load[f] += demands[i]
                    break
            # If no feasible facility found, we have a problem.
            # In this problem, it's assumed feasible. 
            # If not, we might need to repair, but let's assume standard feasibility.
            if result[i] == -1:
                # Fallback: assign to facility with most remaining capacity (least likely to be full)
                # or just break. For robustness, pick any with capacity.
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        result[i] = f
                        load[f] += demands[i]
                        break
                if result[i] == -1:
                    return None # Infeasible
        return result

    # 1. Generate diverse initial solutions
    candidates = []
    
    # Strategy 1: Largest demand first
    order1 = np.argsort(-demands)
    cand1 = greedy_assign(order1)
    if cand1 is not None and is_valid(cand1):
        candidates.append(cand1)
        
    # Strategy 2: Smallest demand first
    order2 = np.argsort(demands)
    cand2 = greedy_assign(order2)
    if cand2 is not None and is_valid(cand2):
        candidates.append(cand2)
        
    # Strategy 3: Random orders (a few)
    rng = np.random.default_rng(42)
    for _ in range(5):
        order3 = rng.permutation(n_cust)
        cand3 = greedy_assign(order3)
        if cand3 is not None and is_valid(cand3):
            candidates.append(cand3)
            
    # Strategy 4: Cheapest first (ignoring demand size, just global min cost edge first? No, customer by customer)
    # Let's try sorting by minimum possible assignment cost for the customer
    min_costs = np.min(assign_cost, axis=1)
    order4 = np.argsort(min_costs)
    cand4 = greedy_assign(order4)
    if cand4 is not None and is_valid(cand4):
        candidates.append(cand4)

    if not candidates:
        # Emergency fallback: original greedy
        return greedy_assign(np.argsort(-demands))

    # Select best candidate
    best_cand = min(candidates, key=calculate_cost)
    best_cost = calculate_cost(best_cand)
    
    # 2. Local Search
    # We will try to improve best_cand
    current_assignment = best_cand.copy()
    current_cost = best_cost
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[current_assignment[i]] += demands[i]
        
    # Precompute current cost components for speed?
    # Current cost = sum(assign_cost[i, x_i]) + sum(open_cost[f] for f in used)
    
    def get_used_facilities(assignment):
        return set(assignment)

    improved = True
    max_iters = 5000
    iter_count = 0
    
    while improved and iter_count < max_iters:
        if time.time() > deadline:
            break
        improved = False
        iter_count += 1
        
        # 1. Relocate moves
        # Try moving each customer to another facility
        for i in range(n_cust):
            if time.time() > deadline:
                break
            f_from = current_assignment[i]
            # Try all other facilities
            for f_to in range(n_fac):
                if f_to == f_from:
                    continue
                # Check capacity
                if load[f_to] + demands[i] > capacities[f_to]:
                    continue
                
                # Calculate cost change
                # Assign cost change
                delta_assign = assign_cost[i, f_to] - assign_cost[i, f_from]
                
                # Open cost change
                # If f_from becomes empty, we save open_cost[f_from]
                # If f_to was empty, we incur open_cost[f_to]
                
                # Check if f_from will be empty
                # load[f_from] - demands[i] == 0 ?
                # Note: load array is updated at the end of successful moves.
                # Here we are evaluating a hypothetical move.
                
                # Current used set logic is expensive to update every time.
                # Let's track counts of customers per facility.
                # Actually, we can just check if load[f_from] == demands[i].
                
                delta_open = 0.0
                if load[f_from] == demands[i]:
                    # f_from becomes empty
                    delta_open -= open_cost[f_from]
                
                if load[f_to] == 0:
                    # f_to becomes non-empty
                    delta_open += open_cost[f_to]
                    
                total_delta = delta_assign + delta_open
                
                if total_delta < -1e-9:
                    # Accept move
                    current_assignment[i] = f_to
                    load[f_from] -= demands[i]
                    load[f_to] += demands[i]
                    current_cost += total_delta
                    improved = True
                    break # Move to next customer, as state changed
            
    # 2. Swap moves
    # Try swapping two customers between two facilities
    # This is O(N^2 * F^2) which is 100^2 * 20^2 = 4,000,000 checks. 
    # With 5000 iterations, that's 20 billion ops. Too slow in Python.
    # We should limit swap search or do it less frequently.
    # Let's do a limited swap search: only try swapping customers that are in the same facility? No, that doesn't change open costs.
    # Swap between different facilities is useful for balancing loads or reducing assign costs.
    
    # Alternative: Just do Relocate for now, it's faster.
    # Let's add a simpler swap: for each pair of facilities, try to swap customers.
    # Or just rely on Relocate which can simulate swaps in 2 steps (if capacity allows).
    # Relocate might get stuck in local minima where a swap is needed.
    
    # Let's add a limited swap phase every 10 iterations or so.
    
    # Actually, let's just do Relocate for now to keep it fast and safe.
    # If time permits, we can add swaps.
    
    # Let's add a simple "Close Facility" heuristic as well.
    # If a facility has only 1 customer, and moving it to the cheapest other facility reduces total cost, do it.
    # This is covered by Relocate.
    
    # Let's refine the loop to include swaps in a smarter way.
    # Instead of all pairs, we can try to find beneficial swaps for specific pairs of facilities.
    
    # Given the small size, let's try to implement a basic swap loop but with early termination.
    
    # Let's restart the local search with a combined Relocate/Swap logic.
    
    # For safety and speed, I will stick to Relocate in the tight loop, 
    # and add a separate "Swap" pass that iterates fewer times.
    
    # Let's add a Swap pass.
    swap_improved = True
    while swap_improved and iter_count < max_iters * 2:
        if time.time() > deadline:
            break
        swap_improved = False
        iter_count += 1
        
        # To avoid O(N^2) customer pairs, we can iterate facility pairs.
        # For each pair of facilities (f1, f2), try swapping customers i in f1 and j in f2.
        used_facs = [f for f in range(n_fac) if load[f] > 0]
        
        for idx1, f1 in enumerate(used_facs):
            for f2 in used_facs[idx1+1:]:
                if time.time() > deadline:
                    break
                # Get customers in f1 and f2
                custs_f1 = [i for i in range(n_cust) if current_assignment[i] == f1]
                custs_f2 = [i for i in range(n_cust) if current_assignment[i] == f2]
                
                # Try swapping each pair
                for i in custs_f1:
                    for j in custs_f2:
                        # Check capacity after swap
                        # i goes to f2, j goes to f1
                        # load[f2] - demands[j] + demands[i] <= capacities[f2]
                        # load[f1] - demands[i] + demands[j] <= capacities[f1]
                        
                        if load[f2] - demands[j] + demands[i] > capacities[f2]:
                            continue
                        if load[f1] - demands[i] + demands[j] > capacities[f1]:
                            continue
                        
                        # Cost change
                        # Assign cost:
                        # - assign_cost[i, f1] - assign_cost[j, f2]
                        # + assign_cost[i, f2] + assign_cost[j, f1]
                        delta_assign = (assign_cost[i, f2] + assign_cost[j, f1]) - (assign_cost[i, f1] + assign_cost[j, f2])
                        
                        # Open cost: 
                        # Does the set of used facilities change?
                        # f1 and f2 are already used.
                        # After swap, f1 and f2 are still used (unless one becomes empty).
                        # If f1 had only i, and we move i away, f1 might become empty if no other customers.
                        # But j is moving TO f1. So f1 will have j. So f1 is not empty.
                        # Similarly, f2 will have i. So f2 is not empty.
                        # So open cost does NOT change in a swap between two non-empty facilities.
                        # Exception: If f1 had ONLY i, and f2 had ONLY j.
                        # Then before: both used. After: both used. No change.
                        # Wait, if f1 had ONLY i, and we swap i with j from f2.
                        # f1 gets j. f2 gets i.
                        # Both are still used.
                        # So delta_open is always 0 for swaps between two currently used facilities.
                        
                        if delta_assign < -1e-9:
                            # Accept swap
                            current_assignment[i] = f2
                            current_assignment[j] = f1
                            load[f1] -= demands[i]
                            load[f1] += demands[j]
                            load[f2] -= demands[j]
                            load[f2] += demands[i]
                            current_cost += delta_assign
                            swap_improved = True
                            break
                    if swap_improved:
                        break
                if swap_improved:
                    break
            if swap_improved:
                break
    
    return current_assignment

# EVOLVE-BLOCK-END

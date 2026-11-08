"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Pre-convert to numpy arrays for speed
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    def calculate_cost(assignment):
        # assignment: array of facility indices
        if np.any(assignment < 0):
            return float('inf')
        total_assign_cost = np.sum(assign_cost[np.arange(n_cust), assignment])
        # Open cost: sum of open_cost for unique facilities used
        used_facilities = np.unique(assignment)
        total_open_cost = np.sum(open_cost[used_facilities])
        return total_assign_cost + total_open_cost

    def is_valid(assignment):
        # Check capacity
        if np.any(assignment < 0) or np.any(assignment >= n_fac):
            return False
        loads = np.zeros(n_fac)
        np.add.at(loads, assignment, demands)
        if np.any(loads > capacities + 1e-9):
            return False
        return True

    def greedy_assign(order, mu=None):
        """
        Assign customers in given order to cheapest feasible facility.
        If mu is provided, use it to adjust costs (Lagrangian).
        """
        result = np.full(n_cust, -1, dtype=int)
        load = np.zeros(n_fac)
        
        if mu is None:
            mu = np.zeros(n_fac)
            
        for i in order:
            # Calculate effective cost for each facility
            eff_costs = assign_cost[i] + mu * demands[i]
            fac_indices = np.argsort(eff_costs)
            
            assigned = False
            for f in fac_indices:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    result[i] = f
                    load[f] += demands[i]
                    assigned = True
                    break
            
            if not assigned:
                # Fallback: assign to facility with most remaining capacity
                rem_cap = capacities - load
                fac_indices_cap = np.argsort(-rem_cap)
                for f in fac_indices_cap:
                    if load[f] + demands[i] <= capacities[f] + 1e-9:
                        result[i] = f
                        load[f] += demands[i]
                        assigned = True
                        break
                        
            if not assigned:
                return None # Infeasible
                
        return result

    def local_search(initial_assignment, max_time=20.0):
        """
        Perform local search starting from initial_assignment.
        Uses Relocate and Swap moves.
        Returns the best assignment found.
        """
        current_assignment = initial_assignment.copy()
        current_cost = calculate_cost(current_assignment)
        
        # Precompute loads
        load = np.zeros(n_fac)
        np.add.at(load, current_assignment, demands)
        
        # Track customer count per facility for fast open cost delta calculation
        cust_count = np.zeros(n_fac, dtype=int)
        for i in range(n_cust):
            cust_count[current_assignment[i]] += 1

        local_deadline = time.time() + max_time
        improved = True
        iter_count = 0
        max_iters = 10000
        
        while improved and iter_count < max_iters:
            if time.time() > local_deadline:
                break
            improved = False
            iter_count += 1
            
            # 1. Relocate moves
            for i in range(n_cust):
                if time.time() > local_deadline:
                    break
                f_from = current_assignment[i]
                
                best_delta = 0.0
                best_f_to = -1
                
                for f_to in range(n_fac):
                    if f_to == f_from:
                        continue
                    # Check capacity
                    if load[f_to] + demands[i] > capacities[f_to] + 1e-9:
                        continue
                    
                    # Calculate cost change
                    delta_assign = assign_cost[i, f_to] - assign_cost[i, f_from]
                    
                    # Open cost change
                    delta_open = 0.0
                    # If f_from becomes empty (cust_count[f_from] == 1)
                    if cust_count[f_from] == 1:
                        delta_open -= open_cost[f_from]
                    # If f_to was empty (cust_count[f_to] == 0)
                    if cust_count[f_to] == 0:
                        delta_open += open_cost[f_to]
                    
                    total_delta = delta_assign + delta_open
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f_to = f_to
                
                if best_f_to != -1:
                    # Accept best move
                    current_assignment[i] = best_f_to
                    load[f_from] -= demands[i]
                    load[best_f_to] += demands[i]
                    cust_count[f_from] -= 1
                    cust_count[best_f_to] += 1
                    current_cost += best_delta
                    improved = True
            
            # 2. Swap moves
            used_facs = [f for f in range(n_fac) if cust_count[f] > 0]
            
            if time.time() < local_deadline - 1.0:
                for idx1, f1 in enumerate(used_facs):
                    for f2 in used_facs[idx1+1:]:
                        if time.time() > local_deadline:
                            break
                        
                        # Get customers in f1 and f2
                        custs_f1 = [i for i in range(n_cust) if current_assignment[i] == f1]
                        custs_f2 = [i for i in range(n_cust) if current_assignment[i] == f2]
                        
                        best_swap_delta = 0.0
                        best_i, best_j = -1, -1
                        
                        for i in custs_f1:
                            for j in custs_f2:
                                # Check capacity after swap
                                if load[f2] - demands[j] + demands[i] > capacities[f2] + 1e-9:
                                    continue
                                if load[f1] - demands[i] + demands[j] > capacities[f1] + 1e-9:
                                    continue
                                
                                # Cost change
                                delta_assign = (assign_cost[i, f2] + assign_cost[j, f1]) - (assign_cost[i, f1] + assign_cost[j, f2])
                                
                                if delta_assign < best_swap_delta:
                                    best_swap_delta = delta_assign
                                    best_i, best_j = i, j
                        
                        if best_i != -1:
                            # Accept best swap
                            current_assignment[best_i] = f2
                            current_assignment[best_j] = f1
                            load[f1] -= demands[best_i]
                            load[f1] += demands[best_j]
                            load[f2] -= demands[best_j]
                            load[f2] += demands[best_i]
                            current_cost += best_swap_delta
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    continue

        return current_assignment, current_cost

    def simulated_annealing(initial_assignment, max_time=30.0):
        """
        Simulated annealing to escape local optima.
        """
        current_assignment = initial_assignment.copy()
        current_cost = calculate_cost(current_assignment)
        
        load = np.zeros(n_fac)
        np.add.at(load, current_assignment, demands)
        cust_count = np.zeros(n_fac, dtype=int)
        for i in range(n_cust):
            cust_count[current_assignment[i]] += 1
        
        best_assignment = current_assignment.copy()
        best_cost = current_cost
        
        local_deadline = time.time() + max_time
        T = 100.0  # Initial temperature
        cooling_rate = 0.9995
        
        while time.time() < local_deadline and T > 0.1:
            T *= cooling_rate
            
            # Pick a random customer and try to move it
            i = np.random.randint(n_cust)
            f_from = current_assignment[i]
            
            # Pick a random different facility
            candidates = [f for f in range(n_fac) if f != f_from]
            f_to = np.random.choice(candidates)
            
            # Check capacity
            if load[f_to] + demands[i] > capacities[f_to] + 1e-9:
                continue
            
            # Calculate cost delta
            delta_assign = assign_cost[i, f_to] - assign_cost[i, f_from]
            delta_open = 0.0
            if cust_count[f_from] == 1:
                delta_open -= open_cost[f_from]
            if cust_count[f_to] == 0:
                delta_open += open_cost[f_to]
            delta = delta_assign + delta_open
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                # Accept the move
                current_assignment[i] = f_to
                load[f_from] -= demands[i]
                load[f_to] += demands[i]
                cust_count[f_from] -= 1
                cust_count[f_to] += 1
                current_cost += delta
                
                # Update best
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assignment = current_assignment.copy()
            
            # Occasionally do a swap
            if np.random.random() < 0.3:
                used_facs = [f for f in range(n_fac) if cust_count[f] > 0]
                if len(used_facs) >= 2:
                    f1, f2 = np.random.choice(used_facs, size=2, replace=False)
                    custs_f1 = [i for i in range(n_cust) if current_assignment[i] == f1]
                    custs_f2 = [i for i in range(n_cust) if current_assignment[i] == f2]
                    
                    if len(custs_f1) > 0 and len(custs_f2) > 0:
                        i1 = np.random.choice(custs_f1)
                        i2 = np.random.choice(custs_f2)
                        
                        # Check capacity
                        if load[f2] - demands[i2] + demands[i1] <= capacities[f2] + 1e-9 and \
                           load[f1] - demands[i1] + demands[i2] <= capacities[f1] + 1e-9:
                            delta = (assign_cost[i1, f2] + assign_cost[i2, f1]) - \
                                    (assign_cost[i1, f1] + assign_cost[i2, f2])
                            
                            if delta < 0 or np.random.random() < np.exp(-delta / T):
                                current_assignment[i1] = f2
                                current_assignment[i2] = f1
                                load[f1] -= demands[i1]
                                load[f1] += demands[i2]
                                load[f2] -= demands[i2]
                                load[f2] += demands[i1]
                                current_cost += delta
                                
                                if current_cost < best_cost:
                                    best_cost = current_cost
                                    best_assignment = current_assignment.copy()
        
        return best_assignment, best_cost

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
        
    # Strategy 3: Random orders
    rng = np.random.default_rng(42)
    for _ in range(10):
        order3 = rng.permutation(n_cust)
        cand3 = greedy_assign(order3)
        if cand3 is not None and is_valid(cand3):
            candidates.append(cand3)
            
    # Strategy 4: Cheapest first (min assign cost)
    min_costs = np.min(assign_cost, axis=1)
    order4 = np.argsort(min_costs)
    cand4 = greedy_assign(order4)
    if cand4 is not None and is_valid(cand4):
        candidates.append(cand4)
        
    # Strategy 5: Lagrangian-inspired initializations
    for _ in range(10):
        mu = rng.random(n_fac) * 5.0
        order5 = np.argsort(-demands)
        cand5 = greedy_assign(order5, mu)
        if cand5 is not None and is_valid(cand5):
            candidates.append(cand5)

    if not candidates:
        return greedy_assign(np.argsort(-demands))

    # Select best candidate
    candidates.sort(key=calculate_cost)
    best_cand = candidates[0]
    best_cost = calculate_cost(best_cand)
    
    # 2. Local Search on the best candidate
    elapsed = time.time() - start_time
    remaining = deadline - start_time - elapsed
    
    ls_time = min(remaining - 10.0, 40.0)
    if ls_time < 1.0:
        ls_time = 1.0
        
    best_assignment, best_cost = local_search(best_cand, max_time=ls_time)
    
    # 3. Simulated annealing to escape local optima
    if time.time() < deadline - 15.0:
        sa_time = min(deadline - time.time() - 10.0, 30.0)
        if sa_time > 5.0:
            sa_assignment, sa_cost = simulated_annealing(best_assignment, max_time=sa_time)
            if sa_cost < best_cost:
                best_assignment = sa_assignment
                best_cost = sa_cost
    
    # 4. Final local search polish
    if time.time() < deadline - 5.0:
        polish_time = min(deadline - time.time() - 2.0, 15.0)
        if polish_time > 1.0:
            final_assignment, final_cost = local_search(best_assignment, max_time=polish_time)
            if final_cost < best_cost:
                best_assignment = final_assignment
                best_cost = final_cost

    return best_assignment

# EVOLVE-BLOCK-END

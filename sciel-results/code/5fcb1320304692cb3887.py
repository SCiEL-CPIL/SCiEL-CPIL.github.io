"""Capacitated facility location: vectorized local search with diverse initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    deadline = time.time() + 135.0  # safety margin before 150s limit
    
    def total_cost(assignment):
        """Compute total cost: open costs for used facilities + assignment costs."""
        used = np.unique(assignment)
        if len(used) == 0:
            return 0.0
        open_sum = float(np.sum(open_cost[used]))
        assign_sum = float(np.sum(assign_cost[np.arange(n_customers), assignment]))
        return open_sum + assign_sum
    
    def is_feasible(assignment):
        """Check if assignment respects capacities."""
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
        return np.all(load <= capacities + 1e-9)
    
    def greedy_cheapest():
        """Assign each customer to cheapest feasible facility, largest demand first."""
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        order = np.argsort(-demands)
        for i in order:
            best_f = -1
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    best_f = f
                    break
            if best_f == -1:
                for f in range(n_facilities):
                    if load[f] + demands[i] <= capacities[f] + 1e-9:
                        best_f = f
                        break
            if best_f == -1:
                best_f = 0
            load[best_f] += demands[i]
            result[i] = best_f
        return result
    
    def greedy_with_penalty(penalty_weight):
        """Greedy with open cost penalty to encourage fewer facilities."""
        result = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        used = np.zeros(n_facilities, dtype=bool)
        order = np.argsort(-demands)
        for i in order:
            best_f = -1
            best_val = np.inf
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    eff = assign_cost[i, f]
                    if not used[f]:
                        eff += penalty_weight * open_cost[f]
                    if eff < best_val:
                        best_val = eff
                        best_f = f
            if best_f == -1:
                for f in range(n_facilities):
                    if load[f] + demands[i] <= capacities[f] + 1e-9:
                        best_f = f
                        break
            if best_f == -1:
                best_f = 0
            load[best_f] += demands[i]
            used[best_f] = True
            result[i] = best_f
        return result
    
    def vectorized_local_search(assignment, max_time=10.0):
        """
        Fast vectorized local search with single moves only.
        This is the core refinement engine - much faster than O(N^2) swap loops.
        """
        assignment = assignment.copy()
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[assignment[i]] += demands[i]
        
        start_time = time.time()
        improved = True
        no_improve_count = 0
        
        while improved and (time.time() - start_time) < max_time and time.time() < deadline - 5.0:
            improved = False
            no_improve_count += 1
            
            if no_improve_count > 50:
                break
            
            # Vectorized single move search
            any_improve_in_pass = False
            for i in range(n_customers):
                if time.time() > deadline - 5.0:
                    return assignment
                
                cur_f = assignment[i]
                d_i = demands[i]
                
                # Find all feasible facilities (excluding current)
                feasible_mask = (load + d_i <= capacities + 1e-9)
                feasible_mask[cur_f] = False
                feasible = np.where(feasible_mask)[0]
                
                if len(feasible) == 0:
                    continue
                
                # Compute effective cost delta for each feasible facility
                base_assign = assign_cost[i, cur_f]
                deltas = assign_cost[i, feasible] - base_assign
                
                # Open cost adjustment: if current facility becomes empty, we save its open cost
                if load[cur_f] == d_i:
                    deltas = deltas - open_cost[cur_f]
                
                # If target facility is currently empty, we pay its open cost
                for idx, f_new in enumerate(feasible):
                    if load[f_new] <= 1e-9:
                        deltas[idx] += open_cost[f_new]
                
                best_idx = int(np.argmin(deltas))
                best_delta = deltas[best_idx]
                
                if best_delta < -1e-9:
                    f_new = feasible[best_idx]
                    assignment[i] = f_new
                    load[cur_f] -= d_i
                    load[f_new] += d_i
                    improved = True
                    any_improve_in_pass = True
            
            if not any_improve_in_pass:
                # Try a few random swaps to escape local optima
                rng = np.random.default_rng(int(time.time() * 1000) % (2**32))
                for _ in range(200):
                    if time.time() > deadline - 5.0:
                        return assignment
                    
                    i, j = rng.integers(0, n_customers, size=2)
                    if i == j:
                        continue
                    
                    f_i = assignment[i]
                    f_j = assignment[j]
                    if f_i == f_j:
                        continue
                    
                    d_i = demands[i]
                    d_j = demands[j]
                    
                    # Check feasibility after swap
                    if load[f_j] - d_j + d_i > capacities[f_j] + 1e-9:
                        continue
                    if load[f_i] - d_i + d_j > capacities[f_i] + 1e-9:
                        continue
                    
                    delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                                   (assign_cost[j, f_i] - assign_cost[j, f_j])
                    
                    old_load_i = load[f_i]
                    old_load_j = load[f_j]
                    new_load_i = old_load_i - d_i + d_j
                    new_load_j = old_load_j - d_j + d_i
                    
                    old_open_i = open_cost[f_i] if old_load_i > 1e-9 else 0.0
                    old_open_j = open_cost[f_j] if old_load_j > 1e-9 else 0.0
                    new_open_i = open_cost[f_i] if new_load_i > 1e-9 else 0.0
                    new_open_j = open_cost[f_j] if new_load_j > 1e-9 else 0.0
                    delta_open = (new_open_i + new_open_j) - (old_open_i + old_open_j)
                    
                    total_delta = delta_assign + delta_open
                    if total_delta < -1e-9:
                        assignment[i], assignment[j] = assignment[j], assignment[i]
                        load[f_i] = new_load_i
                        load[f_j] = new_load_j
                        improved = True
                        break
        
        return assignment
    
    def facility_close_attempt(assignment, load):
        """Try to close a facility and reassign its customers."""
        assignment = assignment.copy()
        load = load.copy()
        
        for f_close in range(n_facilities):
            if time.time() > deadline - 5.0:
                return assignment, load, False
            
            if load[f_close] <= 1e-9:
                continue
            
            custs = np.where(assignment == f_close)[0]
            if len(custs) == 0:
                continue
            
            # Try to reassign all customers from f_close to other facilities
            temp_load = load.copy()
            temp_load[f_close] = 0
            temp_assign = assignment.copy()
            
            success = True
            # Process customers in descending demand order
            order = custs[np.argsort(-demands[custs])]
            
            for c in order:
                d_c = demands[c]
                feasible_mask = (temp_load + d_c <= capacities + 1e-9)
                feasible_mask[f_close] = False
                feasible = np.where(feasible_mask)[0]
                
                if len(feasible) == 0:
                    success = False
                    break
                
                costs = assign_cost[c, feasible]
                # Add open cost penalty for new facilities
                open_pen = np.where(temp_load[feasible] <= 1e-9, open_cost[feasible], 0.0)
                adjusted = costs + open_pen
                best_f = int(feasible[np.argmin(adjusted)])
                temp_assign[c] = best_f
                temp_load[best_f] += d_c
            
            if not success:
                continue
            
            # Check if this closes the facility and improves cost
            old_assign_sum = float(np.sum(assign_cost[custs, f_close]))
            new_assign_sum = float(np.sum(assign_cost[custs, temp_assign[custs]]))
            
            old_open = float(np.sum(open_cost[load > 1e-9]))
            new_open = float(np.sum(open_cost[temp_load > 1e-9]))
            
            delta = (new_assign_sum - old_assign_sum) + (new_open - old_open)
            
            if delta < -1e-9:
                assignment[:] = temp_assign
                load[:] = temp_load
                return assignment, load, True
        
        return assignment, load, False
    
    # Generate diverse initial solutions
    candidates = []
    
    # Strategy 1: greedy cheapest
    if time.time() < deadline - 20:
        a1 = greedy_cheapest()
        if is_feasible(a1):
            a1_refined = vectorized_local_search(a1, max_time=8.0)
            c1 = total_cost(a1_refined)
            candidates.append((c1, a1_refined))
    
    # Strategy 2: greedy with various penalties
    for pw in [0.1, 0.3, 0.5, 1.0, 2.0]:
        if time.time() > deadline - 20:
            break
        a = greedy_with_penalty(pw)
        if is_feasible(a):
            a_refined = vectorized_local_search(a, max_time=8.0)
            c = total_cost(a_refined)
            candidates.append((c, a_refined))
    
    # Strategy 3: random feasible initialization
    for trial in range(3):
        if time.time() > deadline - 20:
            break
        rng = np.random.default_rng(42 + trial)
        a = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities)
        order = rng.permutation(n_customers)
        feasible = True
        for i in order:
            f_order = rng.permutation(n_facilities)
            placed = False
            for f in f_order:
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    a[i] = f
                    load[f] += demands[i]
                    placed = True
                    break
            if not placed:
                feasible = False
                break
        if not feasible:
            continue
        a_refined = vectorized_local_search(a, max_time=8.0)
        c = total_cost(a_refined)
        candidates.append((c, a_refined))
    
    # Select best candidate
    if not candidates:
        a = greedy_cheapest()
        return a.astype(int)
    
    best_cost, best_assign = min(candidates, key=lambda x: x[0])
    
    # Final polish: facility closing attempts + additional local search
    if time.time() < deadline - 10:
        load = np.zeros(n_facilities)
        for i in range(n_customers):
            load[best_assign[i]] += demands[i]
        
        # Try facility closing
        improved = True
        while improved and time.time() < deadline - 10:
            best_assign, load, improved = facility_close_attempt(best_assign, load)
        
        # Final vectorized local search
        if time.time() < deadline - 5:
            best_assign = vectorized_local_search(best_assign, max_time=10.0)
    
    return best_assign.astype(int)
# EVOLVE-BLOCK-END

import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Precompute sorted facility indices for each customer by cost
    # sorted_facilities[i] is a list of facility indices sorted by assign_cost[i, f]
    sorted_facilities = np.argsort(assign_cost, axis=1)
    
    def calculate_cost(assignment, open_set):
        open_cost_sum = np.sum(open_cost[open_set])
        assign_cost_sum = np.sum(assign_cost[np.arange(n_customers), assignment])
        return open_cost_sum + assign_cost_sum

    def try_reassign_customer(i, assignment, open_set, current_load):
        # Try to move customer i to a better open facility
        f_current = assignment[i]
        best_gain = 0
        best_f = -1
        
        # Get facilities sorted by cost for customer i
        for f in sorted_facilities[i]:
            if f == f_current:
                continue
            if not open_set[f]:
                continue
            # Check capacity
            if current_load[f] + demands[i] > capacities[f]:
                continue
            
            cost_new = assign_cost[i, f]
            cost_old = assign_cost[i, f_current]
            gain = cost_old - cost_new
            if gain > best_gain:
                best_gain = gain
                best_f = f
        
        if best_f != -1:
            return best_f, best_gain
        return -1, 0

    def try_close_facility(f, assignment, open_set, current_load):
        # If facility f is open, check if closing it saves cost
        if not open_set[f]:
            return False, 0, None
        
        customers_at_f = [i for i in range(n_customers) if assignment[i] == f]
        if len(customers_at_f) == 0:
            return True, open_cost[f], None
        
        # Try to reassign all customers_at_f to other open facilities
        # Greedy reassignment: Sort by demand desc
        customers_at_f.sort(key=lambda x: -demands[x])
        
        temp_load = current_load.copy()
        temp_load[f] = 0 # Remove load from f
        
        new_assignment = assignment.copy()
        cost_increase = 0
        feasible = True
        
        for i in customers_at_f:
            old_cost = assign_cost[i, f]
            best_new_cost = None
            best_new_f = -1
            
            for g in sorted_facilities[i]:
                if g == f:
                    continue
                if not open_set[g]:
                    continue
                if temp_load[g] + demands[i] > capacities[g]:
                    continue
                
                c = assign_cost[i, g]
                if best_new_cost is None or c < best_new_cost:
                    best_new_cost = c
                    best_new_f = g
            
            if best_new_f == -1:
                feasible = False
                break
            
            temp_load[best_new_f] += demands[i]
            new_assignment[i] = best_new_f
            cost_increase += (best_new_cost - old_cost)
        
        if not feasible:
            return False, 0, None
        
        total_saving = open_cost[f] - cost_increase
        if total_saving > 0:
            return True, total_saving, new_assignment
        else:
            return False, 0, None

    def try_open_facility(f, assignment, open_set, current_load):
        # If facility f is closed, check if opening it saves cost
        if open_set[f]:
            return False, 0, None
        
        # Candidates: customers where f is cheaper than current
        candidates = []
        for i in range(n_customers):
            f_cur = assignment[i]
            if assign_cost[i, f] < assign_cost[i, f_cur]:
                candidates.append(i)
        
        if not candidates:
            return False, 0, None
        
        # Sort candidates by potential savings descending
        candidates.sort(key=lambda x: assign_cost[x, assignment[x]] - assign_cost[x, f], reverse=True)
        
        new_assignment = assignment.copy()
        total_saving = 0
        capacity_remaining = capacities[f]
        
        for i in candidates:
            if capacity_remaining < demands[i]:
                continue
            
            f_cur = assignment[i]
            saving = assign_cost[i, f_cur] - assign_cost[i, f]
            if saving > 0:
                new_assignment[i] = f
                capacity_remaining -= demands[i]
                total_saving += saving
                
                if capacity_remaining <= 0:
                    break
        
        if total_saving > open_cost[f]:
            return True, total_saving - open_cost[f], new_assignment
        else:
            return False, 0, None

    # Initialization: Start with all facilities open, assign to nearest
    open_set = np.ones(n_facilities, dtype=bool)
    assignment = np.zeros(n_customers, dtype=int)
    
    for i in range(n_customers):
        for f in sorted_facilities[i]:
            if open_set[f]:
                assignment[i] = f
                break
    
    current_load = np.zeros(n_facilities)
    for i in range(n_customers):
        current_load[assignment[i]] += demands[i]
        
    # Subtractive Phase: Close facilities that are not worth keeping
    improved = True
    while improved:
        improved = False
        best_saving = 0
        best_f_to_close = -1
        best_new_assignment = None
        
        for f in range(n_facilities):
            if not open_set[f]:
                continue
            success, saving, new_assign = try_close_facility(f, assignment, open_set, current_load)
            if success and saving > best_saving:
                best_saving = saving
                best_f_to_close = f
                best_new_assignment = new_assign
        
        if best_f_to_close != -1:
            open_set[best_f_to_close] = False
            assignment = best_new_assignment
            current_load = np.zeros(n_facilities)
            for i in range(n_customers):
                current_load[assignment[i]] += demands[i]
            improved = True

    # Local Search Phase
    start_time = time.time()
    time_limit = 145.0 # 5s margin from 150s limit
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Reassign customers
        for i in range(n_customers):
            f_new, gain = try_reassign_customer(i, assignment, open_set, current_load)
            if f_new != -1 and gain > 0:
                f_old = assignment[i]
                assignment[i] = f_new
                current_load[f_old] -= demands[i]
                current_load[f_new] += demands[i]
                improved = True
        
        # 2. Close facilities
        best_saving = 0
        best_f_to_close = -1
        best_new_assignment = None
        
        for f in range(n_facilities):
            if not open_set[f]:
                continue
            success, saving, new_assign = try_close_facility(f, assignment, open_set, current_load)
            if success and saving > best_saving:
                best_saving = saving
                best_f_to_close = f
                best_new_assignment = new_assign
        
        if best_f_to_close != -1:
            open_set[best_f_to_close] = False
            assignment = best_new_assignment
            current_load = np.zeros(n_facilities)
            for i in range(n_customers):
                current_load[assignment[i]] += demands[i]
            improved = True
            
        # 3. Open facilities
        best_saving = 0
        best_f_to_open = -1
        best_new_assignment = None
        
        for f in range(n_facilities):
            if open_set[f]:
                continue
            success, saving, new_assign = try_open_facility(f, assignment, open_set, current_load)
            if success and saving > best_saving:
                best_saving = saving
                best_f_to_open = f
                best_new_assignment = new_assign
        
        if best_f_to_open != -1:
            open_set[best_f_to_open] = True
            assignment = best_new_assignment
            current_load = np.zeros(n_facilities)
            for i in range(n_customers):
                current_load[assignment[i]] += demands[i]
            improved = True
            
        if not improved:
            break
            
    return assignment

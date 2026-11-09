import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Greedy Initialization
    # Sort customers by demand descending
    order = np.argsort(-demands)
    assignment = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    for i in order:
        # Find feasible facilities sorted by assign_cost
        # We want the cheapest feasible one
        # For efficiency in init, just iterate all and pick min cost among feasible
        best_f = -1
        min_cost = np.inf
        for f in range(n_facilities):
            if load[f] + demands[i] <= capacities[f]:
                if assign_cost[i, f] < min_cost:
                    min_cost = assign_cost[i, f]
                    best_f = f
        if best_f != -1:
            assignment[i] = best_f
            load[best_f] += demands[i]
        else:
            # Should not happen if capacities are sufficient, but fallback to first feasible
            for f in range(n_facilities):
                if load[f] + demands[i] <= capacities[f]:
                    assignment[i] = f
                    load[f] += demands[i]
                    break
            else:
                # If no capacity (should not happen), assign to 0 to keep valid indices, 
                # though this would be invalid. Assuming valid instances.
                assignment[i] = 0
                load[0] += demands[i]

    # Calculate initial cost
    def calc_cost(assignment, load):
        open_sum = sum(open_cost[f] for f in range(n_facilities) if load[f] > 0)
        assign_sum = sum(assign_cost[i, assignment[i]] for i in range(n_customers))
        return open_sum + assign_sum

    current_cost = calc_cost(assignment, load)
    best_assignment = assignment.copy()
    best_load = load.copy()
    best_cost = current_cost

    # 2. Simulated Annealing
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    T = 100.0  # Initial temperature
    alpha = 0.999  # Cooling factor
    min_T = 0.1
    
    # Estimate iterations: 100 customers * 20 facilities = 2000 possible moves per sweep?
    # We will do random single moves.
    # Let's run until time limit.
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Pick random customer
        i = np.random.randint(n_customers)
        current_f = assignment[i]
        
        # Pick random different facility
        # To be efficient, we can pick a random f != current_f
        if n_facilities == 1:
            break
        f_new = np.random.randint(n_facilities)
        while f_new == current_f:
            f_new = np.random.randint(n_facilities)
            
        # Check capacity
        if load[f_new] + demands[i] > capacities[f_new]:
            continue
            
        # Calculate delta cost
        # Assignment cost change
        delta_assign = assign_cost[i, f_new] - assign_cost[i, current_f]
        
        # Opening cost change
        delta_open = 0.0
        old_load_cur = load[current_f] - demands[i]
        new_load_new = load[f_new] + demands[i]
        
        # Status change for current_f (losing demand)
        # If it becomes empty, we save open_cost[current_f]
        if old_load_cur == 0:
            delta_open -= open_cost[current_f]
            
        # Status change for f_new (gaining demand)
        # If it was empty, we pay open_cost[f_new]
        if load[f_new] == 0:
            delta_open += open_cost[f_new]
            
        delta_total = delta_assign + delta_open
        
        # Acceptance
        if delta_total < 0:
            # Accept
            assignment[i] = f_new
            load[current_f] = old_load_cur
            load[f_new] = new_load_new
            current_cost += delta_total
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_assignment = assignment.copy()
                best_load = load.copy()
        else:
            # Reject or accept with prob
            # Avoid log(0)
            if T > 0:
                prob = np.exp(-delta_total / T)
                if np.random.rand() < prob:
                    assignment[i] = f_new
                    load[current_f] = old_load_cur
                    load[f_new] = new_load_new
                    current_cost += delta_total
        
        # Cool
        T *= alpha

    # 3. Local Search Polish (Hill Climbing)
    # Run until no improvement or time limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try moving each customer to its best feasible alternative
        # To avoid O(N^2) complexity issues if N was large, but N=100 is small.
        # We can randomize the order or just iterate.
        for i in range(n_customers):
            current_f = best_assignment[i]
            best_f = current_f
            best_local_cost = best_cost # This is tricky, cost is global.
            
            # We need to evaluate the move for i.
            # Let's just find the best f' for i given the current state of best_assignment/best_load
            # But moving i affects the cost of i and potentially the open status of current_f and f'.
            
            # To do this properly in a single pass, we might need to recalculate or do a more careful local search.
            # Given time constraints, a simple 1-swap hill climbing on the best solution found so far.
            
            # Let's try all f' for i
            min_delta = 0.0
            best_move_f = -1
            
            for f in range(n_facilities):
                if f == current_f:
                    continue
                if best_load[f] + demands[i] > capacities[f]:
                    continue
                
                # Calculate delta for moving i from current_f to f
                d_assign = assign_cost[i, f] - assign_cost[i, current_f]
                d_open = 0.0
                
                old_load_cur = best_load[current_f] - demands[i]
                new_load_new = best_load[f] + demands[i]
                
                if old_load_cur == 0:
                    d_open -= open_cost[current_f]
                if best_load[f] == 0:
                    d_open += open_cost[f]
                    
                d_total = d_assign + d_open
                
                if d_total < min_delta:
                    min_delta = d_total
                    best_move_f = f
            
            if best_move_f != -1:
                # Apply move
                best_assignment[i] = best_move_f
                best_load[current_f] -= demands[i]
                best_load[best_move_f] += demands[i]
                best_cost += min_delta
                improved = True

    return best_assignment

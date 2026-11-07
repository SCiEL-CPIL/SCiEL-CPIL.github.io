"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Pre-compute sorted facility indices for each customer based on assignment cost
    # This helps in greedy initialization
    sorted_facilities = np.argsort(assign_cost, axis=1)
    
    def compute_cost(assignment, load):
        assign_sum = np.sum(assign_cost[np.arange(n_customers), assignment])
        open_sum = np.sum(open_cost[load > 0])
        return assign_sum + open_sum

    def greedy_init(seed):
        """
        Generate an initial feasible solution using greedy assignment.
        """
        rng = np.random.default_rng(seed)
        assignment = np.full(n_customers, -1, dtype=int)
        load = np.zeros(n_facilities)
        
        # Sort customers by demand descending to place hardest first
        # Add some randomness to break ties
        order = np.argsort(-demands)
        rng.shuffle(order) # Randomize order among same demands or slightly disturb
        
        for i in order:
            d_i = demands[i]
            assigned = False
            # Try facilities in order of increasing assignment cost
            for f in sorted_facilities[i]:
                if load[f] + d_i <= capacities[f]:
                    assignment[i] = f
                    load[f] += d_i
                    assigned = True
                    break
            
            if not assigned:
                # Fallback: assign to facility with most remaining capacity
                remaining = capacities - load
                feasible = np.where(remaining >= d_i)[0]
                if len(feasible) > 0:
                    # Pick randomly from feasible to introduce diversity
                    f = rng.choice(feasible)
                    assignment[i] = f
                    load[f] += d_i
                else:
                    # Should not happen if total capacity is sufficient, but handle gracefully
                    # Assign to facility with max remaining capacity even if it exceeds? 
                    # No, must be feasible. If no feasible, we have a problem.
                    # For this task, total capacity is usually sufficient.
                    # If strictly infeasible, we might need to relax, but let's assume feasible.
                    f = np.argmax(remaining)
                    assignment[i] = f
                    load[f] += d_i
        
        return assignment, load

    def simulated_annealing(assignment, load, max_time=5.0):
        """
        Run Simulated Annealing starting from a given state.
        """
        current_assignment = assignment.copy()
        current_load = load.copy()
        current_cost = compute_cost(current_assignment, current_load)
        
        best_assignment = current_assignment.copy()
        best_load = current_load.copy()
        best_cost = current_cost
        
        # SA Parameters
        T_start = 100.0
        T_end = 0.1
        # Number of iterations to reach T_end
        # We will run until time limit
        start_time = time.time()
        rng = np.random.default_rng(12345)
        
        # We need to track the temperature. Since we don't know exact iterations ahead of time,
        # we can use a cooling factor per iteration.
        # Let's estimate iterations. 
        # Each iteration involves scanning a few customers or a random customer.
        # Let's just run fixed number of iterations based on time, or use a cooling schedule that doesn't depend on total N.
        # Better: Use a cooling factor alpha.
        alpha = 0.995 # Cooling factor
        
        T = T_start
        
        while time.time() - start_time < max_time:
            # Pick a random customer
            i = rng.integers(n_customers)
            f_old = current_assignment[i]
            d_i = demands[i]
            
            # Determine move type: 1-move or 2-swap
            # 80% 1-move, 20% 2-swap
            if rng.random() < 0.8:
                # 1-Move
                # Pick a random new facility, or one from sorted list?
                # Random is better for SA exploration.
                f_new = rng.integers(n_facilities)
                
                if f_new == f_old:
                    continue
                
                # Capacity check
                if current_load[f_new] + d_i > capacities[f_new]:
                    continue
                
                # Calculate delta
                delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                
                delta_open = 0.0
                # Check if f_old becomes empty
                if current_load[f_old] == d_i:
                    delta_open -= open_cost[f_old]
                # Check if f_new becomes open
                if current_load[f_new] == 0:
                    delta_open += open_cost[f_new]
                
                delta = delta_assign + delta_open
                
                # Acceptance criterion
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    # Apply move
                    current_load[f_old] -= d_i
                    current_load[f_new] += d_i
                    current_assignment[i] = f_new
                    current_cost += delta
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_assignment = current_assignment.copy()
                        best_load = current_load.copy()
                        
            else:
                # 2-Swap
                # Pick a random other customer j
                j = rng.integers(n_customers)
                if i == j:
                    continue
                
                f_j = current_assignment[j]
                d_j = demands[j]
                
                if f_j == f_old:
                    continue
                
                # Swap i and j
                # New load f_old: load[f_old] - d_i + d_j
                # New load f_j: load[f_j] - d_j + d_i
                new_load_f_old = current_load[f_old] - d_i + d_j
                new_load_f_j = current_load[f_j] - d_j + d_i
                
                if new_load_f_old > capacities[f_old] or new_load_f_j > capacities[f_j]:
                    continue
                
                # Calculate delta
                # Old cost part: assign_cost[i, f_old] + assign_cost[j, f_j]
                # New cost part: assign_cost[i, f_j] + assign_cost[j, f_old]
                delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_old]) - (assign_cost[i, f_old] + assign_cost[j, f_j])
                
                # Open cost change
                # f_old becomes empty if new_load_f_old == 0? 
                # Since d_j >= 5, new_load_f_old >= 5 if load[f_old]-d_i >= 0. 
                # Wait, if load[f_old] == d_i, then new_load_f_old = d_j > 0. So f_old stays open.
                # If load[f_j] == d_j, then new_load_f_j = d_i > 0. So f_j stays open.
                # So delta_open is always 0 for swaps between distinct facilities with positive demands.
                delta_open = 0.0
                
                delta = delta_assign + delta_open
                
                # Acceptance criterion
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    # Apply swap
                    current_load[f_old] = new_load_f_old
                    current_load[f_j] = new_load_f_j
                    current_assignment[i] = f_j
                    current_assignment[j] = f_old
                    current_cost += delta
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_assignment = current_assignment.copy()
                        best_load = current_load.copy()
            
            # Cool down
            T *= alpha
            if T < T_end:
                # Reheat or stop? Let's reheat to continue exploring
                T = T_start * 0.5
                # Or just stop if we want. Reheating is good for SA.
                # Let's do a mild reheat
                T = max(T, T_end * 10)

        return best_assignment, best_load, best_cost

    # Main logic
    total_time_limit = 110.0 # Leave 10s buffer
    start_time = time.time()
    
    best_overall_cost = float('inf')
    best_overall_assignment = None
    best_overall_load = None
    
    # Run multiple independent SA runs
    # Each run gets a portion of the time
    num_runs = 5
    time_per_run = total_time_limit / num_runs
    
    for run_id in range(num_runs):
        # Check time
        if time.time() - start_time > total_time_limit:
            break
            
        # Generate initial solution
        seed = 42 + run_id * 100
        assignment, load = greedy_init(seed)
        
        # Run SA
        assignment, load, cost = simulated_annealing(assignment, load, max_time=time_per_run)
        
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_assignment = assignment
            best_overall_load = load
            
        # If we have time left, we could do a final polish with local search on the best solution
        # But SA already does local search effectively.
        
    # Final polish: Run a strict local search (no random acceptance) on the best solution found
    # This ensures we are at a local optimum.
    if best_overall_assignment is not None and time.time() - start_time < total_time_limit:
        # Simple local search polish
        assignment = best_overall_assignment.copy()
        load = best_overall_load.copy()
        cost = best_overall_cost
        
        improved = True
        while improved and time.time() - start_time < total_time_limit:
            improved = False
            for i in range(n_customers):
                f_old = assignment[i]
                d_i = demands[i]
                best_delta = 0
                best_f_new = -1
                
                for f_new in sorted_facilities[i]:
                    if f_new == f_old:
                        continue
                    if load[f_new] + d_i > capacities[f_new]:
                        continue
                    
                    delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
                    delta_open = 0.0
                    if load[f_old] == d_i:
                        delta_open -= open_cost[f_old]
                    if load[f_new] == 0:
                        delta_open += open_cost[f_new]
                    
                    total_delta = delta_assign + delta_open
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_f_new = f_new
                
                if best_f_new != -1:
                    load[f_old] -= d_i
                    load[best_f_new] += d_i
                    assignment[i] = best_f_new
                    cost += best_delta
                    improved = True
            
            # 2-Swaps polish
            for i in range(n_customers):
                f_i = assignment[i]
                d_i = demands[i]
                best_delta = 0
                best_j = -1
                best_f_j = -1
                
                for j in range(i + 1, n_customers):
                    f_j = assignment[j]
                    if f_i == f_j:
                        continue
                    d_j = demands[j]
                    
                    new_load_fi = load[f_i] - d_i + d_j
                    new_load_fj = load[f_j] - d_j + d_i
                    
                    if new_load_fi > capacities[f_i] or new_load_fj > capacities[f_j]:
                        continue
                    
                    delta_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                    # delta_open is 0 for swaps
                    total_delta = delta_assign
                    
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_j = j
                        best_f_j = f_j
                
                if best_j != -1:
                    f_j = best_f_j
                    d_j = demands[best_j]
                    new_load_fi = load[f_i] - d_i + d_j
                    new_load_fj = load[f_j] - d_j + d_i
                    
                    load[f_i] = new_load_fi
                    load[f_j] = new_load_fj
                    assignment[i] = f_j
                    assignment[best_j] = f_i
                    cost += best_delta
                    improved = True
        
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_assignment = assignment
            best_overall_load = load

    return best_overall_assignment
# EVOLVE-BLOCK-END

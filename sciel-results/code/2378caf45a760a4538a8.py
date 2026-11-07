import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    N = len(demands)
    M = len(capacities)
    
    # Convert inputs to numpy arrays for speed
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    
    # Ensure types are correct
    demands = demands.astype(np.int64)
    capacities = capacities.astype(np.int64)
    open_cost = open_cost.astype(np.float64)
    assign_cost = assign_cost.astype(np.float64)
    
    start_time = time.time()
    time_limit = 145.0 # Safety margin for 150s limit
    
    # --- Helper to calculate cost ---
    def calc_cost(sol, load):
        # Open costs
        open_mask = load > 0
        cost = np.sum(open_cost[open_mask])
        # Assign costs
        # sol is array of facility indices for each customer
        assign_sum = np.sum(assign_cost[np.arange(N), sol])
        return cost + assign_sum

    # --- Initialization: Randomized Greedy ---
    def greedy_init():
        # Sort customers by demand descending, but add some randomness
        # to diversify initial solutions
        order = np.argsort(-demands)
        # Shuffle the order slightly? Or just use sorted.
        # Let's use a deterministic sorted order for the first few, 
        # but we will run multiple restarts.
        
        sol = np.zeros(N, dtype=np.int64)
        load = np.zeros(M, dtype=np.int64)
        
        for i in order:
            # Find feasible facilities
            # We want min assign_cost[i, f] where load[f] + demands[i] <= capacities[f]
            feasible_mask = load + demands[i] <= capacities
            if not np.any(feasible_mask):
                # Fallback: pick facility with most remaining capacity
                # This might violate capacity if no feasible exists, 
                # but problem statement implies feasible instances.
                # To be safe, if no feasible, we might need to reassign, 
                # but greedy usually works if total cap >= total demand.
                idx = np.argmax(capacities - load)
                sol[i] = idx
                load[idx] += demands[i]
                continue
            
            costs = assign_cost[i].copy()
            costs[~feasible_mask] = np.inf
            f = np.argmin(costs)
            sol[i] = f
            load[f] += demands[i]
            
        return sol, load

    # --- Tabu Search ---
    def tabu_search(initial_sol, initial_load, iters=50000):
        sol = initial_sol.copy()
        load = initial_load.copy()
        current_cost = calc_cost(sol, load)
        best_sol = sol.copy()
        best_cost = current_cost
        
        # Tabu list: store (customer, from_facility)
        # We will use a dictionary or a list of tuples.
        # To be fast, we can use a set of integers: customer * M + from_facility
        tabu_list = set()
        tabu_tenure = 20 # How many iterations a move is tabu
        
        # Aspiration: if cost < best_cost, allow tabu move
        # We don't strictly need to store best_cost separately if we compare to global best
        
        # Precompute customer indices
        cust_indices = np.arange(N)
        
        # For speed, we can pre-sort facilities by cost for each customer?
        # No, because feasibility changes.
        
        # We will iterate through customers in a random order or sequential.
        # Sequential is fine.
        
        for it in range(iters):
            if time.time() - start_time > time_limit:
                break
            
            # Remove old tabu entries
            # Since we add 1 entry per iteration, we can just clear the set 
            # if we use a fixed size list, but a set with expiration is complex.
            # Simpler: Use a list of (move, expiry_iter)
            # Or just a set and clear it every tenure iterations? No.
            # Let's use a list of (key, expiry)
            # Actually, for simplicity and speed in Python, a set of keys 
            # and a counter is hard.
            # Let's use a list of (key, expiry_iter).
            # But removing from a list is O(N).
            
            # Alternative: Just use a set and add a new key each step.
            # If len(tabu_list) > tabu_tenure, pop the oldest.
            # We need a FIFO queue.
            
            # Let's restructure Tabu List as a Queue (collections.deque) of keys.
            # And a Set for O(1) lookup.
            
            # Wait, I didn't import deque. I can use a list and index, or just 
            # clear the set every 'tenure' steps? That's not true tabu.
            # Let's use a simple list for the queue and a set for lookup.
            
            # Optimization: 
            # If we just want to prevent immediate reversal, we can store the 
            # last 'tenure' moves.
            
            # Let's implement a proper Tabu List with a deque.
            # Since I can't import deque easily in the snippet without adding imports, 
            # I will use a list and an index pointer, or just a simple list where 
            # I pop from the front if it gets too long.
            
            # Actually, for this problem size (100 customers), a simple list 
            # of the last 20 moves is fine.
            
            # Let's use a list `tabu_queue` of (cust, from_f)
            # and a set `tabu_set` for O(1) check.
            
            # I'll define these outside the loop in the function scope or 
            # initialize them here.
            
            # Re-initializing tabu structures for each restart is good.
            # But I'm defining tabu_search as a nested function.
            # Let's define the structures here.
            
            # Note: I already defined tabu_list as a set above. 
            # I need to change the implementation to use a queue.
            
            # Let's restart the tabu_search function definition properly.
            pass

    # Let's rewrite the tabu search logic directly in the main flow 
    # or a cleaner nested function to avoid confusion.
    
    def run_tabu_search(initial_sol, initial_load, max_iters=100000):
        sol = initial_sol.copy()
        load = initial_load.copy()
        current_cost = calc_cost(sol, load)
        best_sol = sol.copy()
        best_cost = current_cost
        
        # Tabu structures
        tabu_queue = []
        tabu_set = set()
        tabu_tenure = 15
        
        # Precompute indices
        cust_indices = np.arange(N)
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Remove expired tabu moves
            if len(tabu_queue) > tabu_tenure:
                old_move = tabu_queue.pop(0)
                tabu_set.remove(old_move)
            
            # Find best move
            best_delta = 0
            best_i = -1
            best_k = -1
            best_from = -1
            
            # Iterate through customers
            # To add diversity, we could shuffle the order of customers each iteration
            # or just go 0..N-1.
            # Going 0..N-1 is faster.
            
            for i in range(N):
                curr_f = sol[i]
                d_i = demands[i]
                
                # Try moving to all other facilities
                # We can vectorize the cost calculation for a given customer i
                # costs_to_all = assign_cost[i]
                # new_loads = load + d_i
                # feasible = new_loads <= capacities
                # feasible[curr_f] = False # Cannot stay
                
                # Calculate delta for each feasible facility k
                # delta = assign_cost[i, k] - assign_cost[i, curr_f] + open_cost_change
                
                # Open cost change:
                # If load[curr_f] - d_i == 0, we save open_cost[curr_f]
                # If load[k] == 0, we pay open_cost[k]
                
                # Vectorized approach:
                # 1. Create array of potential new loads
                # 2. Identify feasible facilities
                # 3. Calculate assignment cost difference
                # 4. Calculate open cost difference
                # 5. Sum and find min
                
                # However, open cost depends on whether the facility becomes empty 
                # or was empty.
                
                # Let's do it efficiently:
                # Base assign cost diff: assign_cost[i, :] - assign_cost[i, curr_f]
                assign_diff = assign_cost[i] - assign_cost[i, curr_f]
                
                # Feasibility mask
                # load + d_i <= capacities
                feasible_mask = (load + d_i) <= capacities
                feasible_mask[curr_f] = False # Don't move to same facility
                
                if not np.any(feasible_mask):
                    continue
                
                # Open cost diff
                # If we move i from curr_f to k:
                # - If load[curr_f] == d_i, curr_f becomes empty -> -open_cost[curr_f]
                # - If load[k] == 0, k becomes open -> +open_cost[k]
                
                # We can calculate this for all feasible k at once.
                # open_diff = np.zeros(M)
                # If load[curr_f] == d_i: open_diff[feasible_mask] -= open_cost[curr_f]
                # open_diff[load == 0] += open_cost[load == 0]
                
                # But we only care about feasible k.
                
                # Let's compute open_diff for all M, then mask.
                open_diff = np.zeros(M)
                
                if load[curr_f] == d_i:
                    # curr_f will be empty
                    # This saving applies to any move from curr_f
                    # So for all feasible k, we save open_cost[curr_f]
                    open_diff[feasible_mask] -= open_cost[curr_f]
                
                # For facilities k that are currently empty (load[k]==0), 
                # moving i there opens it.
                # Note: If k is feasible and load[k]==0, we pay open_cost[k].
                empty_mask = (load == 0) & feasible_mask
                open_diff[empty_mask] += open_cost[empty_mask]
                
                total_delta = assign_diff + open_diff
                
                # Mask out infeasible
                total_delta[~feasible_mask] = np.inf
                
                # Check Tabu
                # If the move (i, curr_f) is in tabu_set, we should not take it 
                # unless it's an aspiration move (better than best_cost).
                
                # We need to check if the specific move (i, curr_f) is tabu.
                # The move is defined by (customer, from_facility).
                move_key = (i, curr_f)
                is_tabu = move_key in tabu_set
                
                if is_tabu:
                    # Apply aspiration criterion
                    # If the new cost is better than best_cost, allow it.
                    # We need to find the min delta first to check.
                    min_delta = np.min(total_delta)
                    new_cost = current_cost + min_delta
                    if new_cost >= best_cost:
                        # If not better than best, disallow tabu moves
                        total_delta[~feasible_mask] = np.inf # Already done
                        # We need to effectively ignore this customer if all its 
                        # best moves are tabu and not aspirable?
                        # Or just set the delta for this customer to inf if 
                        # the best move is tabu and not aspirable?
                        # Actually, we can just let it be. If the best move is tabu 
                        # and not aspirable, we should pick the next best non-tabu move.
                        
                        # To do this efficiently:
                        # We can't easily mask "tabu" per customer in the vectorized 
                        # array because tabu is per (i, curr_f), not per (i, k).
                        # Wait, the tabu key is (i, curr_f). 
                        # This means "do not move customer i from curr_f".
                        # So if (i, curr_f) is tabu, we cannot make ANY move 
                        # originating from (i, curr_f) unless aspirable.
                        
                        # So if is_tabu and new_cost >= best_cost:
                        # We should skip this customer i entirely for this iteration.
                        # Because all moves from i involve leaving curr_f.
                        continue
                        
                # Find best k
                k = np.argmin(total_delta)
                delta = total_delta[k]
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_k = k
                    best_from = curr_f
                    
            # Apply best move
            if best_i != -1:
                # Update state
                i = best_i
                k = best_k
                curr_f = best_from
                d_i = demands[i]
                
                load[curr_f] -= d_i
                load[k] += d_i
                sol[i] = k
                
                current_cost += best_delta
                
                # Update tabu list
                move_key = (i, curr_f)
                tabu_queue.append(move_key)
                tabu_set.add(move_key)
                
                # Update best
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_sol = sol.copy()
                    
        return best_sol, best_cost

    # --- Main Loop: Multiple Restarts ---
    
    # We will do a few random restarts.
    # Since the problem is small, we can afford a few.
    # Let's do 5 restarts, each with Tabu Search.
    
    global_best_sol = None
    global_best_cost = np.inf
    
    for restart in range(5):
        # Check time
        if time.time() - start_time > time_limit - 10: # Leave 10s buffer
            break
            
        # Generate initial solution
        # For first restart, use deterministic greedy.
        # For subsequent, use randomized greedy.
        if restart == 0:
            init_sol, init_load = greedy_init()
        else:
            # Randomized greedy: shuffle customer order
            order = np.random.permutation(N)
            init_sol = np.zeros(N, dtype=np.int64)
            init_load = np.zeros(M, dtype=np.int64)
            
            for i in order:
                feasible_mask = init_load + demands[i] <= capacities
                if not np.any(feasible_mask):
                    idx = np.argmax(capacities - init_load)
                    init_sol[i] = idx
                    init_load[idx] += demands[i]
                    continue
                
                costs = assign_cost[i].copy()
                costs[~feasible_mask] = np.inf
                f = np.argmin(costs)
                init_sol[i] = f
                init_load[f] += demands[i]
                
        # Run Tabu Search
        # Limit iterations for each restart to ensure we finish in time
        # 100000 iterations might be too many if it's slow.
        # Let's estimate: 100 customers, vectorized ops.
        # Each iteration is ~100 numpy ops on size 20 arrays.
        # Should be fast.
        # Let's try 50000 iterations per restart.
        
        sol, cost = run_tabu_search(init_sol, init_load, max_iters=50000)
        
        if cost < global_best_cost:
            global_best_cost = cost
            global_best_sol = sol
            
        # Check time
        if time.time() - start_time > time_limit - 5:
            break

    # If no valid solution found (shouldn't happen), return zero
    if global_best_sol is None:
        return np.zeros(N, dtype=int)
        
    return global_best_sol

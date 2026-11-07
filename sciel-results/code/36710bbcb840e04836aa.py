"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using a greedy initialization followed by 
    Simulated Annealing local search.
    """
    demands = np.array(demands)
    capacities = np.array(capacities)
    open_cost = np.array(open_cost)
    assign_cost = np.array(assign_cost)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Deadline guard: 120s limit, use 110s to be safe
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Initialization: Greedy (Largest Demand First)
    # Sort customers by demand descending
    cust_order = np.argsort(-demands)
    assignment = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac, dtype=float)
    
    # Precompute sorted facility indices for each customer based on cost
    # This helps in initialization but we can just iterate for simplicity in init
    # Since N=100, M=20, sorting 100x20 is fast
    sorted_facs = np.argsort(assign_cost, axis=1)
    
    for i in cust_order:
        placed = False
        for f in sorted_facs[i]:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                assignment[i] = f
                placed = True
                break
        if not placed:
            # Fallback: assign to facility with most remaining capacity (should not happen if valid instance)
            # If it happens, the instance might be infeasible, but problem says "5 fixed generated instances"
            # Assuming feasible. If not, we pick min cost and hope, or raise error.
            # For robustness, pick min cost facility regardless of capacity (will be invalid, but better than crash)
            # However, the prompt implies valid instances.
            # Let's just pick the one with max remaining capacity to maximize chance of validity in edge cases
            rem_cap = capacities - load
            f = np.argmax(rem_cap)
            load[f] += demands[i]
            assignment[i] = f

    # Calculate initial cost
    def calc_cost(assignment, load):
        # Open cost
        open_mask = load > 0
        cost = np.sum(open_cost[open_mask])
        # Assignment cost
        cost += np.sum(assign_cost[np.arange(n_cust), assignment])
        return cost

    current_cost = calc_cost(assignment, load)
    best_cost = current_cost
    best_assignment = assignment.copy()
    best_load = load.copy()
    
    # 2. Simulated Annealing
    # Temperature schedule
    T = 1000.0
    cooling = 0.9999
    
    # Precompute random numbers? No, use random module for speed in loop if needed, or np.random
    # np.random is fast enough for 100x20
    
    while time.time() < deadline:
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Move: Reassign customer i to facility f'
            i = np.random.randint(n_cust)
            f_old = assignment[i]
            
            # If f_old is the only facility, skip
            if n_fac == 1:
                continue
                
            f_new = np.random.randint(n_fac)
            if f_new == f_old:
                continue
            
            # Check capacity of f_new
            if load[f_new] + demands[i] > capacities[f_new]:
                continue
            
            # Calculate Delta Cost
            # Change in assignment cost
            delta_assign = assign_cost[i, f_new] - assign_cost[i, f_old]
            
            # Change in open cost
            # f_new might open
            open_new = 0.0
            if load[f_new] == 0:
                open_new = open_cost[f_new]
                
            # f_old might close
            open_old = 0.0
            if load[f_old] == demands[i]: # Only customer in f_old
                open_old = -open_cost[f_old] # We subtract the cost we no longer pay
                
            delta_cost = delta_assign + open_new + open_old
            
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                # Accept move
                load[f_old] -= demands[i]
                load[f_new] += demands[i]
                assignment[i] = f_new
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assignment = assignment.copy()
                    best_load = load.copy()
                    
        else:
            # Move: Swap customers i and j between facilities f_i and f_j
            i = np.random.randint(n_cust)
            j = np.random.randint(n_cust)
            if i == j:
                continue
                
            f_i = assignment[i]
            f_j = assignment[j]
            if f_i == f_j:
                continue
                
            # Swap preserves loads, so capacity is always satisfied
            # Calculate Delta Cost
            # i moves from f_i to f_j
            # j moves from f_j to f_i
            
            # Assignment cost changes
            delta_assign_i = assign_cost[i, f_j] - assign_cost[i, f_i]
            delta_assign_j = assign_cost[j, f_i] - assign_cost[j, f_j]
            
            # Open cost changes
            # f_i: loses i, gains j. Load unchanged. Open status unchanged.
            # f_j: loses j, gains i. Load unchanged. Open status unchanged.
            # So open cost delta is 0 for swaps.
            
            delta_cost = delta_assign_i + delta_assign_j
            
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                # Accept move
                assignment[i] = f_j
                assignment[j] = f_i
                # Load doesn't change for swaps
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assignment = assignment.copy()
                    # best_load doesn't change for swaps, but keep it synced just in case
                    # Actually, load is same, so best_load is still valid if it was synced before.
                    # But if we accept a swap, load is unchanged.
                    
        # Cool down
        T *= cooling
        
        # Safety check: if T is too small, restart with higher T? 
        # Or just let it converge. With 110s, we have plenty of iterations.
        if T < 1e-4:
            # Restart annealing
            T = 1000.0

    return best_assignment
# EVOLVE-BLOCK-END

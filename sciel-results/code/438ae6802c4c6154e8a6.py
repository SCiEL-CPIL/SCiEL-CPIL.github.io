import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities to minimize total cost.
    Uses a greedy initialization followed by local search (moves and swaps).
    """
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Deadline guard: 145 seconds to be safe within the 150s limit
    deadline = time.time() + 145.0
    
    # 1. Greedy Initialization
    # Sort customers by demand descending to place large ones first
    # Tie-break by minimum possible assignment cost to prioritize "hard" to place or expensive customers?
    # Actually, standard LDF (Largest Demand First) is good.
    order = np.argsort(-demands)
    
    assignment = np.full(n_customers, -1, dtype=int)
    load = np.zeros(n_facilities, dtype=float)
    
    # Precompute sorted facility indices for each customer based on cost
    # This is O(N*M log M) which is fine for N=100, M=20
    sorted_facilities = np.argsort(assign_cost, axis=1)
    
    for c in order:
        best_f = -1
        for f in sorted_facilities[c]:
            if load[f] + demands[c] <= capacities[f]:
                best_f = f
                break
        
        if best_f == -1:
            # If no facility can take this customer with the current greedy logic,
            # we might have a problem. But usually capacities are sufficient.
            # If strictly infeasible, we must pick the one with most remaining capacity?
            # The problem statement implies a feasible solution exists.
            # However, greedy might paint itself into a corner.
            # For robustness, if best_f == -1, we try any facility with space, 
            # or if none, we fail (should not happen if total capacity >= total demand).
            # Let's just pick the facility with max remaining capacity as a fallback.
            rem_cap = capacities - load
            best_f = np.argmax(rem_cap)
            
        assignment[c] = best_f
        load[best_f] += demands[c]
        
        if time.time() > deadline:
            return assignment.astype(int)

    # 2. Local Search
    # We track the current total cost
    def calc_cost(assignment, load):
        # Assignment cost
        idx_c = np.arange(n_customers)
        idx_f = assignment
        assign_total = np.sum(assign_cost[idx_c, idx_f])
        
        # Open cost
        # A facility is open if load > 0
        open_mask = load > 1e-9
        open_total = np.sum(open_cost[open_mask])
        
        return assign_total + open_total

    current_cost = calc_cost(assignment, load)
    
    # Precompute customer indices for faster access
    customers = np.arange(n_customers)
    
    # Local Search Loop
    # We alternate between single moves and pair swaps.
    # Single move: customer c from f_old to f_new
    # Swap: customer c1 from f1 to f2, customer c2 from f2 to f1
    
    improved = True
    # To avoid getting stuck in very shallow local minima, we can do multiple passes
    # until no improvement is found or time runs out.
    
    while improved and time.time() < deadline:
        improved = False
        pass_count = 0
        
        # --- Pass 1: Single Moves ---
        # Iterate customers in random order to break symmetry
        c_order = np.random.permutation(n_customers)
        for c in c_order:
            if time.time() > deadline - 5.0: # Safety margin for loop overhead
                break
                
            f_old = assignment[c]
            d_c = demands[c]
            
            # Try all other facilities
            # We can skip f_old
            for f_new in range(n_facilities):
                if f_new == f_old:
                    continue
                
                # Check capacity
                if load[f_new] + d_c > capacities[f_new]:
                    continue
                
                # Calculate cost change
                # Delta Assignment: cost(c, f_new) - cost(c, f_old)
                delta_assign = assign_cost[c, f_new] - assign_cost[c, f_old]
                
                # Delta Open Cost:
                # 1. If f_old becomes empty: -open_cost[f_old]
                # 2. If f_new was empty: +open_cost[f_new]
                delta_open = 0.0
                if load[f_old] == d_c: # f_old becomes empty
                    delta_open -= open_cost[f_old]
                if load[f_new] == 0.0: # f_new becomes open
                    delta_open += open_cost[f_new]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < -1e-9:
                    # Accept move
                    assignment[c] = f_new
                    load[f_old] -= d_c
                    load[f_new] += d_c
                    current_cost += total_delta
                    improved = True
                    break # Move to next customer
    
        # --- Pass 2: Pair Swaps ---
        # This is more expensive O(N^2 * M) but can find better solutions.
        # We limit this to a subset or run it less frequently if time is tight.
        # Given N=100, N^2=10000. M=20. 200k checks per pass. Very fast.
        
        # Randomize customer order for swaps to avoid bias
        c_order = np.random.permutation(n_customers)
        
        for i in range(n_customers):
            if time.time() > deadline - 10.0:
                break
            c1 = c_order[i]
            f1 = assignment[c1]
            d1 = demands[c1]
            
            for j in range(i + 1, n_customers):
                c2 = c_order[j]
                f2 = assignment[c2]
                d2 = demands[c2]
                
                if f1 == f2:
                    continue
                
                # Check capacity for swap
                # f1 loses d1, gains d2
                # f2 loses d2, gains d1
                if load[f1] - d1 + d2 > capacities[f1]:
                    continue
                if load[f2] - d2 + d1 > capacities[f2]:
                    continue
                
                # Calculate cost change
                # Assignment:
                # c1: f1 -> f2
                # c2: f2 -> f1
                delta_assign = (assign_cost[c1, f2] - assign_cost[c1, f1]) + \
                               (assign_cost[c2, f1] - assign_cost[c2, f2])
                
                # Open Cost:
                # Check if f1 or f2 become empty
                # f1 new load: load[f1] - d1 + d2
                # f2 new load: load[f2] - d2 + d1
                
                new_load_f1 = load[f1] - d1 + d2
                new_load_f2 = load[f2] - d2 + d1
                
                delta_open = 0.0
                # If f1 was open and becomes empty
                if load[f1] > 1e-9 and new_load_f1 <= 1e-9:
                    delta_open -= open_cost[f1]
                # If f1 was empty and becomes open (impossible in swap if f1 had c1)
                # Wait, if f1 had c1, it was open. It can only become empty or stay open.
                
                # If f2 was open and becomes empty
                if load[f2] > 1e-9 and new_load_f2 <= 1e-9:
                    delta_open -= open_cost[f2]
                
                # Note: In a swap, facilities don't go from empty to open 
                # because they already had at least one customer (c1 in f1, c2 in f2).
                # So we only subtract open costs if they become empty.
                
                total_delta = delta_assign + delta_open
                
                if total_delta < -1e-9:
                    # Accept swap
                    assignment[c1] = f2
                    assignment[c2] = f1
                    load[f1] = new_load_f1
                    load[f2] = new_load_f2
                    current_cost += total_delta
                    improved = True
                    # We can break the inner loop and continue with next i, 
                    # or continue searching for more improvements.
                    # Breaking is safer for speed, but might miss better swaps.
                    # Let's break to keep the pass fast and rely on multiple passes.
                    break
        
        # If no improvement in a full pass, we might be in a local minimum.
        # We can try a "kick" (random perturbation) to escape, but it's risky.
        # Given the small size, simple local search is often sufficient.
        # If improved is False, we exit the while loop.

    return assignment.astype(int)

# EVOLVE-BLOCK-END

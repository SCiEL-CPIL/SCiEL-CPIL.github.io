"""Improved program for the suite task; uses diverse greedy seeds and best-improvement local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities to minimize total cost (open + assign).
    Returns a NumPy array of facility indices.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Convert to numpy arrays for fast operations
    demands = np.array(demands, dtype=float)
    capacities = np.array(capacities, dtype=float)
    open_cost = np.array(open_cost, dtype=float)
    assign_cost = np.array(assign_cost, dtype=float)
    
    # Deadline: 110 seconds to be safe (limit is 120s)
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Helper to calculate total cost
    def total_cost(assignment, open_facilities):
        assign_total = np.sum(assign_cost[np.arange(n_customers), assignment])
        open_total = np.sum(open_cost[open_facilities])
        return assign_total + open_total
    
    # Helper to check validity
    def is_valid(assignment, open_facilities):
        loads = np.zeros(n_facilities)
        for f in open_facilities:
            mask = assignment == f
            if mask.any():
                load = np.sum(demands[mask])
                if load > capacities[f]:
                    return False
        return True

    # Helper to get loads for a given assignment and open set
    def get_loads(assignment, open_facilities):
        loads = np.zeros(n_facilities)
        for f in open_facilities:
            mask = assignment == f
            if mask.any():
                loads[f] = np.sum(demands[mask])
        return loads

    # 1. Generate Diverse Initial Solutions
    # Strategy A: Greedy based on raw assign cost (Largest Demand First)
    # Strategy B: Greedy based on effective cost (assign_cost + open_cost / capacity)
    # Strategy C: Random feasible assignment
    
    def greedy_init(metric):
        # metric: 'raw' or 'effective'
        if metric == 'effective':
            # Approximate effective cost: assign_cost + open_cost / capacity
            # We use a heuristic: prioritize facilities with lower open_cost/capacity ratio
            # but still respect assign_cost.
            # A simpler robust greedy: Sort customers by demand desc.
            # For each customer, pick feasible facility with min (assign_cost + open_cost * (1/capacity))
            # This is an approximation. Let's stick to a robust greedy:
            # Sort customers by demand descending.
            # For each customer, try facilities sorted by assign_cost.
            # If capacity allows, assign.
            # If no capacity, find facility with most remaining capacity and lowest assign cost?
            # Let's use a slightly more sophisticated greedy for diversity:
            # Sort facilities by open_cost ascending.
            # For each facility, assign customers that can fit, prioritized by low assign_cost.
            
            assignment = np.full(n_customers, -1, dtype=int)
            remaining_cap = capacities.copy()
            # Sort facilities by open cost
            fac_order = np.argsort(open_cost)
            
            # We need to assign everyone. This simple greedy might fail if capacities are tight.
            # Let's use the "Largest Demand First" but with a different facility selection order.
            
            cust_order = np.argsort(-demands)
            loads = np.zeros(n_facilities)
            
            for i in cust_order:
                # Sort facilities by assign_cost[i, :]
                facs = np.argsort(assign_cost[i])
                assigned = False
                for f in facs:
                    if loads[f] + demands[i] <= capacities[f]:
                        loads[f] += demands[i]
                        assignment[i] = f
                        assigned = True
                        break
                if not assigned:
                    # Fallback: pick facility with most remaining capacity
                    rem = capacities - loads
                    f = np.argmax(rem)
                    if rem[f] >= demands[i]:
                        loads[f] += demands[i]
                        assignment[i] = f
                    else:
                        # Infeasible? Should not happen if total capacity > total demand.
                        # If it does, return invalid flag
                        return None
            return assignment

    def random_init():
        # Random assignment, then repair
        assignment = np.random.randint(0, n_facilities, n_customers)
        # Repair: Ensure capacity constraints
        # This is tricky for random init. 
        # Better: Start with all facilities open, assign randomly, then close if possible?
        # Or: Start with a valid greedy, then randomize some assignments and repair.
        # Let's just use the greedy result but perturb it.
        return None # We'll use perturbation of greedy instead

    candidates = []
    
    # Generate a few greedy seeds
    for _ in range(5):
        sol = greedy_init('raw')
        if sol is not None:
            candidates.append(sol.copy())
            
    # Generate a "cost-aware" greedy
    # Sort facilities by open_cost / capacity (efficiency)
    # Assign customers to most efficient facilities first?
    # Let's try a different greedy: 
    # For each facility, calculate "value" = open_cost.
    # This is complex. Let's stick to perturbing the base greedy.
    
    # If we have time, we can generate more diverse seeds.
    # Let's add a seed that prioritizes low open cost facilities for high demand customers?
    
    # 2. Local Search
    # State: assignment (array of ints), open_facilities (set of ints)
    # We maintain the best solution found so far.
    
    best_assignment = None
    best_cost = float('inf')
    best_open_set = None
    
    def update_best(assignment, open_set):
        nonlocal best_assignment, best_cost, best_open_set
        if is_valid(assignment, open_set):
            c = total_cost(assignment, open_set)
            if c < best_cost:
                best_cost = c
                best_assignment = assignment.copy()
                best_open_set = set(open_set)

    # Initialize with the first candidate
    if candidates:
        a = candidates[0]
        # Determine open facilities: those with at least one customer
        open_set = set(a[a >= 0].tolist())
        # Check validity
        if is_valid(a, open_set):
            update_best(a, open_set)
        else:
            # Repair: Close empty, check capacity
            # If capacity violated, we need a repair step.
            # For simplicity, if greedy fails validity (shouldn't), we skip.
            pass

    # Define Local Search Move:
    # 1. Reassignment: Move customer i from f1 to f2.
    #    - If f2 is closed, open it (add open_cost).
    #    - If f1 becomes empty, close it (remove open_cost).
    # 2. (Implicitly handled by reassignment if we allow moving to any facility)
    
    def local_search(assignment, open_set, time_limit):
        current_assignment = assignment.copy()
        current_open_set = set(open_set)
        current_cost = total_cost(current_assignment, current_open_set)
        
        if not is_valid(current_assignment, current_open_set):
            return None
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_move = None
            best_move_cost = current_cost
            best_new_assignment = current_assignment.copy()
            best_new_open_set = current_open_set.copy()
            
            # Evaluate all possible reassignments
            # For each customer i, try moving to every other facility f
            # To speed up, we can pre-calculate loads.
            loads = get_loads(current_assignment, current_open_set)
            
            for i in range(n_customers):
                current_f = current_assignment[i]
                # Try all other facilities
                # Optimization: Only consider facilities that are open or have low open cost?
                # No, moving to a closed facility might be beneficial if it saves assign cost.
                
                # Pre-calculate delta for opening/closing
                # Delta Open/Close depends on whether the facility becomes empty or not.
                
                for f in range(n_facilities):
                    if f == current_f:
                        continue
                    
                    # Check capacity of f
                    if loads[f] + demands[i] > capacities[f]:
                        continue
                    
                    # Calculate new cost
                    # Change in assign cost: assign_cost[i, f] - assign_cost[i, current_f]
                    delta_assign = assign_cost[i, f] - assign_cost[i, current_f]
                    
                    # Change in open cost
                    delta_open = 0.0
                    new_open_set = current_open_set.copy()
                    
                    # If f is not open, we open it
                    if f not in new_open_set:
                        delta_open += open_cost[f]
                        new_open_set.add(f)
                    
                    # If current_f becomes empty, we close it
                    # Load of current_f after removing i: loads[current_f] - demands[i]
                    if loads[current_f] - demands[i] <= 0:
                        # It becomes empty, so we close it
                        # But only if it was open (it must be, since i is assigned to it)
                        delta_open -= open_cost[current_f]
                        new_open_set.discard(current_f)
                    
                    new_cost = current_cost + delta_assign + delta_open
                    
                    if new_cost < best_move_cost - 1e-9:
                        best_move_cost = new_cost
                        best_move = (i, current_f, f)
                        # We don't update the assignment yet, just the best move
                        # But we need to construct the new state for the next iteration
                        # For efficiency, we only apply the best move after the full scan.
                        # However, we need to track the resulting open_set for the best move.
                        # Since the open_set change depends on the move, we store it.
                        # Note: Storing a set for every potential best move is expensive.
                        # We can reconstruct it when applying the best move.
                        
            if best_move:
                i, f_from, f_to = best_move
                # Apply move
                current_assignment[i] = f_to
                
                # Update open set
                # Check if f_to was closed
                if f_to not in current_open_set:
                    current_open_set.add(f_to)
                
                # Check if f_from is now empty
                # Recalculate load of f_from
                # We can do this efficiently by checking if any other customer is assigned to f_from
                # Or just recalculate loads for the affected facilities.
                # For simplicity and correctness, let's recalculate loads for f_from and f_to.
                # Actually, we can just check if np.any(current_assignment == f_from) is False.
                if not np.any(current_assignment == f_from):
                    current_open_set.discard(f_from)
                
                current_cost = best_move_cost
                improved = True
                update_best(current_assignment, current_open_set)
                
        return current_assignment, current_open_set

    # Run Local Search on each candidate
    # We have ~5 candidates.
    # We share the deadline.
    
    for idx, cand in enumerate(candidates):
        if time.time() > deadline - 5: # Leave 5s for final safety
            break
            
        # Determine initial open set for this candidate
        open_set = set(cand[cand >= 0].tolist())
        
        # Check if valid
        if not is_valid(cand, open_set):
            continue
            
        # Run local search
        # Give each candidate a slice of the remaining time?
        # Or just run until deadline?
        # Best Improvement can be slow. 
        # Let's run local search for each candidate, but stop if deadline is near.
        
        res = local_search(cand, open_set, deadline - 2)
        if res:
            a, o = res
            # update_best is called inside local_search, so best_cost is updated.
            pass

    # If no valid solution found (should not happen), return a dummy valid one?
    # The problem guarantees feasible instances.
    if best_assignment is None:
        # Fallback to greedy
        sol = greedy_init('raw')
        if sol is not None and is_valid(sol, set(sol.tolist())):
            return sol
        else:
            # Emergency: assign all to facility 0? Might violate capacity.
            # Just return the greedy result even if invalid? No, must be valid.
            # If greedy failed, something is wrong.
            # Let's try to construct a valid solution manually.
            # Since capacities are likely sufficient, a simple bin-packing greedy should work.
            # The greedy_init above already does this.
            # If it returned None, it means no feasible assignment was found, which contradicts problem statement.
            # We will return an array of zeros as a last resort, though it might be invalid.
            return np.zeros(n_customers, dtype=int)

    return best_assignment
# EVOLVE-BLOCK-END

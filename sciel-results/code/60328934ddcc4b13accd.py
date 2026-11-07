"""Capacitated facility location via Multi-Start Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n = len(demands)
    m = len(capacities)
    
    # Precompute assignment costs for faster access
    # assign_cost is (n, m)
    
    def compute_cost(assignment):
        """Compute total cost for a given assignment."""
        total_assign = np.sum(assign_cost[np.arange(n), assignment])
        used_facilities = np.unique(assignment)
        total_open = np.sum(open_cost[used_facilities])
        return total_assign + total_open
    
    def is_valid(assignment):
        """Check if assignment satisfies capacity constraints."""
        loads = np.zeros(m)
        for c in range(n):
            loads[assignment[c]] += demands[c]
        return np.all(loads <= capacities + 1e-9)
    
    def greedy_init(seed=None):
        """Greedy initialization: assign largest demands first to cheapest feasible facility."""
        if seed is not None:
            random.seed(seed)
        
        assignment = np.zeros(n, dtype=int)
        loads = np.zeros(m)
        
        # Sort customers by demand descending
        order = np.argsort(-demands)
        
        for c in order:
            # Get facilities sorted by assignment cost for this customer
            fac_order = np.argsort(assign_cost[c])
            placed = False
            for f in fac_order:
                if loads[f] + demands[c] <= capacities[f] + 1e-9:
                    assignment[c] = f
                    loads[f] += demands[c]
                    placed = True
                    break
            if not placed:
                # If no facility has capacity, assign to least loaded (will be invalid, but we'll fix later or restart)
                # For greedy init, we assume a feasible solution exists. If not, we need a repair step.
                # Let's try to find the facility with most remaining capacity
                remaining_cap = capacities - loads
                f = np.argmax(remaining_cap)
                assignment[c] = f
                loads[f] += demands[c]
        
        return assignment
    
    def random_init():
        """Random initialization with repair."""
        assignment = np.random.randint(0, m, n)
        # Repair: ensure capacity constraints are met
        # This is a simple repair: repeatedly move customers from over-capacity facilities to under-capacity ones
        # We'll use a more robust repair in the local search if needed, but for init, let's try a quick fix.
        
        # A better random init: assign randomly, then fix violations
        loads = np.zeros(m)
        for c in range(n):
            loads[assignment[c]] += demands[c]
        
        # Fix violations
        for _ in range(1000): # Max iterations for repair
            violated = np.where(loads > capacities + 1e-9)[0]
            if len(violated) == 0:
                break
            
            # Pick a random violated facility
            f_bad = violated[random.randint(0, len(violated)-1)]
            
            # Find a customer to move from f_bad
            custs_at_bad = np.where(assignment == f_bad)[0]
            if len(custs_at_bad) == 0:
                continue
                
            c = custs_at_bad[random.randint(0, len(custs_at_bad)-1)]
            
            # Find a facility to move to
            # Prefer facilities with most remaining capacity
            remaining_cap = capacities - loads
            # Exclude f_bad
            remaining_cap[f_bad] = -1
            
            # Try facilities in order of remaining capacity
            fac_order = np.argsort(-remaining_cap)
            moved = False
            for f in fac_order:
                if loads[f] + demands[c] <= capacities[f] + 1e-9:
                    # Move c to f
                    assignment[c] = f
                    loads[f_bad] -= demands[c]
                    loads[f] += demands[c]
                    moved = True
                    break
            
            if not moved:
                # If we can't move anyone, this init is invalid. Return a dummy valid solution?
                # Or just return the greedy one.
                return greedy_init()
        
        return assignment
    
    def local_search(assignment, time_limit=10.0):
        """Refine assignment using local search (moves)."""
        assignment = assignment.copy()
        loads = np.zeros(m)
        for c in range(n):
            loads[assignment[c]] += demands[c]
        
        current_cost = compute_cost(assignment)
        best_cost = current_cost
        best_assignment = assignment.copy()
        
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() - start_time < DEADLINE - 5.0:
            improved = False
            
            # Shuffle customer order to avoid bias
            cust_order = np.random.permutation(n)
            
            for c in cust_order:
                if time.time() - start_time > DEADLINE - 5.0:
                    break
                
                current_f = assignment[c]
                current_load_f = loads[current_f]
                
                # Try moving c to all other facilities
                # Sort facilities by assignment cost for c
                fac_order = np.argsort(assign_cost[c])
                
                for f in fac_order:
                    if f == current_f:
                        continue
                    
                    # Check capacity
                    if loads[f] + demands[c] <= capacities[f] + 1e-9:
                        # Calculate cost difference
                        # Cost change: 
                        # - Remove c from current_f: -assign_cost[c, current_f]
                        # - Add c to f: +assign_cost[c, f]
                        # - Open cost change:
                        #   If current_f becomes empty: -open_cost[current_f]
                        #   If f was empty: +open_cost[f]
                        
                        delta_assign = assign_cost[c, f] - assign_cost[c, current_f]
                        delta_open = 0.0
                        
                        # Check if current_f becomes empty
                        if current_load_f == demands[c]:
                            delta_open -= open_cost[current_f]
                        
                        # Check if f was empty
                        if loads[f] == 0:
                            delta_open += open_cost[f]
                        
                        delta_total = delta_assign + delta_open
                        
                        if delta_total < -1e-9: # Improvement
                            # Perform move
                            loads[current_f] -= demands[c]
                            loads[f] += demands[c]
                            assignment[c] = f
                            
                            current_cost += delta_total
                            improved = True
                            no_improve_count = 0
                            
                            # If we made a move, break to restart the loop for this customer
                            # Or continue to see if another move is better? 
                            # First-improvement is faster.
                            break
                
                # Early exit if no improvements in a long time
                if not improved and no_improve_count > max_no_improve:
                    break
            else:
                # If we went through all customers without breaking the inner loop
                if not improved:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                
                # Update best
                if current_cost < best_cost - 1e-9:
                    best_cost = current_cost
                    best_assignment = assignment.copy()
                
                if no_improve_count > max_no_improve:
                    break
            # If we broke out of the customer loop due to time or move, we continue the while loop
            
            # If we didn't improve in the entire pass, we can stop
            if not improved:
                break
        
        return best_assignment, best_cost
    
    # Multi-start local search
    best_overall_cost = float('inf')
    best_overall_assignment = None
    
    # Start with greedy
    init_assign = greedy_init()
    if is_valid(init_assign):
        refined_assign, refined_cost = local_search(init_assign)
        if refined_cost < best_overall_cost:
            best_overall_cost = refined_cost
            best_overall_assignment = refined_assign
    
    # Try several random starts
    num_starts = 0
    while time.time() - start_time < DEADLINE - 10.0:
        num_starts += 1
        
        # Random init
        init_assign = random_init()
        
        if not is_valid(init_assign):
            continue
        
        # Refine
        refined_assign, refined_cost = local_search(init_assign)
        
        if refined_cost < best_overall_cost:
            best_overall_cost = refined_cost
            best_overall_assignment = refined_assign
    
    # If no valid solution found (should not happen), return greedy
    if best_overall_assignment is None:
        best_overall_assignment = greedy_init()
    
    return best_overall_assignment
# EVOLVE-BLOCK-END

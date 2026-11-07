"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Simulated Annealing with Random Restarts.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Pre-compute total demand and check feasibility
    total_demand = np.sum(demands)
    total_capacity = np.sum(capacities)
    
    if total_demand > total_capacity:
        # Infeasible, return a dummy valid-looking array (though it will fail capacity)
        # But the prompt says "invalid result... scores 0". 
        # We should try to return the best valid one we can find, or just return zeros.
        # Assuming instances are feasible.
        return np.zeros(n_cust, dtype=int)

    # 1. Greedy Initialization
    def greedy_init():
        """
        Greedy assignment: Largest demand first, cheapest feasible facility.
        Returns assignment array and load array.
        """
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        # Sort customers by demand descending
        order = np.argsort(-demands)
        
        for i in order:
            d = demands[i]
            # Find facilities with enough capacity
            # We want the one with min assign_cost[i, f]
            # To speed up, we can just iterate through facilities sorted by cost for this customer
            # But n_fac is small (20), so sorting 20 items is fine.
            fac_indices = np.argsort(assign_cost[i])
            assigned = False
            for f in fac_indices:
                if load[f] + d <= capacities[f]:
                    result[i] = f
                    load[f] += d
                    assigned = True
                    break
            if not assigned:
                # Fallback: assign to facility with most remaining capacity (might exceed, but greedy usually works)
                # If greedy fails, we have a problem. Let's just pick max remaining capacity.
                rem = capacities - load
                f = np.argmax(rem)
                result[i] = f
                load[f] += d
        return result, load

    # 2. Cost Calculation (Vectorized)
    def calc_cost(assignments):
        """
        Calculates total cost: Open Costs + Assignment Costs + Penalty for Capacity Violation.
        """
        # Assignment costs: sum of assign_cost[i, assignments[i]]
        # Use advanced indexing
        assign_costs = np.sum(assign_cost[np.arange(n_cust), assignments])
        
        # Open costs: sum of open_cost[f] for f in set(assignments)
        unique_facs = np.unique(assignments)
        open_costs = np.sum(open_cost[unique_facs])
        
        # Capacity check & Penalty
        # Calculate load for each facility
        # bincount is fast
        loads = np.bincount(assignments, minlength=n_fac)
        
        # Violations: max(0, loads - capacities)
        # We add a heavy penalty to make SA avoid these states
        # Penalty weight should be high enough to make violations "expensive"
        # Max possible assign cost is roughly 100 * sqrt(2) * 100 (dist) ~ 14000 per customer? 
        # No, assign_cost is 100 * dist. Max dist in unit square is ~1.41. So max cost ~141.
        # 100 customers * 141 = 14100 total assign cost.
        # Open cost max 150 * 20 = 3000.
        # Total legit cost < 20000.
        # If we exceed capacity by 1 unit, penalty should be > 20000 to force avoidance.
        # Let's use a penalty of 1000 per unit of excess capacity.
        violations = np.maximum(0, loads - capacities)
        penalty = 1000.0 * np.sum(violations)
        
        return assign_costs + open_costs + penalty

    # 3. Simulated Annealing
    def simulate_annealing(init_assign, init_load):
        current_assign = init_assign.copy()
        current_cost = calc_cost(current_assign)
        best_assign = current_assign.copy()
        best_cost = current_cost
        
        # SA Parameters
        # T0 = 1000, T_end = 1, alpha = 0.999?
        # Or time-based cooling.
        # Let's use a fixed number of iterations or time-based.
        # Since we have a deadline, we run until time runs out.
        
        T = 100.0
        T_min = 1e-6
        alpha = 0.9995 # Cooling factor
        
        # We need to update loads efficiently.
        # Maintaining a load array is faster than bincount every time if we update incrementally.
        # But bincount is C-optimized. For n=100, bincount is very fast.
        # Let's stick with bincount for simplicity and correctness, unless it's too slow.
        
        while True:
            if time.time() > deadline:
                break
            
            # Generate neighbor
            # Pick a random customer
            i = np.random.randint(n_cust)
            old_f = current_assign[i]
            
            # Pick a new facility
            # Strategy: Pick a random facility, or a facility near the current one?
            # Random is fine.
            new_f = np.random.randint(n_fac)
            if old_f == new_f:
                continue
                
            # Apply move
            current_assign[i] = new_f
            
            # Calculate new cost
            new_cost = calc_cost(current_assign)
            
            # Delta
            delta = new_cost - current_cost
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta < 0:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_assign = current_assign.copy()
            else:
                # Probabilistic acceptance
                # Avoid overflow in exp
                if delta < 500: # T is small, delta large -> prob ~ 0
                    prob = np.exp(-delta / T)
                    if np.random.rand() < prob:
                        current_cost = new_cost
                    else:
                        # Revert
                        current_assign[i] = old_f
                else:
                    # Revert
                    current_assign[i] = old_f
            
            # Cool down
            T *= alpha
            if T < T_min:
                break
                
        return best_assign, best_cost

    # 4. Main Loop: Random Restarts
    # We will perform multiple restarts.
    # Each restart: Greedy Init -> SA -> Keep best.
    
    best_overall_assign = None
    best_overall_cost = np.inf
    
    # Run at least one full SA
    init_assign, init_load = greedy_init()
    sa_assign, sa_cost = simulate_annealing(init_assign, init_load)
    
    if sa_cost < best_overall_cost:
        best_overall_cost = sa_cost
        best_overall_assign = sa_assign
        
    # Check time
    if time.time() > deadline - 5:
        return best_overall_assign
        
    # Try more restarts if time permits
    # Greedy init is deterministic based on order, so we need to randomize the tie-breaking or order?
    # The greedy init above sorts by demand. Ties are broken by index order (stable sort).
    # To get diverse inits, we can shuffle customers with equal demand, or add small random noise to costs.
    # Simpler: Just run SA again from the same init? No, SA is stochastic.
    # But restarting from the *same* init with different random seeds in SA is effectively just more SA steps.
    # A true restart would need a different init.
    
    # Let's modify greedy_init to accept a noise factor or randomization.
    # Or, we can just run SA multiple times. The stochasticity of SA provides the restarts.
    # However, if SA gets stuck, a different init helps.
    
    # Let's create a randomized greedy init.
    def randomized_greedy_init():
        result = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac)
        order = np.argsort(-demands)
        
        # Add small random noise to costs to break ties randomly
        # Or just shuffle the order of facilities considered?
        # Let's shuffle the facility order for each customer.
        
        for i in order:
            d = demands[i]
            # Get indices sorted by cost, then shuffle ties?
            # Actually, just pick the best feasible one, but if multiple are close, pick random?
            # Simpler: Sort facilities by cost + small random noise.
            noise = np.random.rand(n_fac) * 0.01
            scores = assign_cost[i] + noise
            fac_indices = np.argsort(scores)
            
            assigned = False
            for f in fac_indices:
                if load[f] + d <= capacities[f]:
                    result[i] = f
                    load[f] += d
                    assigned = True
                    break
            if not assigned:
                rem = capacities - load
                f = np.argmax(rem)
                result[i] = f
                load[f] += d
        return result, load

    # Loop for more restarts
    while time.time() < deadline - 10:
        init_assign, init_load = randomized_greedy_init()
        sa_assign, sa_cost = simulate_annealing(init_assign, init_load)
        
        if sa_cost < best_overall_cost:
            best_overall_cost = sa_cost
            best_overall_assign = sa_assign
            
        # If we found a valid solution with cost 0 (impossible) or very low, we might stop?
        # No, just keep going until time.

    # Final Check: Ensure the best solution is valid (no capacity violations)
    # If the best solution found by SA had violations (penalty), it might be invalid for the final answer.
    # We must return a VALID assignment.
    # If best_overall_assign has violations, we need to fix it or return a different valid one.
    # The greedy init is always valid (assuming total cap >= total demand and greedy succeeds).
    # So we should keep track of the best VALID solution separately, or check at the end.
    
    # Let's re-evaluate best_overall_assign
    loads = np.bincount(best_overall_assign, minlength=n_fac)
    if np.any(loads > capacities):
        # It's invalid. We need to find a valid one.
        # The first greedy init was valid. Let's use that if it was better than this invalid one?
        # Actually, the penalty makes invalid solutions "expensive", so SA should prefer valid ones.
        # But if it ends in an invalid state, we need to repair.
        # Simple repair: Re-run greedy on the customers that are causing issues?
        # Or just return the best VALID solution we encountered during SA.
        
        # Let's modify simulate_annealing to track best VALID solution.
        # For now, if the final best is invalid, we fall back to the first greedy init.
        # This is a safety net.
        
        # Let's try to repair:
        # Identify overloaded facilities.
        # Move customers from overloaded to underloaded.
        # This is complex.
        
        # Better approach: In simulate_annealing, only update best_assign if the solution is valid.
        # Let's assume the penalty is high enough that the global best in SA is valid.
        # If not, we have a bug in the penalty logic or the search got stuck.
        
        # Let's check the cost. If penalty > 0, it's invalid.
        # If best_overall_cost > 0 and contains penalty, it's invalid.
        # We need to ensure we return a valid assignment.
        
        # Fallback: Use the result of the first greedy_init which is guaranteed valid (if greedy succeeds).
        # We stored init_assign from the first run? No, we didn't store it globally.
        # Let's re-run greedy_init to get a valid baseline.
        valid_baseline, _ = greedy_init()
        baseline_cost = calc_cost(valid_baseline)
        
        if baseline_cost < best_overall_cost:
            best_overall_assign = valid_baseline
        else:
            # Try to repair best_overall_assign
            # Simple local search to fix capacity:
            # While any capacity violation:
            #   Find a customer in an overloaded facility.
            #   Move to a facility with capacity and lowest cost increase.
            
            current = best_overall_assign.copy()
            for _ in range(1000): # Max repair steps
                loads = np.bincount(current, minlength=n_fac)
                violations = loads > capacities
                if not np.any(violations):
                    break
                
                # Find an overloaded facility
                overloaded_facs = np.where(violations)[0]
                # Pick one
                of = overloaded_facs[0]
                
                # Find a customer assigned to of
                customers_in_of = np.where(current == of)[0]
                # Pick one (e.g., the one with highest cost to move? or random?)
                # Let's pick the one with the highest assignment cost to 'of' (least "painful" to move if we find a better one? No, we just need to move it).
                # Let's pick the one with the smallest demand to make it easier to fit?
                # Or just the first one.
                ci = customers_in_of[0]
                
                # Find a facility with capacity
                # Calculate cost for moving ci to each f
                # f must have capacity: loads[f] + demands[ci] <= capacities[f]
                # And f != of (since of is overloaded, it definitely doesn't have capacity for one more)
                
                best_new_f = -1
                min_cost = np.inf
                
                for f in range(n_fac):
                    if f == of:
                        continue
                    if loads[f] + demands[ci] <= capacities[f]:
                        # Cost change:
                        # Remove ci from of: -assign_cost[ci, of]
                        # Add ci to f: +assign_cost[ci, f]
                        # Open cost change:
                        # If of becomes empty, we save open_cost[of]
                        # If f was empty, we pay open_cost[f]
                        
                        # This is getting complicated to calculate delta exactly in the repair loop.
                        # Let's just calculate the new total cost for the candidate move?
                        # That's O(N) per check. N=100. 1000 steps * 100 checks * 100 cost calc = 10M ops. Fast enough.
                        
                        trial = current.copy()
                        trial[ci] = f
                        c = calc_cost(trial)
                        if c < min_cost:
                            min_cost = c
                            best_new_f = f
                            
                if best_new_f != -1:
                    current[ci] = best_new_f
                else:
                    # No feasible move found?
                    # This implies infeasibility or we are stuck.
                    # Break and hope it's valid? No, it's not.
                    # Return baseline if this fails.
                    return valid_baseline
                    
            # After repair loop
            final_loads = np.bincount(current, minlength=n_fac)
            if not np.any(final_loads > capacities):
                best_overall_assign = current
            else:
                best_overall_assign = valid_baseline

    return best_overall_assign

# EVOLVE-BLOCK-END

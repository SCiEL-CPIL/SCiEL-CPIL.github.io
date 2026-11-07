"""Capacitated facility location: improved heuristic with local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility respecting capacity constraints.
    Uses a greedy construction followed by local search (relocate + 2-swap).
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Deadline guard: 115s safety margin out of 120s
    deadline = time.time() + 115.0
    
    # Convert to numpy arrays for speed
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)
    
    # --- Greedy construction: largest demand first, cheapest feasible facility ---
    assignment = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities, dtype=np.float64)
    
    # Sort customers by demand descending
    order = np.argsort(-demands)
    for i in order:
        # Get facilities sorted by assignment cost for this customer
        fac_order = np.argsort(assign_cost[i])
        for f in fac_order:
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                assignment[i] = f
                break
        else:
            # Fallback: assign to facility with most remaining capacity
            # This shouldn't happen if instance is feasible, but just in case
            remaining = capacities - load
            f = np.argmax(remaining)
            if load[f] + demands[i] <= capacities[f]:
                load[f] += demands[i]
                assignment[i] = f
            else:
                # Force assign (should not happen in valid instances)
                assignment[i] = f
                load[f] += demands[i]
    
    # Compute initial cost
    def compute_cost(assignment):
        """Compute total cost: open costs + assignment costs."""
        # Assignment cost
        total_assign = np.sum(assign_cost[np.arange(n_customers), assignment])
        # Open cost: facilities with at least one customer
        used_facilities = np.unique(assignment)
        total_open = np.sum(open_cost[used_facilities])
        return total_assign + total_open
    
    current_cost = compute_cost(assignment)
    best_assignment = assignment.copy()
    best_cost = current_cost
    
    # --- Local search: relocate moves ---
    # For each customer, try moving to a different facility if it reduces cost
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Process customers in random-ish order for diversity
        for i in range(n_customers):
            if time.time() > deadline:
                break
            current_f = assignment[i]
            # Try all other facilities
            for f in range(n_facilities):
                if f == current_f:
                    continue
                # Check capacity
                if load[f] + demands[i] > capacities[f]:
                    continue
                # Compute delta cost
                # Old: customer i at current_f
                # New: customer i at f
                delta_assign = assign_cost[i, f] - assign_cost[i, current_f]
                
                # Open cost change:
                # If current_f has no other customers, we save open_cost[current_f]
                # If f has no other customers, we add open_cost[f]
                # Count customers at each facility
                # For efficiency, track counts
                # Let's use a simple approach: count how many are at current_f and f
                # This is O(n) per check which is too slow. Let's maintain counts.
                
                # Actually, let's maintain a count array
                pass
            
            # This inner loop is too slow with the open cost check. Let's restructure.
            # We'll maintain facility counts for open cost tracking.
        
        # Restart with a better approach: maintain facility customer counts
        break  # Break out of the while loop, we'll redo with better structure
    
    # Better local search with maintained counts
    def local_search(assignment, load, max_time):
        """Perform local search with relocate and 2-swap moves."""
        n_customers = len(assignment)
        n_facilities = len(load)
        
        # Maintain count of customers per facility
        fac_counts = np.zeros(n_facilities, dtype=int)
        for i in range(n_customers):
            fac_counts[assignment[i]] += 1
        
        def get_cost():
            total_assign = np.sum(assign_cost[np.arange(n_customers), assignment])
            used = fac_counts > 0
            total_open = np.sum(open_cost[used])
            return total_assign + total_open
        
        current_cost = get_cost()
        best_cost = current_cost
        best_assignment = assignment.copy()
        
        deadline = time.time() + max_time
        
        # Relocate moves
        no_improve = 0
        while time.time() < deadline and no_improve < 50:
            no_improve += 1
            # Try relocating each customer
            for i in range(n_customers):
                if time.time() > deadline:
                    break
                cf = assignment[i]
                # Try all other facilities
                # Sort by assignment cost for efficiency
                for f in np.argsort(assign_cost[i]):
                    if f == cf:
                        continue
                    if load[f] + demands[i] > capacities[f]:
                        continue
                    
                    # Delta assignment cost
                    d_assign = assign_cost[i, f] - assign_cost[i, cf]
                    
                    # Delta open cost
                    d_open = 0.0
                    if fac_counts[cf] == 1:
                        d_open -= open_cost[cf]  # We're closing cf
                    if fac_counts[f] == 0:
                        d_open += open_cost[f]  # We're opening f
                    
                    if d_assign + d_open < -1e-9:
                        # Make the move
                        assignment[i] = f
                        load[cf] -= demands[i]
                        load[f] += demands[i]
                        fac_counts[cf] -= 1
                        fac_counts[f] += 1
                        current_cost += d_assign + d_open
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_assignment = assignment.copy()
                        no_improve = 0
                        break  # Move to next customer
        
        # 2-swap moves: swap two customers between facilities
        if time.time() < deadline:
            # Try swapping pairs of customers
            # For efficiency, only try swaps that might help
            swapped = True
            while swapped and time.time() < deadline:
                swapped = False
                for i in range(n_customers):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, n_customers):
                        if time.time() > deadline:
                            break
                        fi = assignment[i]
                        fj = assignment[j]
                        if fi == fj:
                            continue
                        
                        # Check capacity for swap
                        # After swap: i goes to fj, j goes to fi
                        # Load at fi after: load[fi] - demands[i] + demands[j]
                        # Load at fj after: load[fj] - demands[j] + demands[i]
                        if load[fi] - demands[i] + demands[j] > capacities[fi]:
                            continue
                        if load[fj] - demands[j] + demands[i] > capacities[fj]:
                            continue
                        
                        # Delta assignment cost
                        d_assign = (assign_cost[i, fj] + assign_cost[j, fi]) - \
                                   (assign_cost[i, fi] + assign_cost[j, fj])
                        
                        # Delta open cost
                        d_open = 0.0
                        # Facility fi: loses customer i, gains customer j
                        # If fac_counts[fi] == 1 and demands[j] > 0: still has j, so no change unless...
                        # Actually, the count at fi becomes fac_counts[fi] - 1 + 1 = fac_counts[fi]
                        # So the count doesn't change for either facility!
                        # Wait, that's only if both customers move. Let me reconsider.
                        # fi: had fac_counts[fi] customers including i. After: fac_counts[fi] - 1 (remove i) + 1 (add j) = fac_counts[fi]
                        # fj: had fac_counts[fj] customers including j. After: fac_counts[fj] - 1 (remove j) + 1 (add i) = fac_counts[fj]
                        # So open cost doesn't change for 2-swaps between two non-empty facilities!
                        # Unless one facility becomes empty, but since we're swapping, both still have customers.
                        # So d_open = 0 for 2-swaps.
                        
                        if d_assign < -1e-9:
                            # Make the swap
                            assignment[i] = fj
                            assignment[j] = fi
                            load[fi] = load[fi] - demands[i] + demands[j]
                            load[fj] = load[fj] - demands[j] + demands[i]
                            current_cost += d_assign
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_assignment = assignment.copy()
                            swapped = True
                            no_improve = 0
        
        return best_assignment, best_cost, load, fac_counts
    
    # Run local search with remaining time
    remaining_time = deadline - time.time() - 2.0  # Leave 2s margin
    if remaining_time > 1.0:
        assignment, current_cost, load, fac_counts = local_search(assignment, load, remaining_time)
    
    # Final validation
    # Check capacities
    final_load = np.zeros(n_facilities, dtype=np.float64)
    for i in range(n_customers):
        final_load[assignment[i]] += demands[i]
    
    # If any capacity violated, fall back to greedy
    if np.any(final_load > capacities + 1e-9):
        # Redo greedy
        assignment = np.zeros(n_customers, dtype=int)
        load = np.zeros(n_facilities, dtype=np.float64)
        order = np.argsort(-demands)
        for i in order:
            fac_order = np.argsort(assign_cost[i])
            for f in fac_order:
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    assignment[i] = f
                    break
            else:
                remaining = capacities - load
                f = np.argmax(remaining)
                assignment[i] = f
                load[f] += demands[i]
    
    return assignment.astype(int)
# EVOLVE-BLOCK-END

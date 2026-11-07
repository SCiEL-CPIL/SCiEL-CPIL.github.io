"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assigns customers to facilities using Mixed Integer Programming (MIP)
    with a robust, fast heuristic fallback.
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # Try to use scipy.optimize.milp for exact solution
    try:
        if not hasattr(scipy.optimize, 'milp'):
            raise AttributeError("milp not available")
            
        n_x = n_customers * n_facilities
        n_y = n_facilities
        n_vars = n_x + n_y
        
        # Objective coefficients
        c = np.zeros(n_vars)
        c[:n_x] = assign_cost.flatten()
        c[n_x:] = open_cost
        
        # Build constraint matrix efficiently
        # A_eq: n_customers x n_vars
        # A_ineq: n_facilities x n_vars
        
        # Vectorized construction of A_eq
        # Indices for x_{ij} in the flattened array are i * n_facilities + j
        # We want A_eq[i, i*n_f + j] = 1 for all j
        A_eq = np.zeros((n_customers, n_vars))
        for i in range(n_customers):
            start_idx = i * n_facilities
            A_eq[i, start_idx:start_idx + n_facilities] = 1.0
        
        # Vectorized construction of A_ineq
        # Row j corresponds to facility j
        # A_ineq[j, i*n_f + j] = demands[i]
        # A_ineq[j, n_x + j] = -capacities[j]
        A_ineq = np.zeros((n_facilities, n_vars))
        for j in range(n_facilities):
            # For each customer i, the variable is at index i * n_facilities + j
            # We can construct the column indices for this facility
            col_indices = np.arange(j, n_x, n_facilities)
            A_ineq[j, col_indices] = demands
            A_ineq[j, n_x + j] = -capacities[j]
        
        A = np.vstack([A_eq, A_ineq])
        lb = np.concatenate([np.ones(n_customers), -np.inf * np.ones(n_facilities)])
        ub = np.concatenate([np.ones(n_customers), np.zeros(n_facilities)])
        
        cons = scipy.optimize.LinearConstraint(A, lb, ub)
        integrality = np.ones(n_vars)
        bounds = scipy.optimize.Bounds(0, 1)
        
        # Solve with a reasonable time limit to ensure we return within 120s
        # 10s is usually enough for 100x20, leaving time for fallback if needed
        res = scipy.optimize.milp(c, constraints=cons, integrality=integrality, bounds=bounds, 
                                  options={"presolve": True, "time_limit": 10})
        
        if res.status in [0, 1, 2]: # Optimal, Feasible, or Locally Optimal
            x_flat = res.x[:n_x]
            x = x_flat.reshape(n_customers, n_facilities)
            result = np.argmax(x, axis=1).astype(int)
            
            # Verify validity
            load = np.zeros(n_facilities)
            for i in range(n_customers):
                load[result[i]] += demands[i]
            if np.any(load > capacities + 1e-5):
                return _fallback_assign(demands, capacities, open_cost, assign_cost)
            return result
        else:
            return _fallback_assign(demands, capacities, open_cost, assign_cost)

    except Exception:
        return _fallback_assign(demands, capacities, open_cost, assign_cost)


def _fallback_assign(demands, capacities, open_cost, assign_cost):
    """
    Fast Greedy + Local Search (Moves and Swaps).
    """
    n_customers = len(demands)
    n_facilities = len(capacities)
    
    # 1. Initial Greedy Assignment
    # Sort by demand descending to place large items first
    order = np.argsort(-demands)
    result = np.zeros(n_customers, dtype=int)
    load = np.zeros(n_facilities)
    
    for i in order:
        # Find feasible facility with min cost
        # We can vectorize the cost check
        remaining_cap = capacities - load
        # Mask infeasible facilities
        feasible_mask = remaining_cap >= demands[i]
        
        if not np.any(feasible_mask):
            # If no feasible facility, pick one with max remaining capacity
            # This should be rare if instance is feasible
            best_fac = np.argmax(remaining_cap)
            result[i] = best_fac
            load[best_fac] += demands[i]
        else:
            # Set infeasible costs to inf
            costs = assign_cost[i].copy()
            costs[~feasible_mask] = np.inf
            best_fac = np.argmin(costs)
            result[i] = best_fac
            load[best_fac] += demands[i]
            
    # 2. Local Search
    # Maintain count of customers per facility to quickly check if a facility becomes empty
    count = np.zeros(n_facilities, dtype=int)
    for fac in result:
        count[fac] += 1
        
    def get_cost():
        total_assign = np.sum(assign_cost[np.arange(n_customers), result])
        open_facs = np.where(count > 0)[0]
        total_open = np.sum(open_cost[open_facs])
        return total_assign + total_open

    current_cost = get_cost()
    improved = True
    max_iters = 50
    iter_count = 0
    customer_indices = np.arange(n_customers)
    
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        
        # Pass 1: Single Moves
        for i in customer_indices:
            current_fac = result[i]
            # Try moving i to any other facility j
            # We can vectorize the cost delta calculation for all j
            # delta_assign[j] = assign_cost[i, j] - assign_cost[i, current_fac]
            
            # Check capacity feasibility for all j
            new_loads = load.copy()
            new_loads[current_fac] -= demands[i]
            new_loads += demands[i] # This is wrong, we add to specific j
            
            # Let's do it slightly more efficiently for N=100, F=20
            # It's small enough that a loop is fine, but let's be smart.
            
            best_delta = 0
            best_j = -1
            
            for j in range(n_facilities):
                if j == current_fac:
                    continue
                
                # Capacity check
                if load[j] + demands[i] > capacities[j]:
                    continue
                
                # Cost Delta
                delta_assign = assign_cost[i, j] - assign_cost[i, current_fac]
                delta_open = 0
                
                # Open cost changes
                if count[current_fac] == 1: # current_fac becomes empty
                    delta_open -= open_cost[current_fac]
                if count[j] == 0: # j becomes occupied
                    delta_open += open_cost[j]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_j = j
            
            if best_j != -1:
                # Perform move
                load[current_fac] -= demands[i]
                load[best_j] += demands[i]
                count[current_fac] -= 1
                count[best_j] += 1
                result[i] = best_j
                current_cost += best_delta
                improved = True
        
        # Pass 2: Swaps (Optional, but helpful for capacitated problems)
        # To keep it fast, we might skip swaps if moves are still improving significantly.
        # However, swaps can escape local minima where a single move is blocked by capacity.
        # Given N=100, O(N^2) swaps per iteration is 10k checks. 50 iterations = 500k checks.
        # This is very fast in Python/NumPy if vectorized or simple loops.
        
        # Let's add a limited swap search
        swap_improved = False
        # Sample a subset of swaps to keep it fast? Or all?
        # All pairs is 100*99/2 = 4950. Very fast.
        for i in range(n_customers):
            fac_i = result[i]
            for k in range(i + 1, n_customers):
                fac_k = result[k]
                if fac_i == fac_k:
                    continue
                
                # Check if swap is feasible
                # Load at fac_i: -d_i + d_k
                # Load at fac_k: -d_k + d_i
                new_load_i = load[fac_i] - demands[i] + demands[k]
                new_load_k = load[fac_k] - demands[k] + demands[i]
                
                if new_load_i > capacities[fac_i] or new_load_k > capacities[fac_k]:
                    continue
                
                # Cost Delta
                delta_assign = (assign_cost[i, fac_k] + assign_cost[k, fac_i]) - \
                               (assign_cost[i, fac_i] + assign_cost[k, fac_k])
                
                # Open cost delta
                delta_open = 0
                # If fac_i becomes empty (count was 1 and we moved i out, but k moves in? No, k moves to fac_i)
                # Count at fac_i: was c_i. i leaves, k enters. Count remains c_i.
                # Unless c_i == 1? If c_i == 1, i was the only one. i leaves (count 0), k enters (count 1).
                # So count stays same?
                # Wait.
                # Initial: fac_i has i. fac_k has k.
                # Final: fac_i has k. fac_k has i.
                # The set of occupied facilities does NOT change unless one of them was the ONLY customer at that facility.
                # If count[fac_i] == 1: Before: fac_i occupied (by i). After: fac_i occupied (by k). No change in open cost?
                # If count[fac_k] == 1: Before: fac_k occupied (by k). After: fac_k occupied (by i). No change?
                # So swaps NEVER change the open cost?
                # Yes, because the facility remains open in both cases.
                # So delta_open = 0 always for swaps between two different facilities.
                
                total_delta = delta_assign
                
                if total_delta < -1e-9:
                    # Perform swap
                    # Update loads
                    load[fac_i] = new_load_i
                    load[fac_k] = new_load_k
                    # Counts don't change
                    result[i] = fac_k
                    result[k] = fac_i
                    current_cost += total_delta
                    improved = True
                    swap_improved = True
                    # Break inner loop? Or continue?
                    # Continue to find more swaps in this pass?
                    # Breaking is safer to avoid interference, but continuing is faster convergence.
                    # Let's break to re-evaluate state.
                    break
            if swap_improved:
                break # Break outer loop for this pass?
                # Actually, if we swap, the state changed. We should probably restart the swap pass or just continue?
                # Let's just mark improved and continue the outer while loop.
                # But we are inside the for i loop.
                # Let's just set a flag and break out of the swap loops.
                break

    return result

# EVOLVE-BLOCK-END

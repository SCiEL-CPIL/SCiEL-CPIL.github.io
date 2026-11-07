"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost for a given assignment."""
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        load[f] += demands[i]
    
    # Check validity
    if np.any(load > capacities + 1e-6):
        return np.inf
    
    # Opening costs
    open_facs = np.unique(result)
    open_cost_sum = np.sum(open_cost[open_facs])
    
    # Assignment costs
    assign_cost_sum = np.sum([assign_cost[i, result[i]] for i in range(len(result))])
    
    return open_cost_sum + assign_cost_sum

def _is_valid(result, demands, capacities):
    """Check if assignment is valid (capacity constraints satisfied)."""
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        if f < 0 or f >= n_fac:
            return False
        load[f] += demands[i]
    return np.all(load <= capacities + 1e-6)

def _greedy_init(demands, capacities, open_cost, assign_cost):
    """
    Improved greedy initialization: consider both assignment cost and potential opening cost savings.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac)
    
    # Sort customers by demand descending (larger demands first for better packing)
    order = np.argsort(-demands)
    
    for i in order:
        best_f = -1
        best_cost = np.inf
        
        for f in range(n_fac):
            if load[f] + demands[i] <= capacities[f]:
                # Cost: assignment cost + amortized opening cost if this is the first customer
                if load[f] == 0:
                    cost = assign_cost[i, f] + open_cost[f] / max(1, int(capacities[f] / demands[i]))
                else:
                    cost = assign_cost[i, f]
                
                if cost < best_cost:
                    best_cost = cost
                    best_f = f
        
        if best_f >= 0:
            result[i] = best_f
            load[best_f] += demands[i]
        else:
            # Fallback: assign to facility with most remaining capacity
            rem = capacities - load
            best_f = np.argmax(rem)
            result[i] = best_f
            load[best_f] += demands[i]
    
    return result

def _local_search_move(result, demands, capacities, open_cost, assign_cost, max_iters=10000):
    """
    Local search: try moving individual customers to better facilities.
    """
    n_cust = len(result)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        load[f] += demands[i]
    
    current_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(current_cost):
        return result
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        for i in range(n_cust):
            old_f = result[i]
            old_load = load[old_f]
            
            # Try moving to each other facility
            for new_f in range(n_fac):
                if new_f == old_f:
                    continue
                if load[new_f] + demands[i] > capacities[new_f] + 1e-6:
                    continue
                
                # Compute cost delta
                delta_assign = assign_cost[i, new_f] - assign_cost[i, old_f]
                
                # Opening cost change
                delta_open = 0.0
                # If old facility becomes empty
                if old_load == demands[i] and np.all(result != old_f):
                    # Actually need to check if any other customer is at old_f
                    pass
                # Simpler: recompute opening costs
                new_result = result.copy()
                new_result[i] = new_f
                
                # Check if this move is valid and improves cost
                new_load = load.copy()
                new_load[old_f] -= demands[i]
                new_load[new_f] += demands[i]
                
                if np.any(new_load > capacities + 1e-6):
                    continue
                
                # Compute new cost
                new_open_facs = set(new_result.tolist())
                old_open_facs = set(result.tolist())
                
                # Facilities that were opened and are now closed
                closed_facs = old_open_facs - new_open_facs
                # Facilities that are now opened but weren't before
                opened_facs = new_open_facs - old_open_facs
                
                delta_open = -np.sum(open_cost[list(closed_facs)]) + np.sum(open_cost[list(opened_facs)])
                
                total_delta = delta_assign + delta_open
                
                if total_delta < -1e-9:
                    # Accept the move
                    result[i] = new_f
                    load[old_f] -= demands[i]
                    load[new_f] += demands[i]
                    current_cost += total_delta
                    improved = True
                    break  # Restart scan from beginning after a move
    
    return result

def _local_search_swap(result, demands, capacities, open_cost, assign_cost, max_iters=5000):
    """
    Local search: try swapping two customers between facilities.
    """
    n_cust = len(result)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    for i, f in enumerate(result):
        load[f] += demands[i]
    
    current_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    if np.isinf(current_cost):
        return result
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # Group customers by facility
        fac_customers = {f: [] for f in range(n_fac)}
        for i in range(n_cust):
            fac_customers[result[i]].append(i)
        
        # Try swapping customers from different facilities
        for f1 in range(n_fac):
            if len(fac_customers[f1]) < 1:
                continue
            for f2 in range(f1 + 1, n_fac):
                if len(fac_customers[f2]) < 1:
                    continue
                
                for i in fac_customers[f1]:
                    for j in fac_customers[f2]:
                        if i == j:
                            continue
                        
                        # Check capacity after swap
                        # i moves from f1 to f2, j moves from f2 to f1
                        if load[f1] - demands[i] + demands[j] > capacities[f1] + 1e-6:
                            continue
                        if load[f2] - demands[j] + demands[i] > capacities[f2] + 1e-6:
                            continue
                        
                        # Compute cost delta
                        delta_assign = (assign_cost[i, f2] + assign_cost[j, f1]) - (assign_cost[i, f1] + assign_cost[j, f2])
                        
                        # Opening cost: usually no change unless a facility becomes empty
                        delta_open = 0.0
                        new_load_f1 = load[f1] - demands[i] + demands[j]
                        new_load_f2 = load[f2] - demands[j] + demands[i]
                        
                        # Check if f1 becomes empty
                        if new_load_f1 == 0:
                            # Only if no other customers
                            if len(fac_customers[f1]) == 1:
                                delta_open += -open_cost[f1]
                        # Check if f2 becomes empty
                        if new_load_f2 == 0:
                            if len(fac_customers[f2]) == 1:
                                delta_open += -open_cost[f2]
                        
                        total_delta = delta_assign + delta_open
                        
                        if total_delta < -1e-9:
                            # Accept the swap
                            result[i] = f2
                            result[j] = f1
                            load[f1] -= demands[i]
                            load[f1] += demands[j]
                            load[f2] -= demands[j]
                            load[f2] += demands[i]
                            current_cost += total_delta
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    return result

def _refine_with_local_search(result, demands, capacities, open_cost, assign_cost, time_limit=60.0):
    """
    Apply local search moves and swaps to refine the solution.
    """
    start_time = time.time()
    
    # First pass: individual moves
    result = _local_search_move(result, demands, capacities, open_cost, assign_cost, max_iters=20000)
    
    elapsed = time.time() - start_time
    if elapsed > time_limit * 0.4:
        return result
    
    # Second pass: swaps
    result = _local_search_swap(result, demands, capacities, open_cost, assign_cost, max_iters=10000)
    
    elapsed = time.time() - start_time
    if elapsed > time_limit * 0.7:
        return result
    
    # Third pass: more moves
    result = _local_search_move(result, demands, capacities, open_cost, assign_cost, max_iters=10000)
    
    return result

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # 1. Start with greedy initialization
    result = _greedy_init(demands, capacities, open_cost, assign_cost)
    best_result = result.copy()
    best_cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
    
    # 2. Try MIP with time limit
    try:
        elapsed = time.time() - start_time
        mip_time_limit = min(60.0, deadline - start_time - 30.0)  # Leave 30s for local search
        
        if mip_time_limit > 5.0:  # Only try MIP if we have enough time
            # Variables: x_{ij} for i in n_cust, j in n_fac (n_cust * n_fac vars)
            #             y_j for j in n_fac (n_fac vars)
            # Total vars: n_cust * n_fac + n_fac
            
            # Objective coefficients
            c = np.zeros(n_cust * n_fac + n_fac)
            
            # x_{ij} coefficients are assign_cost[i, j]
            # y_j coefficients are open_cost[j]
            
            # Flatten assign_cost: row-major order (i, j)
            c[:n_cust * n_fac] = assign_cost.flatten()
            c[n_cust * n_fac:] = open_cost
            
            # Constraints
            # 1. Assignment: sum_j x_{ij} = 1 for all i
            #    Matrix A_eq shape: (n_cust, n_cust*n_fac + n_fac)
            A_eq = np.zeros((n_cust, n_cust * n_fac + n_fac))
            for i in range(n_cust):
                # x_{ij} are at indices i*n_fac + j
                for j in range(n_fac):
                    A_eq[i, i * n_fac + j] = 1.0
            
            b_eq = np.ones(n_cust)
            
            # 2. Capacity: sum_i demands[i] * x_{ij} <= capacities[j] * y_j for all j
            #    sum_i demands[i] * x_{ij} - capacities[j] * y_j <= 0
            #    Matrix A_ub shape: (n_fac, n_cust*n_fac + n_fac)
            A_ub = np.zeros((n_fac, n_cust * n_fac + n_fac))
            for j in range(n_fac):
                # Coefficients for x_{ij}
                for i in range(n_cust):
                    A_ub[j, i * n_fac + j] = demands[i]
                # Coefficient for y_j
                # y_j is at index n_cust*n_fac + j
                A_ub[j, n_cust * n_fac + j] = -capacities[j]
                
            b_ub = np.zeros(n_fac)
            
            # Bounds
            # x_{ij} in [0, 1]
            # y_j in [0, 1]
            integrality = np.zeros(n_cust * n_fac + n_fac)
            integrality[:] = 1  # All binary
            
            lb = np.zeros(n_cust * n_fac + n_fac)
            ub = np.ones(n_cust * n_fac + n_fac)
            
            # Define constraints
            constraint_eq = LinearConstraint(A_eq, b_eq, b_eq)
            constraint_ub = LinearConstraint(A_ub, -np.inf, b_ub)
            
            bounds = Bounds(lb, ub)
            
            # Solve with time limit
            res = milp(
                c=c,
                constraints=[constraint_eq, constraint_ub],
                integrality=integrality,
                bounds=bounds,
                options={'time_limit': mip_time_limit}
            )
            
            if res.status == 0 or res.status == 8:  # Optimal or time limit reached
                x = res.x
                # Extract x_{ij}
                x_mat = x[:n_cust * n_fac].reshape(n_cust, n_fac)
                # Find assigned facility for each customer
                result_mip = np.argmax(x_mat, axis=1).astype(int)
                
                # Verify validity
                if _is_valid(result_mip, demands, capacities):
                    cost_mip = _compute_cost(result_mip, demands, capacities, open_cost, assign_cost)
                    if cost_mip < best_cost:
                        best_result = result_mip
                        best_cost = cost_mip
                        result = result_mip
                else:
                    # If MIP result is invalid, keep greedy result
                    pass
            elif res.status == 2:  # Infeasible (shouldn't happen)
                pass
            # For other statuses, keep current best
            
    except Exception:
        # If MIP throws an error, keep current best
        pass
    
    # 3. Refine with local search if we have time
    elapsed = time.time() - start_time
    local_search_time = deadline - start_time - 5.0  # Leave 5s buffer
    
    if local_search_time > 10.0:  # Only do local search if we have enough time
        try:
            refined = _refine_with_local_search(result, demands, capacities, open_cost, assign_cost, 
                                                 time_limit=local_search_time)
            if _is_valid(refined, demands, capacities):
                cost_refined = _compute_cost(refined, demands, capacities, open_cost, assign_cost)
                if cost_refined < best_cost:
                    best_result = refined
                    best_cost = cost_refined
        except Exception:
            # If local search fails, keep current best
            pass
    
    # 4. Final validation
    if not _is_valid(best_result, demands, capacities):
        # Emergency fallback
        best_result = _greedy_init(demands, capacities, open_cost, assign_cost)
    
    return best_result

# EVOLVE-BLOCK-END

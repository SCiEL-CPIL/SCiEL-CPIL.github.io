"""Capacitated facility location using min-cost flow + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost of an assignment."""
    if result is None:
        return float('inf')
    n = len(result)
    total = 0.0
    open_fac = set()
    for i in range(n):
        f = result[i]
        if f < 0 or f >= len(capacities):
            return float('inf')
        total += assign_cost[i, f]
        open_fac.add(f)
    for f in open_fac:
        total += open_cost[f]
    # Verify capacity
    load = np.zeros(len(capacities))
    for i in range(n):
        load[result[i]] += demands[i]
    if np.any(load > capacities + 1e-6):
        return float('inf')
    return total

def _solve_mcf(demands, capacities, assign_cost, n_cust, n_fac):
    """
    Solve capacitated assignment via min-cost flow.
    Creates a bipartite graph: source -> customers -> facilities -> sink
    with capacities on facility arcs.
    """
    # Build cost matrix for Hungarian: each customer must be assigned to one facility
    # But we need capacity constraints, so we use a different approach.
    # We'll use a successive shortest path or just do a relaxed assignment + repair.
    
    # Approach: Use linear_sum_assignment on a modified cost matrix
    # where we duplicate facilities based on capacity, but that's complex.
    
    # Simpler: iterative improvement with capacity-aware moves.
    # Start with a feasible greedy solution, then improve.
    pass

def _greedy_feasible(demands, capacities, assign_cost, order):
    """Greedy assignment in given order, respecting capacities."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    
    for i in order:
        # Find feasible facility with minimum cost
        feasible = []
        for f in range(n_fac):
            if load[f] + demands[i] <= capacities[f]:
                feasible.append((assign_cost[i, f], f))
        if not feasible:
            return None
        feasible.sort()
        best_f = feasible[0][1]
        result[i] = best_f
        load[best_f] += demands[i]
    
    return result

def _local_search_improve(result, demands, capacities, open_cost, assign_cost, deadline):
    """
    Improve assignment via local search with move and swap operations.
    Handles facility opening costs by tracking which facilities are open.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    best_result = result.copy()
    load = np.zeros(n_fac)
    for i in range(n_cust):
        load[best_result[i]] += demands[i]
    
    def get_cost():
        total = 0.0
        open_set = set()
        for i in range(n_cust):
            f = best_result[i]
            total += assign_cost[i, f]
            open_set.add(f)
        for f in open_set:
            total += open_cost[f]
        return total
    
    best_cost = get_cost()
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        no_improve_count += 1
        
        # Try moving each customer to a different facility
        for i in range(n_cust):
            if time.time() > deadline:
                break
            current_f = best_result[i]
            
            # Get all feasible alternative facilities sorted by cost
            alternatives = []
            for f in range(n_fac):
                if f == current_f:
                    continue
                if load[f] + demands[i] <= capacities[f]:
                    alternatives.append((assign_cost[i, f], f))
            
            if not alternatives:
                continue
            
            alternatives.sort()
            
            for new_cost, new_f in alternatives:
                # Try the move
                old_cost = best_cost
                
                load[current_f] -= demands[i]
                load[new_f] += demands[i]
                best_result[i] = new_f
                
                new_total = get_cost()
                
                if new_total < old_cost - 1e-9:
                    best_cost = new_total
                    improved = True
                    no_improve_count = 0
                    break  # Move accepted, try next customer
                else:
                    # Revert
                    best_result[i] = current_f
                    load[current_f] += demands[i]
                    load[new_f] -= demands[i]
                    break  # Cheapest move didn't help, stop for this customer
        
        if not improved:
            # Try swap operations: exchange two customers
            if time.time() > deadline:
                break
            swap_improved = False
            for i in range(n_cust):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n_cust):
                    if time.time() > deadline:
                        break
                    fi = best_result[i]
                    fj = best_result[j]
                    if fi == fj:
                        continue
                    
                    # Check if swap is feasible
                    if (load[fi] - demands[i] + demands[j] <= capacities[fi] and
                        load[fj] - demands[j] + demands[i] <= capacities[fj]):
                        # Try swap
                        old_cost = best_cost
                        load[fi] -= demands[i]
                        load[fj] -= demands[j]
                        load[fi] += demands[j]
                        load[fj] += demands[i]
                        best_result[i] = fj
                        best_result[j] = fi
                        
                        new_total = get_cost()
                        
                        if new_total < old_cost - 1e-9:
                            best_cost = new_total
                            swap_improved = True
                            improved = True
                            no_improve_count = 0
                            break
                        else:
                            # Revert
                            load[fi] -= demands[j]
                            load[fj] -= demands[i]
                            load[fi] += demands[i]
                            load[fj] += demands[j]
                            best_result[i] = fi
                            best_result[j] = fj
                if swap_improved:
                    break
    
    return best_result, best_cost

def _multistart(demands, capacities, open_cost, assign_cost, deadline):
    """
    Run multiple greedy starts with different orders, then local search on best.
    """
    n_cust = len(demands)
    n_fac = len(capacities)
    
    best_result = None
    best_cost = float('inf')
    
    # Generate diverse initial orders
    orders = []
    
    # 1. Largest demand first
    orders.append(np.argsort(-demands))
    
    # 2. Smallest demand first
    orders.append(np.argsort(demands))
    
    # 3. By minimum assignment cost
    min_costs = np.min(assign_cost, axis=1)
    orders.append(np.argsort(min_costs))
    
    # 4. By maximum assignment cost (reverse)
    orders.append(np.argsort(-min_costs))
    
    # 5. Random orders
    rng = np.random.RandomState(42)
    for _ in range(20):
        orders.append(rng.permutation(n_cust))
    
    # Time budget per start
    time_per_start = max(0.5, (deadline - time.time()) / (len(orders) + 1))
    
    for order in orders:
        if time.time() > deadline - 5:  # Leave time for final local search
            break
        
        result = _greedy_feasible(demands, capacities, assign_cost, order)
        if result is None:
            continue
        
        cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
    
    if best_result is None:
        # Fallback: any feasible solution
        result = _greedy_feasible(demands, capacities, assign_cost, np.argsort(-demands))
        if result is None:
            return np.zeros(n_cust, dtype=int)
        best_result = result
    
    # Final local search with remaining time
    ls_deadline = min(deadline, time.time() + (deadline - time.time()) * 0.7)
    best_result, best_cost = _local_search_improve(
        best_result, demands, capacities, open_cost, assign_cost, ls_deadline
    )
    
    return best_result

def assign(demands, capacities, open_cost, assign_cost):
    """
    Assign each customer to a facility minimizing total cost.
    
    Args:
        demands: array of customer demands (integers 5-35)
        capacities: array of facility capacities (integers)
        open_cost: array of facility opening costs (floats 50-150)
        assign_cost: (n_customers, n_facilities) cost matrix
    
    Returns:
        Integer array with facility index per customer.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    demands = np.asarray(demands, dtype=int)
    capacities = np.asarray(capacities, dtype=int)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Main solver: multistart greedy + local search
    result = _multistart(demands, capacities, open_cost, assign_cost, deadline)
    
    # Final validation
    if result is None or len(result) != n_cust:
        return np.zeros(n_cust, dtype=int)
    
    result = np.asarray(result, dtype=int)
    
    # Verify validity
    load = np.zeros(n_fac)
    for i in range(n_cust):
        f = result[i]
        if f < 0 or f >= n_fac:
            return np.zeros(n_cust, dtype=int)
        load[f] += demands[i]
    
    if np.any(load > capacities + 1e-6):
        # Try to repair by reassigning overloaded customers
        for i in range(n_cust):
            f = result[i]
            if load[f] > capacities[f]:
                # Find alternative facility
                for alt_f in range(n_fac):
                    if alt_f != f and load[alt_f] + demands[i] <= capacities[alt_f]:
                        load[f] -= demands[i]
                        load[alt_f] += demands[i]
                        result[i] = alt_f
                        break
    
    return result

# EVOLVE-BLOCK-END

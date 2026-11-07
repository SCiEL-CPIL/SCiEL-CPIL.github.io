"""Implemented with Multi-Start Greedy + Swap-Based Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment


def _construct_greedy(demands, capacities, open_cost, assign_cost, order):
    n = len(demands)
    m = len(capacities)
    assignment = np.full(n, -1, dtype=int)
    load = np.zeros(m, dtype=float)
    
    for i in order:
        d = demands[i]
        # Feasible: capacity not exceeded
        feasible = (load + d) <= (capacities + 1e-9)
        if not np.any(feasible):
            # Should not happen for valid instances; assign to least loaded feasible by capacity
            # As a fallback, pick facility with most remaining capacity
            remaining = capacities - load
            f = int(np.argmax(remaining))
            assignment[i] = f
            load[f] += d
            continue
        
        costs = assign_cost[i].copy()
        costs[~feasible] = np.inf
        f = int(np.argmin(costs))
        assignment[i] = f
        load[f] += d
    
    return assignment, load


def _compute_cost(assignment, open_cost, assign_cost):
    used = np.unique(assignment[assignment >= 0])
    open_sum = open_cost[used].sum()
    assign_sum = assign_cost[np.arange(len(assignment)), assignment].sum()
    return open_sum + assign_sum


def _swap_local_search(assignment, load, demands, capacities, open_cost, assign_cost, max_iters=200):
    n = len(demands)
    m = len(capacities)
    
    def calc_cost(assignment, load):
        used = np.unique(assignment)
        return open_cost[used].sum() + assign_cost[np.arange(n), assignment].sum()
    
    current_cost = calc_cost(assignment, load)
    improved = True
    it = 0
    
    while improved and it < max_iters:
        improved = False
        it += 1
        for i in range(n):
            for j in range(i + 1, n):
                if assignment[i] == assignment[j]:
                    continue
                f_i = assignment[i]
                f_j = assignment[j]
                
                # Check if swapping is feasible
                # After swap: i goes to f_j, j goes to f_i
                # f_i loses demands[i], gains demands[j]
                # f_j loses demands[j], gains demands[i]
                new_load_fi = load[f_i] - demands[i] + demands[j]
                new_load_fj = load[f_j] - demands[j] + demands[i]
                
                if new_load_fi > capacities[f_i] + 1e-9:
                    continue
                if new_load_fj > capacities[f_j] + 1e-9:
                    continue
                
                # Compute delta
                d_assign = (assign_cost[i, f_j] + assign_cost[j, f_i]) - (assign_cost[i, f_i] + assign_cost[j, f_j])
                
                # Open cost changes only if a facility becomes empty or was empty
                d_open = 0.0
                # f_i: was load[f_i], now new_load_fi
                if load[f_i] == demands[i] and new_load_fi > 0:
                    # f_i was occupied by i only? No, load[f_i] was total. 
                    # Actually, if load[f_i] == demands[i], then only i was at f_i.
                    # After swap, new_load_fi = demands[j] > 0, so f_i stays open.
                    # Wait, if only i was at f_i, then after swap, j is at f_i, so it's still open.
                    # So no change in open cost for f_i unless it becomes empty.
                    # It becomes empty only if new_load_fi == 0.
                    pass
                if new_load_fi == 0:
                    # f_i becomes empty
                    if load[f_i] > 0:
                        d_open -= open_cost[f_i]
                # f_j: similar
                if new_load_fj == 0:
                    if load[f_j] > 0:
                        d_open -= open_cost[f_j]
                
                # Also check if f_i or f_j were empty before? No, they had at least i and j respectively.
                # So they were open. If they become empty, we save open cost.
                
                total_delta = d_assign + d_open
                
                if total_delta < -1e-9:
                    # Perform swap
                    assignment[i], assignment[j] = f_j, f_i
                    load[f_i] = new_load_fi
                    load[f_j] = new_load_fj
                    current_cost += total_delta
                    improved = True
    
    return assignment, current_cost


def _move_local_search(assignment, load, demands, capacities, open_cost, assign_cost, max_iters=200):
    n = len(demands)
    m = len(capacities)
    
    def calc_cost(assignment, load):
        used = np.unique(assignment)
        return open_cost[used].sum() + assign_cost[np.arange(n), assignment].sum()
    
    current_cost = calc_cost(assignment, load)
    improved = True
    it = 0
    
    while improved and it < max_iters:
        improved = False
        it += 1
        for i in range(n):
            cur_f = assignment[i]
            best_f = -1
            best_delta = 0.0
            
            cur_assign_cost = assign_cost[i, cur_f]
            
            for new_f in range(m):
                if new_f == cur_f:
                    continue
                if load[new_f] + demands[i] > capacities[new_f] + 1e-9:
                    continue
                
                d_assign = assign_cost[i, new_f] - cur_assign_cost
                d_open = 0.0
                
                # If current facility becomes empty
                if load[cur_f] == demands[i]:
                    d_open -= open_cost[cur_f]
                # If new facility was empty
                if load[new_f] == 0.0:
                    d_open += open_cost[new_f]
                
                total_delta = d_assign + d_open
                
                if total_delta < best_delta - 1e-9:
                    best_delta = total_delta
                    best_f = new_f
            
            if best_f != -1:
                load[cur_f] -= demands[i]
                load[best_f] += demands[i]
                assignment[i] = best_f
                current_cost += best_delta
                improved = True
    
    return assignment, current_cost


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n = len(demands)
    m = len(capacities)
    
    best_assignment = None
    best_cost = np.inf
    
    rng = np.random.default_rng(42)
    
    # Generate multiple random orderings
    num_starts = 10
    orders = []
    # Deterministic strategies
    orders.append(np.argsort(-demands))  # Largest first
    orders.append(np.argsort(demands))   # Smallest first
    orders.append(np.argsort(assign_cost.sum(axis=1)))  # Cheapest row first
    orders.append(np.argsort(assign_cost.sum(axis=1)))[::-1]  # Most expensive row first
    
    # Random strategies
    for _ in range(num_starts - len(orders)):
        orders.append(rng.permutation(n))
    
    for order in orders:
        assignment, load = _construct_greedy(demands, capacities, open_cost, assign_cost, order)
        
        # Check feasibility
        if not np.all((load <= capacities + 1e-9)):
            continue
        
        # Local search: swap first, then move
        assignment, cost1 = _swap_local_search(assignment.copy(), load.copy(), demands, capacities, open_cost, assign_cost)
        assignment, cost2 = _move_local_search(assignment.copy(), load.copy(), demands, capacities, open_cost, assign_cost)
        
        # Also try move first, then swap
        assignment2, load2 = _construct_greedy(demands, capacities, open_cost, assign_cost, order)
        if not np.all((load2 <= capacities + 1e-9)):
            continue
        assignment2, cost3 = _move_local_search(assignment2.copy(), load2.copy(), demands, capacities, open_cost, assign_cost)
        assignment2, cost4 = _swap_local_search(assignment2.copy(), load2.copy(), demands, capacities, open_cost, assign_cost)
        
        if cost1 < best_cost:
            best_cost = cost1
            best_assignment = assignment.copy()
        if cost2 < best_cost:
            best_cost = cost2
            best_assignment = assignment.copy()
        if cost3 < best_cost:
            best_cost = cost3
            best_assignment = assignment2.copy()
        if cost4 < best_cost:
            best_cost = cost4
            best_assignment = assignment2.copy()
    
    if best_assignment is None:
        # Fallback: simple greedy
        assignment, load = _construct_greedy(demands, capacities, open_cost, assign_cost, np.arange(n))
        best_assignment = assignment
    
    return best_assignment.astype(int)
# EVOLVE-BLOCK-END

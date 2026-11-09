"""Capacitated facility location: Robust local search with time bounds."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def assign(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    
    # Strict deadline guard: 110s to ensure we return well before 120s kill
    deadline = time.time() + 110.0

    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    def calc_cost(res):
        used = np.unique(res)
        return float(np.sum(open_cost[used])) + float(np.sum(assign_cost[np.arange(n_cust), res]))

    def loads_of(res):
        load = np.zeros(n_fac, dtype=np.int64)
        np.add.at(load, res, demands)
        return load

    # 1. Greedy Initialization
    def greedy_init():
        res = np.zeros(n_cust, dtype=int)
        load = np.zeros(n_fac, dtype=np.int64)
        order = np.argsort(-demands)
        for i in order:
            feasible = np.where(load + demands[i] <= capacities)[0]
            if len(feasible) == 0:
                # Fallback: pick facility with most remaining capacity
                f = int(np.argmax(capacities - load))
                res[i] = f
                load[f] += demands[i]
            else:
                costs = assign_cost[i, feasible]
                f_idx = int(feasible[np.argmin(costs)])
                res[i] = f_idx
                load[f_idx] += demands[i]
        return res, load

    best_result, best_load = greedy_init()
    best_score = calc_cost(best_result)

    # 2. Local Search: Single Moves and Swaps
    def local_search(res, load, max_time=10.0):
        t_start = time.time()
        res = res.copy()
        load = load.copy()
        
        while time.time() - t_start < max_time:
            improved = False
            
            # Pass 1: Single moves (Best Improvement)
            for i in range(n_cust):
                if time.time() - t_start > max_time:
                    break
                f_old = res[i]
                d_i = demands[i]
                
                # Check if moving i to any other facility improves cost
                # We need to check capacity and open cost changes
                feasible_mask = (load + d_i <= capacities)
                feasible_mask[f_old] = False # Cannot move to same facility
                
                if not np.any(feasible_mask):
                    continue
                
                # Calculate delta cost for each feasible facility
                # Delta Assign: assign_cost[i, f_new] - assign_cost[i, f_old]
                delta_assign = assign_cost[i, :] - assign_cost[i, f_old]
                
                # Delta Open:
                # If f_old becomes empty, we save open_cost[f_old]
                # If f_new was empty, we pay open_cost[f_new]
                delta_open = np.zeros(n_fac)
                if load[f_old] == d_i:
                    delta_open[f_old] = -open_cost[f_old]
                
                # Facilities that are currently empty (load 0) and not f_old
                # If we move to one of these, we open it.
                empty_mask = (load == 0) & (np.arange(n_fac) != f_old)
                delta_open[empty_mask] = open_cost[empty_mask]
                
                total_delta = delta_assign + delta_open
                total_delta[~feasible_mask] = np.inf
                total_delta[f_old] = np.inf
                
                best_f_new = int(np.argmin(total_delta))
                best_delta = total_delta[best_f_new]
                
                if best_delta < -1e-9:
                    # Apply move
                    res[i] = best_f_new
                    load[f_old] -= d_i
                    load[best_f_new] += d_i
                    improved = True
            
            # Pass 2: Swaps (Random sampling to avoid O(N^2) explosion)
            # Sample a limited number of pairs
            n_samples = 2000
            for _ in range(n_samples):
                if time.time() - t_start > max_time:
                    break
                i, j = np.random.choice(n_cust, 2, replace=False)
                f_i = res[i]
                f_j = res[j]
                if f_i == f_j:
                    continue
                
                d_i = demands[i]
                d_j = demands[j]
                
                # Check capacity for swap
                # f_i loses i, gains j. f_j loses j, gains i.
                if load[f_i] - d_i + d_j > capacities[f_i]:
                    continue
                if load[f_j] - d_j + d_i > capacities[f_j]:
                    continue
                
                # Calculate delta assign
                delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                               (assign_cost[j, f_i] - assign_cost[j, f_j])
                
                # Calculate delta open
                # This is complex. Only facilities that might change status are f_i and f_j.
                # If load[f_i] == d_i, f_i becomes empty (if j doesn't go there, but j goes to f_i? No, swap means i->f_j, j->f_i).
                # Wait, swap: i moves to f_j, j moves to f_i.
                # New load f_i = load[f_i] - d_i + d_j
                # New load f_j = load[f_j] - d_j + d_i
                
                # Old open status:
                old_open_i = 1 if load[f_i] > 0 else 0
                old_open_j = 1 if load[f_j] > 0 else 0
                
                new_load_fi = load[f_i] - d_i + d_j
                new_load_fj = load[f_j] - d_j + d_i
                
                new_open_i = 1 if new_load_fi > 0 else 0
                new_open_j = 1 if new_load_fj > 0 else 0
                
                delta_open = (new_open_i - old_open_i) * open_cost[f_i] + \
                             (new_open_j - old_open_j) * open_cost[f_j]
                
                total_delta = delta_assign + delta_open
                
                if total_delta < -1e-9:
                    # Apply swap
                    res[i] = f_j
                    res[j] = f_i
                    load[f_i] = new_load_fi
                    load[f_j] = new_load_fj
                    improved = True
            
            if not improved:
                break
        
        return res, load

    # 3. Multi-start / Iterative Improvement
    # We run local search on the greedy result.
    # We can also try perturbing the solution if we have time, but single run is often sufficient for 100x20.
    
    current_res, current_load = local_search(best_result, best_load, max_time=30.0)
    current_score = calc_cost(current_res)
    if current_score < best_score:
        best_score = current_score
        best_result = current_res.copy()
        best_load = current_load.copy()

    # If we still have time, try a few more local search passes or restarts
    # For safety, we'll just do one more polish on the best result
    if time.time() < deadline - 5.0:
        current_res, current_load = local_search(best_result, best_load, max_time=20.0)
        current_score = calc_cost(current_res)
        if current_score < best_score:
            best_score = current_score
            best_result = current_res.copy()
            best_load = current_load.copy()

    # Final Validation
    final_load = loads_of(best_result)
    if np.any(final_load > capacities):
        # Fallback to greedy if something went wrong
        greedy_res, _ = greedy_init()
        return greedy_res.astype(int)
    
    return best_result.astype(int)
# EVOLVE-BLOCK-END


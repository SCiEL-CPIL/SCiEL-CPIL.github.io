# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def _obj(demands, capacities, open_cost, assign_cost, res):
    n_fac = len(capacities)
    counts = np.bincount(res, minlength=n_fac)
    total = np.sum(assign_cost[np.arange(len(demands)), res])
    total += np.sum(open_cost[counts > 0])
    return total

def _feasible(demands, capacities, res):
    n_fac = len(capacities)
    if res.shape[0] != len(demands):
        return False
    if np.any(res < 0) or np.any(res >= n_fac):
        return False
    load = np.zeros(n_fac)
    np.add.at(load, res, demands)
    return bool(np.all(load <= capacities + 1e-9))

def _initial_solution(demands, capacities, open_cost, assign_cost, seed=0):
    rng = np.random.RandomState(seed)
    n_cust = len(demands)
    n_fac = len(capacities)
    res = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    order = np.arange(n_cust)
    rng.shuffle(order)
    for idx in order:
        d_i = demands[idx]
        feas = capacities - load
        feas[feas < d_i] = -1
        if np.any(feas >= 0):
            cands = np.where(feas >= 0)[0]
            costs = assign_cost[idx, cands].copy()
            new_open_mask = (load[cands] == 0)
            costs[new_open_mask] += open_cost[cands][new_open_mask]
            f = cands[np.argmin(costs)]
        else:
            f = int(np.argmin(load))
        res[idx] = f
        load[f] += d_i
    return res

def _local_search(demands, capacities, open_cost, assign_cost, res, time_limit=30.0):
    start = time.time()
    res = res.copy()
    n_cust = len(demands)
    n_fac = len(capacities)
    load = np.zeros(n_fac)
    counts = np.zeros(n_fac)
    for i in range(n_cust):
        load[res[i]] += demands[i]
        counts[res[i]] += 1
    
    cur_cost = _obj(demands, capacities, open_cost, assign_cost, res)
    best_cost = cur_cost
    best_res = res.copy()
    no_improve = 0
    
    while time.time() - start < time_limit:
        if no_improve > 10:
            break
        improved = False
        order = np.random.permutation(n_cust)
        for i in order:
            if time.time() - start > time_limit:
                break
            old_f = res[i]
            d_i = demands[i]
            
            # Move neighborhood
            for f in range(n_fac):
                if f == old_f:
                    continue
                if load[f] + d_i > capacities[f] + 1e-9:
                    continue
                delta = assign_cost[i, f] - assign_cost[i, old_f]
                if counts[f] == 0:
                    delta += open_cost[f]
                if counts[old_f] == 1:
                    delta -= open_cost[old_f]
                if delta < -1e-9:
                    # Apply move
                    load[old_f] -= d_i
                    load[f] += d_i
                    counts[old_f] -= 1
                    counts[f] += 1
                    res[i] = f
                    cur_cost += delta
                    improved = True
                    break
            
            # Swap neighborhood (sample a few)
            if not improved and counts[old_f] > 1:
                # Pick a random customer j from another facility
                other_facs = [f for f in range(n_fac) if f != old_f and counts[f] > 0]
                if other_facs:
                    f2 = other_facs[np.random.randint(len(other_facs))]
                    custs_at_f2 = np.where(res == f2)[0]
                    if len(custs_at_f2) > 0:
                        j = custs_at_f2[np.random.randint(len(custs_at_f2))]
                        d_j = demands[j]
                        # Check capacity
                        if load[old_f] - d_i + d_j <= capacities[old_f] + 1e-9 and \
                           load[f2] - d_j + d_i <= capacities[f2] + 1e-9:
                            delta = (assign_cost[i, f2] - assign_cost[i, old_f]) + \
                                    (assign_cost[j, old_f] - assign_cost[j, f2])
                            # Open cost changes
                            oc_delta = 0.0
                            new_count_old = counts[old_f] - 1 + 1
                            new_count_f2 = counts[f2] - 1 + 1
                            # Counts don't change in swap, so open costs stay same unless a facility becomes empty
                            # But since we're swapping one out and one in, counts stay the same.
                            if delta < -1e-9:
                                # Apply swap
                                res[i] = f2
                                res[j] = old_f
                                load[old_f] -= d_i
                                load[old_f] += d_j
                                load[f2] -= d_j
                                load[f2] += d_i
                                cur_cost += delta
                                improved = True
        
        if improved:
            no_improve = 0
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_res = res.copy()
        else:
            no_improve += 1
    
    if best_cost < cur_cost:
        return best_res, best_cost
    return res, cur_cost

def assign(demands, capacities, open_cost, assign_cost):
    start_time = time.time()
    deadline = start_time + 140.0
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    
    n_cust, n_fac = len(demands), len(capacities)
    
    best_result = None
    best_cost = np.inf
    
    # Phase 1: Lagrangian relaxation for a good initial guess
    # We solve the dual problem using subgradient method
    lam = np.zeros(n_fac)
    best_dual_obj = -np.inf
    best_lam = lam.copy()
    
    step_scale = 0.5
    for it in range(50):
        if time.time() > deadline - 60.0:
            break
        # Solve subproblem
        adj_cost = assign_cost + lam[None, :] * demands[:, None]
        sub_res = np.argmin(adj_cost, axis=1)
        
        # Check feasibility
        load = np.zeros(n_fac)
        np.add.at(load, sub_res, demands)
        
        # Dual objective
        dual_obj = np.sum(adj_cost[np.arange(n_cust), sub_res])
        used = np.where(np.bincount(sub_res, minlength=n_fac) > 0)[0]
        dual_obj += np.sum(open_cost[used])
        dual_obj -= np.dot(lam, capacities)
        
        if dual_obj > best_dual_obj:
            best_dual_obj = dual_obj
            best_lam = lam.copy()
        
        # If feasible, update best
        if np.all(load <= capacities + 1e-9):
            true_cost = _obj(demands, capacities, open_cost, assign_cost, sub_res)
            if true_cost < best_cost:
                best_cost = true_cost
                best_result = sub_res.copy()
        
        # Subgradient update
        grad = load - capacities
        norm_sq = np.dot(grad, grad)
        if norm_sq > 1e-12:
            step = step_scale * (best_cost - best_dual_obj) / norm_sq if best_cost < np.inf else step_scale / np.sqrt(norm_sq)
            step = min(step, 1.0)
            lam = np.maximum(lam + step * grad, 0.0)
        
        step_scale *= 0.95
    
    # If we have a feasible result from Lagrangian, use it
    if best_result is not None:
        res, cost = _local_search(demands, capacities, open_cost, assign_cost, best_result, time_limit=min(40.0, deadline - time.time() - 5.0))
        if cost < best_cost:
            best_cost = cost
            best_result = res
    
    # Phase 2: Multiple restarts with greedy init + local search
    remaining = deadline - time.time() - 5.0
    if remaining > 5.0:
        n_restarts = max(1, int(remaining / 15.0))
        for r in range(n_restarts):
            if time.time() > deadline - 5.0:
                break
            seed = r * 17 + 3
            init_res = _initial_solution(demands, capacities, open_cost, assign_cost, seed=seed)
            # Repair if infeasible
            load = np.zeros(n_fac)
            np.add.at(load, init_res, demands)
            for j in range(n_fac):
                if load[j] > capacities[j] + 1e-9:
                    custs = np.where(init_res == j)[0]
                    custs = custs[np.argsort(demands[custs])]
                    for i in custs:
                        if load[j] <= capacities[j] + 1e-9:
                            break
                        rem = capacities - load
                        rem[j] = -np.inf
                        feas = np.where(rem >= demands[i])[0]
                        if len(feas) > 0:
                            f = feas[np.argmin(rem[feas])]
                        else:
                            f = int(np.argmax(rem))
                        init_res[i] = f
                        load[j] -= demands[i]
                        load[f] += demands[i]
            
            t_lim = max(3.0, (deadline - time.time() - 5.0 - r) / max(1, n_restarts - r))
            res, cost = _local_search(demands, capacities, open_cost, assign_cost, init_res, time_limit=min(t_lim, 30.0))
            if cost < best_cost:
                best_cost = cost
                best_result = res
    
    # Final fallback
    if best_result is None:
        best_result = _initial_solution(demands, capacities, open_cost, assign_cost, seed=0)
        best_cost = _obj(demands, capacities, open_cost, assign_cost, best_result)
    
    # Final feasibility repair
    load = np.zeros(n_fac)
    np.add.at(load, best_result, demands)
    for j in range(n_fac):
        if load[j] > capacities[j] + 1e-9:
            custs = np.where(best_result == j)[0]
            custs = custs[np.argsort(demands[custs])]
            for i in custs:
                if load[j] <= capacities[j] + 1e-9:
                    break
                rem = capacities - load
                rem[j] = -np.inf
                feas = np.where(rem >= demands[i])[0]
                if len(feas) > 0:
                    f = feas[np.argmin(rem[feas])]
                else:
                    f = int(np.argmax(rem))
                best_result[i] = f
                load[j] -= demands[i]
                load[f] += demands[i]
    
    best_result = best_result.astype(int)
    for i in range(n_cust):
        if best_result[i] < 0 or best_result[i] >= n_fac:
            best_result[i] = 0
    
    return best_result
# EVOLVE-BLOCK-END

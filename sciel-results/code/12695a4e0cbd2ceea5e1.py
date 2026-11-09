# EVOLVE-BLOCK-START
import time
import numpy as np


def _objective(res, demands, capacities, open_cost, assign_cost, load):
    """Objective = open cost of used facilities + assignment cost. inf if infeasible."""
    if np.any(load > capacities):
        return float("inf")
    open_sum = float(open_cost[load > 0].sum())
    assign_sum = float(assign_cost[np.arange(len(res)), res].sum())
    return open_sum + assign_sum


def _greedy(demands, capacities, assign_cost):
    """Largest-demand-first, cheapest feasible facility."""
    n_cust = len(demands)
    res = np.zeros(n_cust, dtype=int)
    load = np.zeros(len(capacities))
    order = np.argsort(-demands)
    for i in order:
        d = demands[i]
        best_f = -1
        best_c = None
        for f in np.argsort(assign_cost[i]):
            if load[f] + d <= capacities[f]:
                best_f = f
                break
        if best_f < 0:
            # fallback: any facility with room
            for f in range(len(capacities)):
                if load[f] + d <= capacities[f]:
                    best_f = f
                    break
        res[i] = best_f
        load[best_f] += d
    return res, load


def assign(demands, capacities, open_cost, assign_cost):
    t0 = time.time()
    deadline = t0 + 110.0   # safety margin before the 120s kill (150s wall-clock)

    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    res, load = _greedy(demands, capacities, assign_cost)
    best_res = res.copy()
    best_obj = _objective(best_res, demands, capacities, open_cost, assign_cost, load)

    n_cust = len(demands)
    n_fac = len(capacities)
    by_fac = [list() for _ in range(n_fac)]
    for c in range(n_cust):
        by_fac[res[c]].append(c)

    # Precompute, for each customer, the sorted list of facilities by assignment cost.
    sorted_facs = np.argsort(assign_cost, axis=1)

    # 1-swap local search: move customer c from f_old to f_new (f_new != f_old).
    while time.time() < deadline:
        improved = False
        for c in range(n_cust):
            if time.time() > deadline:
                break
            d = demands[c]
            f_old = res[c]
            # candidate facilities: all except f_old, sorted by cost
            for f_new in sorted_facs[c]:
                if f_new == f_old:
                    continue
                # feasibility
                if load[f_new] + d > capacities[f_new]:
                    continue
                # objective delta
                delta = assign_cost[c, f_new] - assign_cost[c, f_old]
                if load[f_old] == d:      # f_old becomes empty
                    delta -= open_cost[f_old]
                if load[f_new] == 0:      # f_new becomes used
                    delta += open_cost[f_new]
                if delta < -1e-9:
                    # accept move
                    res[c] = f_new
                    load[f_old] -= d
                    load[f_new] += d
                    by_fac[f_old].remove(c)
                    by_fac[f_new].append(c)
                    best_obj += delta
                    improved = True
                    break
        if not improved:
            break

    if best_obj == float("inf") or np.any(load > capacities):
        return best_res
    return best_res
# EVOLVE-BLOCK-END

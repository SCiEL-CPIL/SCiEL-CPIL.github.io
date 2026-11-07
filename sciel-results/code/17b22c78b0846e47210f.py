# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cost(result, demands, capacities, open_cost, assign_cost):
    """Compute total cost of an assignment. Returns np.inf if infeasible."""
    n_cust = len(demands)
    n_fac = len(capacities)
    if len(result) != n_cust:
        return np.inf
    result = np.asarray(result)
    if np.any(result < 0) or np.any(result >= n_fac):
        return np.inf
    load = np.bincount(result, minlength=n_fac)
    if np.any(load > capacities):
        return np.inf
    open_mask = load > 0
    total = float(np.dot(open_mask, open_cost))
    total += float(np.sum(assign_cost[np.arange(n_cust), result]))
    return total


def _feasible_greedy(demands, capacities, open_cost, assign_cost, order=None):
    """Largest-first greedy. Returns (result, load)."""
    n_cust = len(demands)
    n_fac = len(capacities)
    if order is None:
        order = np.argsort(-demands)
    result = np.full(n_cust, -1, dtype=int)
    load = np.zeros(n_fac)
    for i in order:
        d = demands[i]
        # facilities that can take this customer
        ok = load + d <= capacities
        if not np.any(ok):
            # fallback: facility with most remaining capacity
            rem = capacities - load
            f = int(np.argmax(rem))
            load[f] += d
            result[i] = f
            continue
        # among feasible, pick min assign_cost
        costs = np.where(ok, assign_cost[i], np.inf)
        f = int(np.argmin(costs))
        load[f] += d
        result[i] = f
    return result, load


def _vectorized_move_pass(demands, capacities, open_cost, assign_cost, result, load):
    """One pass of first-improvement single-customer moves, vectorized per customer.
    Returns (result, load, improved)."""
    n_cust = len(demands)
    n_fac = len(capacities)
    improved = False
    for i in range(n_cust):
        cur = result[i]
        d = demands[i]
        # delta for moving i to facility f:
        #   assign_cost[i,f] - assign_cost[i,cur]
        #   + open_cost[f] if load[f]==0 else 0
        #   - open_cost[cur] if load[cur]==d else 0
        base = assign_cost[i] - assign_cost[i, cur]
        # open cost for target
        open_target = np.where(load == 0, open_cost, 0.0)
        # close cost for source (only if this is the last customer there)
        close_src = open_cost[cur] if load[cur] == d else 0.0
        delta = base + open_target - close_src
        # feasibility: load[f] + d <= capacities[f], and f != cur
        feas = (load + d <= capacities)
        feas[cur] = False
        if not np.any(feas):
            continue
        delta = np.where(feas, delta, np.inf)
        f = int(np.argmin(delta))
        if delta[f] < -1e-9:
            load[cur] -= d
            load[f] += d
            result[i] = f
            improved = True
    return result, load, improved


def _swap_pass(demands, capacities, open_cost, assign_cost, result, load):
    """First-improvement swap of two customers in different facilities.
    Vectorized over j for each i. Returns (result, load, improved)."""
    n_cust = len(demands)
    n_fac = len(capacities)
    improved = False
    for i in range(n_cust):
        fi = result[i]
        di = demands[i]
        # For each j != i with fj != fi, check swap feasibility and delta.
        fj_arr = result
        # only consider j where fj != fi
        mask = (fj_arr != fi)
        mask[i] = False
        if not np.any(mask):
            continue
        j_idx = np.where(mask)[0]
        fj = fj_arr[j_idx]
        dj = demands[j_idx]
        # feasibility:
        #   load[fi] - di + dj <= capacities[fi]
        #   load[fj] - dj + di <= capacities[fj]
        feas1 = load[fi] - di + dj <= capacities[fi]
        feas2 = load[fj] - dj + di <= capacities[fj]
        feas = feas1 & feas2
        if not np.any(feas):
            continue
        # delta in assignment cost:
        #   assign_cost[i, fj] + assign_cost[j, fi] - assign_cost[i, fi] - assign_cost[j, fj]
        ac_i_fj = assign_cost[i, fj]
        ac_j_fi = assign_cost[j_idx, fi]
        ac_i_fi = assign_cost[i, fi]
        ac_j_fj = assign_cost[j_idx, fj]
        delta = ac_i_fj + ac_j_fi - ac_i_fi - ac_j_fj
        # open cost changes:
        # after swap:
        #   new_load_fi = load[fi] - di + dj
        #   new_load_fj = load[fj] - dj + di
        # fi becomes empty if new_load_fi == 0 -> -open_cost[fi]
        # fj becomes empty if new_load_fj == 0 -> -open_cost[fj]
        # (both fi and fj were non-empty before since they held i and j)
        new_load_fi = load[fi] - di + dj
        new_load_fj = load[fj] - dj + di
        open_delta = -np.where(new_load_fi == 0, open_cost[fi], 0.0) - np.where(new_load_fj == 0, open_cost[fj], 0.0)
        delta = delta + open_delta
        delta = np.where(feas, delta, np.inf)
        k = int(np.argmin(delta))
        if delta[k] < -1e-9:
            j = j_idx[k]
            fjv = fj[k]
            # perform swap
            load[fi] = load[fi] - di + demands[j]
            load[fjv] = load[fjv] - demands[j] + di
            result[i] = fjv
            result[j] = fi
            improved = True
    return result, load, improved


def _relocate_facility_pass(demands, capacities, open_cost, assign_cost, result, load):
    """Try to close a facility by relocating all its customers to others.
    Greedy: for the most 'expensive to keep open' facility, try moving each customer
    to the cheapest feasible other facility. Only accept if total cost decreases.
    Returns (result, load, improved)."""
    n_cust = len(demands)
    n_fac = len(capacities)
    improved = False
    # consider facilities that are open
    open_fac = np.where(load > 0)[0]
    if len(open_fac) == 0:
        return result, load, improved
    # process in order of highest open_cost (most beneficial to close)
    for f in open_fac[np.argsort(-open_cost[open_fac])]:
        # customers at f
        custs = np.where(result == f)[0]
        if len(custs) == 0:
            continue
        # simulate: try to move all custs away from f
        sim_load = load.copy()
        sim_result = result.copy()
        sim_load[f] = 0
        feasible = True
        for i in custs:
            d = demands[i]
            ok = sim_load + d <= capacities
            ok[f] = False  # don't put back on f
            if not np.any(ok):
                feasible = False
                break
            costs = np.where(ok, assign_cost[i], np.inf)
            nf = int(np.argmin(costs))
            sim_load[nf] += d
            sim_result[i] = nf
        if not feasible:
            continue
        # compute cost delta
        # old: open_cost[f] + sum assign_cost[i, f] for i in custs
        old_part = open_cost[f] + float(np.sum(assign_cost[custs, f]))
        # new: open cost for newly opened facilities + sum assign_cost[i, sim_result[i]]
        new_open = 0.0
        for g in range(n_fac):
            if g == f:
                continue
            if load[g] == 0 and sim_load[g] > 0:
                new_open += open_cost[g]
        new_assign = float(np.sum(assign_cost[custs, sim_result[custs]]))
        # also, f is now closed so we remove its open cost (already in old_part)
        delta = (new_open + new_assign) - old_part
        if delta < -1e-9:
            load = sim_load
            result = sim_result
            improved = True
    return result, load, improved


def _local_search(demands, capacities, open_cost, assign_cost, result, load, time_limit):
    """Iterate move + swap + relocate passes until no improvement or time out."""
    start = time.time()
    result = result.copy()
    load = load.copy()
    while True:
        if time.time() - start > time_limit:
            break
        improved = False
        # single moves
        result, load, imp = _vectorized_move_pass(demands, capacities, open_cost, assign_cost, result, load)
        improved = improved or imp
        # swaps
        result, load, imp = _swap_pass(demands, capacities, open_cost, assign_cost, result, load)
        improved = improved or imp
        # relocate
        result, load, imp = _relocate_facility_pass(demands, capacities, open_cost, assign_cost, result, load)
        improved = improved or imp
        if not improved:
            break
    return result, load


def _perturb(demands, capacities, open_cost, assign_cost, result, load, rng, n_perturb):
    """Randomly reassign a few customers (2-opt style perturbation) to escape local optima."""
    n_cust = len(demands)
    n_fac = len(capacities)
    result = result.copy()
    load = load.copy()
    for _ in range(n_perturb):
        i = int(rng.randint(n_cust))
        cur = result[i]
        d = demands[i]
        load[cur] -= d
        ok = load + d <= capacities
        if not np.any(ok):
            # put back
            load[cur] += d
            continue
        # pick random feasible facility (not necessarily best)
        feas = np.where(ok)[0]
        f = feas[int(rng.randint(len(feas)))]
        load[f] += d
        result[i] = f
    return result, load


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    open_cost = np.asarray(open_cost)
    assign_cost = np.asarray(assign_cost)
    n_cust = len(demands)
    n_fac = len(capacities)

    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_result = None
    best_cost = np.inf

    # Phase 1: multiple diverse greedy initializations
    base_order = np.argsort(-demands)
    seeds = [0, 1, 2, 3, 4]
    for s in seeds:
        if time.time() > deadline - 30.0:
            break
        rng = np.random.RandomState(s)
        # slightly perturb order for diversity
        order = base_order.copy()
        # swap a few elements
        for _ in range(5):
            a, b = rng.randint(n_cust), rng.randint(n_cust)
            order[a], order[b] = order[b], order[a]
        result, load = _feasible_greedy(demands, capacities, open_cost, assign_cost, order=order)
        cost = _compute_cost(result, demands, capacities, open_cost, assign_cost)
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()

    if best_result is None:
        # fallback
        result, load = _feasible_greedy(demands, capacities, open_cost, assign_cost)
        return result

    # Phase 2: local search on best
    if time.time() < deadline - 10.0:
        tlimit = deadline - 10.0 - time.time()
        best_result, best_load = _local_search(demands, capacities, open_cost, assign_cost,
                                               best_result,
                                               np.bincount(best_result, minlength=n_fac),
                                               tlimit)
        best_cost = _compute_cost(best_result, demands, capacities, open_cost, assign_cost)

    # Phase 3: iterated local search (perturb + local search) with remaining time
    rng = np.random.RandomState(12345)
    best_load = np.bincount(best_result, minlength=n_fac)
    while time.time() < deadline - 5.0:
        # perturb
        n_pert = int(np.clip(n_cust * 0.1, 3, 15))
        pert_result, pert_load = _perturb(demands, capacities, open_cost, assign_cost,
                                          best_result, best_load, rng, n_pert)
        # quick local search with limited time
        tlimit = min(20.0, deadline - 5.0 - time.time())
        if tlimit < 1.0:
            break
        ls_result, ls_load = _local_search(demands, capacities, open_cost, assign_cost,
                                           pert_result, pert_load, tlimit)
        ls_cost = _compute_cost(ls_result, demands, capacities, open_cost, assign_cost)
        if ls_cost < best_cost - 1e-9:
            best_cost = ls_cost
            best_result = ls_result.copy()
            best_load = ls_load.copy()

    return best_result
# EVOLVE-BLOCK-END

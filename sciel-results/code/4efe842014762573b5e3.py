"""Capacitated facility location: k-median with capacity + local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _total_cost(demands, capacities, open_cost, assign_cost, assign):
    assign = np.asarray(assign)
    cost = assign_cost[np.arange(len(demands)), assign].sum()
    used = np.zeros(len(capacities), dtype=bool)
    for f in assign:
        used[f] = True
    cost += open_cost[used].sum()
    return cost


def _greedy_assign(demands, assign_cost, caps, order):
    """Assign customers in given order to cheapest facility with remaining capacity."""
    n = len(demands)
    assign = np.full(n, -1, dtype=int)
    load = np.zeros(len(caps), dtype=float)
    for i in order:
        d = demands[i]
        # feasible facilities
        feas = load + d <= caps
        if not np.any(feas):
            # cannot fit anywhere; pick least-overloaded (shouldn't happen)
            f = int(np.argmin(load))
        else:
            costs = np.where(feas, assign_cost[i], np.inf)
            f = int(np.argmin(costs))
        assign[i] = f
        load[f] += d
    return assign, load


def _local_search(demands, capacities, open_cost, assign_cost, assign, load, deadline, iters=200):
    """Local search: try moving a customer to another facility if it improves total cost."""
    n = len(demands)
    m = len(capacities)
    best = _total_cost(demands, capacities, open_cost, assign_cost, assign)
    used = np.zeros(m, dtype=bool)
    for f in assign:
        used[f] = True
    open_sum = open_cost[used].sum()
    # Precompute per-customer current cost
    cur_cost = assign_cost[np.arange(n), assign].copy()
    assign_sum = cur_cost.sum()

    for _ in range(iters):
        if time.time() > deadline:
            break
        improved = False
        # iterate customers in random order
        idx = np.random.permutation(n)
        for i in idx:
            if time.time() > deadline:
                break
            f_old = assign[i]
            d = demands[i]
            # best alternative facility
            # cost delta if move to f: assign_cost[i,f] - assign_cost[i,f_old] + open delta
            # open delta: if f_old becomes empty -> -open_cost[f_old]; if f_new was empty -> +open_cost[f_new]
            # find best f_new
            base = assign_cost[i].copy()
            # open cost adjustments
            # if f_old loses its last customer, we save open_cost[f_old]
            old_empty_after = (load[f_old] - d) <= 0
            # compute effective delta per facility
            delta = base - cur_cost[i]
            # facilities where moving makes them empty (currently 0 load)
            empty_new = load <= 0
            # if moving to a facility currently empty, add open_cost
            delta = delta + np.where(empty_new, open_cost, 0.0)
            # if old facility becomes empty, subtract open_cost[f_old] (constant across f_new)
            if old_empty_after:
                delta = delta - open_cost[f_old]
            # exclude current facility
            delta[f_old] = np.inf
            # capacity feasibility: load[f] + d <= capacities[f]
            delta[load + d > capacities] = np.inf
            f_new = int(np.argmin(delta))
            if np.isinf(delta[f_new]):
                continue
            new_delta = delta[f_new]
            if new_delta < -1e-9:
                # perform move
                load[f_old] -= d
                load[f_new] += d
                assign[i] = f_new
                cur_cost[i] = assign_cost[i, f_new]
                # update used/open
                if load[f_old] <= 0:
                    used[f_old] = False
                if load[f_new] > 0 and not used[f_new]:
                    used[f_new] = True
                best += new_delta
                improved = True
        if not improved:
            break
    return assign, load, best


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    n = len(demands)
    m = len(capacities)

    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    # --- Phase 1: choose which facilities to open (k-median style) ---
    # We'll do a few restarts with different facility subsets and local search.
    # Strategy: try opening all facilities (ignore open cost) as a baseline,
    # then try subsets chosen by a greedy set-cover / Lagrangian heuristic.

    best_assign = None
    best_val = np.inf

    # Baseline: open all, greedy largest-first
    order = np.argsort(-demands)
    a, l = _greedy_assign(demands, assign_cost, capacities, order)
    if np.all(a >= 0):
        v = _total_cost(demands, capacities, open_cost, assign_cost, a)
        if v < best_val:
            best_val = v
            best_assign = a.copy()

    # Local search on baseline
    if time.time() < deadline:
        a, l, v = _local_search(demands, capacities, open_cost, assign_cost, a, l, deadline, iters=150)
        if v < best_val:
            best_val = v
            best_assign = a.copy()

    # --- Phase 2: facility subset selection via greedy + local search ---
    # Try to find a good subset of facilities to open.
    # Approach: start with all open, iteratively drop the most expensive facility
    # (reassigning its customers), if it doesn't violate capacity, then local search.

    if time.time() < deadline - 5.0:
        # Start from best_assign, try dropping facilities
        cur = best_assign.copy()
        load = np.zeros(m, dtype=float)
        for f in cur:
            load[f] += demands[f if False else 0]  # placeholder
        # recompute load properly
        load = np.zeros(m, dtype=float)
        for i in range(n):
            load[cur[i]] += demands[i]
        used = np.zeros(m, dtype=bool)
        for f in cur:
            used[f] = True

        # Try dropping each open facility (in order of high open_cost)
        open_fac = np.where(used)[0]
        open_fac = open_fac[np.argsort(-open_cost[open_fac])]
        for f_drop in open_fac:
            if time.time() > deadline - 3.0:
                break
            # reassign customers of f_drop to other facilities
            custs = np.where(cur == f_drop)[0]
            if len(custs) == 0:
                continue
            # try to move each to cheapest other feasible facility
            ok = True
            new_assign = cur.copy()
            new_load = load.copy()
            for i in custs:
                d = demands[i]
                # feasible: other facilities with capacity
                feas = (new_load + d <= capacities)
                feas[f_drop] = False
                if not np.any(feas):
                    ok = False
                    break
                costs = np.where(feas, assign_cost[i], np.inf)
                f_new = int(np.argmin(costs))
                new_assign[i] = f_new
                new_load[f_new] += d
            if not ok:
                continue
            new_load[f_drop] = 0
            # evaluate
            v = _total_cost(demands, capacities, open_cost, assign_cost, new_assign)
            if v < best_val - 1e-9:
                best_val = v
                best_assign = new_assign.copy()
                cur = new_assign
                load = new_load
                used = np.zeros(m, dtype=bool)
                for f in cur:
                    used[f] = True

        # local search again
        if time.time() < deadline - 2.0:
            a, l, v = _local_search(demands, capacities, open_cost, assign_cost, cur, load, deadline, iters=100)
            if v < best_val:
                best_val = v
                best_assign = a.copy()

    # --- Phase 3: random restarts with different initial subsets ---
    if time.time() < deadline - 5.0:
        for restart in range(3):
            if time.time() > deadline - 3.0:
                break
            # randomly open a subset: keep facilities that are "good"
            # heuristic: open facilities with low open_cost and high centrality
            # simple: random subset of size between m//2 and m
            k = np.random.randint(m // 2, m + 1)
            subset = np.random.choice(m, size=k, replace=False)
            sub_caps = capacities[subset]
            # greedy assign only to subset facilities
            # map: only allow subset
            allowed = np.zeros(m, dtype=bool)
            allowed[subset] = True
            order = np.argsort(-demands)
            a = np.full(n, -1, dtype=int)
            load = np.zeros(m, dtype=float)
            ok = True
            for i in order:
                d = demands[i]
                feas = allowed & (load + d <= capacities)
                if not np.any(feas):
                    ok = False
                    break
                costs = np.where(feas, assign_cost[i], np.inf)
                f = int(np.argmin(costs))
                a[i] = f
                load[f] += d
            if not ok:
                continue
            v = _total_cost(demands, capacities, open_cost, assign_cost, a)
            if v < best_val:
                best_val = v
                best_assign = a.copy()
            # local search
            if time.time() < deadline - 2.0:
                a, l, v = _local_search(demands, capacities, open_cost, assign_cost, a, load, deadline, iters=80)
                if v < best_val:
                    best_val = v
                    best_assign = a.copy()

    # --- Final: ensure validity, return ---
    if best_assign is None:
        # fallback: assign all to facility 0 (may be invalid but better than nothing)
        best_assign = np.zeros(n, dtype=int)

    # Final local search if time remains
    if time.time() < deadline - 1.0:
        load = np.zeros(m, dtype=float)
        for i in range(n):
            load[best_assign[i]] += demands[i]
        a, l, v = _local_search(demands, capacities, open_cost, assign_cost, best_assign, load, deadline, iters=50)
        if v < best_val:
            best_assign = a

    # Validate: clip to range, ensure capacity (should already be)
    best_assign = np.clip(best_assign, 0, m - 1).astype(int)

    return best_assign
# EVOLVE-BLOCK-END

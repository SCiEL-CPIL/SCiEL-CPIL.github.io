# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 140.0  # safety margin below the 150s wall-clock kill


def _init_solution(demands, capacities, open_cost, assign_cost):
    n_cust = len(demands)
    n_fac = len(capacities)
    result = np.zeros(n_cust, dtype=int)
    load = np.zeros(n_fac, dtype=float)
    order = np.argsort(-demands, kind='stable')
    for i in order:
        d = demands[i]
        # feasible facilities
        feas = np.where(load + d <= capacities)[0]
        if len(feas) == 0:
            # pick facility with most remaining capacity
            feas = np.argsort(capacities - load)[:5]
        # effective cost = assign_cost + small share of open cost
        eff = assign_cost[i, feas] + open_cost[feas] / max(capacities[feas], 1)
        f = feas[np.argmin(eff)]
        load[f] += d
        result[i] = f
    return result, load


def _total_cost(result, demands, capacities, open_cost, assign_cost):
    n_fac = len(capacities)
    load = np.bincount(result, minlength=n_fac).astype(float)
    used = load > 0
    open_sum = float(np.sum(open_cost[used]))
    asg_sum = float(np.sum(assign_cost[np.arange(len(result)), result]))
    return open_sum + asg_sum, load


def _local_search(result, demands, capacities, open_cost, assign_cost, deadline):
    n_cust = len(result)
    n_fac = len(capacities)
    load = np.bincount(result, minlength=n_fac).astype(float)
    used = load > 0
    total, _ = _total_cost(result, demands, capacities, open_cost, assign_cost)

    # precompute sorted facility order per customer by assign_cost
    order = np.argsort(assign_cost, axis=1, kind='stable')

    no_improve = 0
    while time.time() < deadline:
        improved = False

        # --- Single moves ---
        for i in range(n_cust):
            if time.time() >= deadline:
                break
            f0 = result[i]
            d = demands[i]
            # try facilities in increasing assign_cost, skip current
            for f in order[i]:
                if f == f0:
                    continue
                if load[f] + d > capacities[f]:
                    continue
                delta = assign_cost[i, f] - assign_cost[i, f0]
                # closing f0 saves open_cost[f0]
                if load[f0] == d:
                    delta -= open_cost[f0]
                if delta < -1e-9:
                    load[f0] -= d
                    load[f] += d
                    result[i] = f
                    total += delta
                    improved = True
                    break
        if time.time() >= deadline:
            break

        # --- Swap moves (two customers trade facilities) ---
        for i in range(n_cust):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n_cust):
                f_i = result[i]
                f_j = result[j]
                if f_i == f_j:
                    continue
                di = demands[i]
                dj = demands[j]
                # capacity checks
                if load[f_i] - di + dj > capacities[f_i]:
                    continue
                if load[f_j] - dj + di > capacities[f_j]:
                    continue
                # cost delta
                delta = (assign_cost[i, f_j] + assign_cost[j, f_i]
                         - assign_cost[i, f_i] - assign_cost[j, f_j])
                # open cost changes: facility becomes unused?
                if load[f_i] == di:  # f_i would become empty
                    delta += open_cost[f_i]
                elif load[f_i] == di and load[f_j] == dj:
                    pass
                if load[f_j] == dj:  # f_j would become empty
                    delta += open_cost[f_j]
                # if a facility gains a customer it was empty, we pay open cost
                if load[f_i] - di == 0 and dj > 0:
                    # f_i was going to be empty, now gets dj -> but we already added open above if it was empty
                    # Actually if load[f_i]==di then after removing i it's empty; adding j makes it non-empty.
                    # We added open_cost[f_i] for closing, but it's not closing now.
                    delta -= open_cost[f_i]
                if load[f_j] - dj == 0 and di > 0:
                    delta -= open_cost[f_j]

                if delta < -1e-9:
                    load[f_i] = load[f_i] - di + dj
                    load[f_j] = load[f_j] - dj + di
                    result[i] = f_j
                    result[j] = f_i
                    total += delta
                    improved = True
                    break
            if time.time() >= deadline:
                break
        if time.time() >= deadline:
            break

        # --- Close-facility move: move all customers of one facility elsewhere ---
        for f in range(n_fac):
            if time.time() >= deadline:
                break
            if load[f] == 0:
                continue
            # try to relocate all customers of f
            custs = np.where(result == f)[0]
            if len(custs) == 0:
                continue
            # compute new load without f
            new_load = load.copy()
            new_load[f] = 0
            ok = True
            new_result = result.copy()
            extra_cost = 0.0
            for i in custs:
                d = demands[i]
                # find best feasible facility (not f)
                feas = np.where(new_load + d <= capacities)[0]
                feas = feas[feas != f]
                if len(feas) == 0:
                    ok = False
                    break
                eff = assign_cost[i, feas] + open_cost[feas] / max(capacities[feas], 1)
                nf = feas[np.argmin(eff)]
                new_load[nf] += d
                new_result[i] = nf
                extra_cost += assign_cost[i, nf] - assign_cost[i, f]
            if not ok:
                continue
            # open cost: f is closed (saves open_cost[f]); new facilities may be opened
            new_used = new_load > 0
            old_used = load > 0
            delta_open = float(np.sum(open_cost[new_used & ~old_used])) - float(np.sum(open_cost[old_used & ~new_used]))
            delta = extra_cost + delta_open
            if delta < -1e-9:
                result = new_result
                load = new_load
                total += delta
                improved = True

        if not improved:
            no_improve += 1
            if no_improve >= 3:
                break
        else:
            no_improve = 0

    return result, total


def assign(demands, capacities, open_cost, assign_cost):
    t0 = time.time()
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)

    best_result, best_total = _init_solution(demands, capacities, open_cost, assign_cost)
    best_result, best_total = _local_search(
        best_result.copy(), demands, capacities, open_cost, assign_cost,
        _DEADLINE)

    # If time remains, try a few more restarts with perturbation
    while time.time() < _DEADLINE - 5:
        # perturb: random reassign of a few customers
        pert = best_result.copy()
        n_cust = len(pert)
        n_swap = max(1, n_cust // 20)
        idx = np.random.choice(n_cust, size=n_swap, replace=False)
        load = np.bincount(pert, minlength=len(capacities)).astype(float)
        for i in idx:
            feas = np.where(load + demands[i] <= capacities)[0]
            if len(feas) == 0:
                feas = np.argsort(capacities - load)[:3]
            f = feas[np.random.randint(len(feas))]
            load[pert[i]] -= demands[i]
            load[f] += demands[i]
            pert[i] = f
        pert, ptot = _local_search(pert, demands, capacities, open_cost, assign_cost, _DEADLINE)
        if ptot < best_total:
            best_result, best_total = pert, ptot

    return best_result.astype(int)
# EVOLVE-BLOCK-END

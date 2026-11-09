"""Capacitated facility location: Lagrangian subgradient + repair + local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import distance


def _lagrangian_init(demands, capacities, open_cost, assign_cost, deadline, rng):
    n, m = assign_cost.shape
    best_sol = None
    best_cost = np.inf
    lam = np.zeros(m)
    last_improve = 0.0
    best_lam = lam.copy()
    step = 1.0
    for _ in range(3000):
        if time.time() > deadline - 3.0:
            break
        # primal subproblem: each customer picks min over f of (assign + lam_f),
        # paying open_cost if f is used.
        adjusted = assign_cost + lam
        choice = np.argmin(adjusted, axis=1)
        used = np.zeros(m, dtype=bool)
        used[choice] = True
        if not np.all(used):
            oc = np.where(used, open_cost, 0.0)
            adjusted = assign_cost + lam + oc
            choice = np.argmin(adjusted, axis=1)
        load = np.zeros(m)
        np.add.at(load, choice, demands)
        # dual objective = sum_c min_f(...)  (lower bound)
        dual_obj = np.sum(np.min(adjusted, axis=1))
        # subgradient of dual w.r.t. lam_f = cap_f - load_f
        grad = capacities - load
        # step size
        if best_cost > dual_obj:
            best_cost = dual_obj
            last_improve = time.time()
        gap = best_cost - dual_obj
        if gap > 0:
            step = 2.0 * gap / (np.dot(grad, grad) + 1e-12)
        lam = np.maximum(lam + step * grad, 0.0)
        # periodically check feasibility of the primal solution
        if _ in range(0, 3000, 25):
            if np.all(load <= capacities + 1e-9):
                cost = np.sum(oc) + np.sum(assign_cost[np.arange(n), choice])
                if cost < best_sol_cost if (best_sol is not None and 'best_sol_cost' in dir()) else True:
                    pass
            # track best feasible
            if np.all(load <= capacities + 1e-9):
                c = np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n), choice])
                if c < (best_sol_cost if best_sol is not None else np.inf):
                    best_sol = choice.copy()
                    best_sol_cost = c
    return best_sol


def _repair(choice, demands, capacities, open_cost, assign_cost, deadline, rng):
    n, m = assign_cost.shape
    choice = choice.copy()
    load = np.zeros(m)
    np.add.at(load, choice, demands)
    # while any facility over capacity, move its heaviest customer to the cheapest
    # feasible facility (or the facility with most free capacity).
    over = load > capacities + 1e-9
    guard = 0
    while np.any(over) and guard < 5000 and time.time() < deadline - 1.0:
        guard += 1
        over = load > capacities + 1e-9
        if not np.any(over):
            break
        bad_f = np.where(over)[0]
        # pick a customer in the most-overloaded facility
        f = bad_f[np.argmax(load[bad_f])]
        custs = np.where(choice == f)[0]
        # move the heaviest customer
        c = custs[np.argmax(demands[custs])]
        # find feasible facility (cheapest assign cost among those with room)
        room = capacities - load
        room[f] = -1  # exclude current
        cand = np.where(room >= demands[c])[0]
        if len(cand) == 0:
            # fallback: facility with the most free space
            cand = np.where(room >= 0)[0]
            if len(cand) == 0:
                break
            nf = cand[np.argmax(room[cand])]
        else:
            nf = cand[np.argmin(assign_cost[c, cand])]
        load[f] -= demands[c]
        load[nf] += demands[c]
        choice[c] = nf
    # if still infeasible (shouldn't happen), brute force last resort
    if np.any(load > capacities + 1e-9):
        # assign one-by-one greedily
        choice = np.zeros(n, dtype=int)
        load = np.zeros(m)
        order = np.argsort(-demands)
        for c in order:
            room = capacities - load
            cand = np.where(room >= demands[c])[0]
            if len(cand) == 0:
                nf = np.argmax(room)
            else:
                nf = cand[np.argmin(assign_cost[c, cand])]
            load[nf] += demands[c]
            choice[c] = nf
    return choice, load


def _local_search(choice, demands, capacities, open_cost, assign_cost, deadline, rng):
    n, m = assign_cost.shape
    choice = choice.copy()
    load = np.zeros(m)
    np.add.at(load, choice, demands)

    def total_cost():
        used = np.zeros(m, dtype=bool)
        used[choice] = True
        return np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n), choice])

    best = total_cost()
    improved = True
    passes = 0
    while improved and time.time() < deadline - 2.0 and passes < 60:
        passes += 1
        improved = False
        # single moves
        for c in np.random.permutation(n):
            if time.time() > deadline - 2.0:
                break
            f0 = choice[c]
            d = demands[c]
            # candidate facilities: those with room after removing c from f0
            room = capacities - load
            room[f0] += d  # we free d if we move away
            cand = np.where(room >= d)[0]
            cand = cand[cand != f0]
            if len(cand) == 0:
                continue
            costs = assign_cost[c, cand]
            # account for opening cost change
            oc_delta = np.zeros(len(cand))
            used = np.zeros(m, dtype=bool)
            used[choice] = True
            for j, nf in enumerate(cand):
                if used[f0] and load[f0] - d <= 0:
                    oc_delta[j] -= open_cost[f0]
                if not used[nf]:
                    oc_delta[j] += open_cost[nf]
            deltas = costs - assign_cost[c, f0] + oc_delta
            j = int(np.argmin(deltas))
            if deltas[j] < -1e-9:
                nf = cand[j]
                load[f0] -= d
                load[nf] += d
                choice[c] = nf
                newc = total_cost()
                if newc < best:
                    best = newc
                    improved = True
        # 2-swaps
        if time.time() > deadline - 2.0:
            break
        for _ in range(400):
            if time.time() > deadline - 2.0:
                break
            c1, c2 = rng.choice(n, 2, replace=False)
            f1, f2 = choice[c1], choice[c2]
            if f1 == f2:
                continue
            d1, d2 = demands[c1], demands[c2]
            # after swap: f1 loses d1 gains d2 ; f2 loses d2 gains d1
            if load[f1] - d1 + d2 > capacities[f1] + 1e-9:
                continue
            if load[f2] - d2 + d1 > capacities[f2] + 1e-9:
                continue
            used = np.zeros(m, dtype=bool)
            used[choice] = True
            delta = (assign_cost[c1, f2] + assign_cost[c2, f1]
                     - assign_cost[c1, f1] - assign_cost[c2, f2])
            # opening cost change
            if used[f1] and load[f1] - d1 + d2 <= 0:
                delta -= open_cost[f1]
            if used[f2] and load[f2] - d2 + d1 <= 0:
                delta -= open_cost[f2]
            if not used[f1] and load[f1] - d1 + d2 > 0:
                delta += open_cost[f1]
            if not used[f2] and load[f2] - d2 + d1 > 0:
                delta += open_cost[f2]
            if delta < -1e-9:
                load[f1] += d2 - d1
                load[f2] += d1 - d2
                choice[c1], choice[c2] = f2, f1
                newc = total_cost()
                if newc < best:
                    best = newc
                    improved = True
    return choice


def assign(demands, capacities, open_cost, assign_cost):
    t0 = time.time()
    deadline = t0 + 135.0
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    n, m = assign_cost.shape
    rng = np.random.default_rng(12345)

    # candidate pool
    candidates = []

    # 1) greedy largest-first (baseline)
    def greedy(seed_rng):
        res = np.zeros(n, dtype=int)
        load = np.zeros(m)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f] + 1e-9:
                    load[f] += demands[i]
                    res[i] = f
                    break
        return res
    candidates.append(greedy(rng))

    # 2) Lagrangian subgradient
    try:
        lag = _lagrangian_init(demands, capacities, open_cost, assign_cost,
                               deadline, rng)
        if lag is not None:
            candidates.append(lag)
    except Exception:
        pass

    # 3) random-feasible
    def random_feasible():
        res = np.zeros(n, dtype=int)
        load = np.zeros(m)
        order = np.argsort(-demands)
        for c in order:
            room = capacities - load
            cand = np.where(room >= demands[c])[0]
            if len(cand) == 0:
                nf = int(np.argmax(room))
            else:
                # pick among feasible with some randomness
                w = 1.0 / (assign_cost[c, cand] + 1e-6)
                nf = cand[rng.choice(len(cand), p=w / w.sum())]
            load[nf] += demands[c]
            res[c] = nf
        return res
    for _ in range(2):
        candidates.append(random_feasible())

    # repair + local search each candidate, keep best
    best = None
    best_cost = np.inf
    for cand in candidates:
        if time.time() > deadline - 6.0:
            break
        rep, load = _repair(cand, demands, capacities, open_cost, assign_cost,
                            deadline, rng)
        if not np.all(load <= capacities + 1e-9):
            continue
        sol = _local_search(rep, demands, capacities, open_cost, assign_cost,
                            deadline, rng)
        used = np.zeros(m, dtype=bool)
        used[sol] = True
        cost = np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(n), sol])
        if cost < best_cost:
            best_cost = cost
            best = sol
    if best is None:
        best = candidates[0]
    return np.asarray(best, dtype=int)
# EVOLVE-BLOCK-END

"""Capacitated facility location: greedy init + vectorized 1-move / 2-swap local search with restarts."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random


def _compute_total(assign_vec, demands, open_cost, assign_cost):
    used = np.zeros(len(open_cost), dtype=bool)
    used[assign_vec] = True
    open_part = open_cost[used].sum()
    assign_part = assign_cost[np.arange(len(assign_vec)), assign_vec].sum()
    return open_part + assign_part


def _greedy_init(demands, capacities, open_cost, assign_cost, rng, mode):
    n = len(demands)
    m = len(capacities)
    result = np.zeros(n, dtype=int)
    load = np.zeros(m)
    if mode == 0:
        order = np.argsort(-demands)
    elif mode == 1:
        order = np.arange(n)
        rng.shuffle(order.tolist())
    else:
        order = np.argsort(-demands)
    sorted_f = np.argsort(assign_cost, axis=1)
    for i in order:
        d = demands[i]
        placed = False
        for f in sorted_f[i]:
            if load[f] + d <= capacities[f]:
                load[f] += d
                result[i] = f
                placed = True
                break
        if not placed:
            rem = capacities - load
            f = int(np.argmax(rem))
            load[f] += d
            result[i] = f
    return result


def _local_search_1move(assign_vec, demands, capacities, open_cost, assign_cost, rng, iters=40):
    n = len(assign_vec)
    m = len(capacities)
    assign_vec = assign_vec.copy()
    load = np.bincount(assign_vec, minlength=m).astype(float)
    np.add.at(load, assign_vec, demands)
    improved = True
    it = 0
    while improved and it < iters:
        it += 1
        improved = False
        order = np.arange(n)
        rng.shuffle(order.tolist())
        for i in order:
            d = demands[i]
            c = assign_vec[i]
            feasible = (load + d) <= capacities
            feasible[c] = False
            if not feasible.any():
                continue
            cost_to = assign_cost[i].copy()
            cost_to[~feasible] = np.inf
            f = int(np.argmin(cost_to))
            new_cost_i = cost_to[f]
            delta = new_cost_i - assign_cost[i, c]
            if load[c] == d:
                delta -= open_cost[c]
            if load[f] == 0:
                delta += open_cost[f]
            if delta < -1e-9:
                load[c] -= d
                load[f] += d
                assign_vec[i] = f
                improved = True
    return assign_vec


def _local_search_2swap(assign_vec, demands, capacities, open_cost, assign_cost, rng, iters=15):
    n = len(assign_vec)
    m = len(capacities)
    assign_vec = assign_vec.copy()
    load = np.bincount(assign_vec, minlength=m).astype(float)
    np.add.at(load, assign_vec, demands)
    improved = True
    it = 0
    while improved and it < iters:
        it += 1
        improved = False
        groups = {f: np.where(assign_vec == f)[0] for f in range(m) if np.any(assign_vec == f)}
        facility_list = list(groups.keys())
        rng.shuffle(facility_list)
        for a in facility_list:
            ga = groups[a]
            if len(ga) == 0:
                continue
            for b in facility_list:
                if b == a:
                    continue
                gb = groups[b]
                if len(gb) == 0:
                    continue
                di = demands[ga][:, None]
                dj = demands[gb][None, :]
                feasible = (load[a] - di + dj) <= capacities[a]
                feasible = feasible & ((load[b] - dj + di) <= capacities[b])
                cost_gain = (assign_cost[ga][:, None, b] + assign_cost[gb][None, :, a]
                             - assign_cost[ga][:, None, a] - assign_cost[gb][None, :, b])
                mask = feasible & (cost_gain > 1e-9)
                if not mask.any():
                    continue
                gain_mat = np.where(mask, cost_gain, -np.inf)
                idx = np.unravel_index(np.argmax(gain_mat), gain_mat.shape)
                i = ga[idx[0]]
                j = gb[idx[1]]
                assign_vec[i] = b
                assign_vec[j] = a
                load[a] = load[a] - demands[i] + demands[j]
                load[b] = load[b] - demands[j] + demands[i]
                improved = True
    return assign_vec


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    n = len(demands)
    m = len(capacities)

    start_time = time.time()
    deadline = start_time + 140.0

    rng = random.Random(12345)

    best_assign = None
    best_total = np.inf

    def consider(a):
        nonlocal best_assign, best_total
        if np.any(a < 0) or np.any(a >= m):
            return
        load = np.bincount(a, minlength=m).astype(float)
        np.add.at(load, a, demands)
        if np.any(load > capacities + 1e-6):
            return
        t = _compute_total(a, demands, open_cost, assign_cost)
        if t < best_total - 1e-9:
            best_total = t
            best_assign = a.copy()

    for mode in range(3):
        for _ in range(4):
            if time.time() > deadline:
                break
            a = _greedy_init(demands, capacities, open_cost, assign_cost, rng, mode)
            a = _local_search_1move(a, demands, capacities, open_cost, assign_cost, rng, iters=40)
            a = _local_search_2swap(a, demands, capacities, open_cost, assign_cost, rng, iters=15)
            a = _local_search_1move(a, demands, capacities, open_cost, assign_cost, rng, iters=40)
            consider(a)

    while time.time() < deadline:
        if best_assign is None:
            a = _greedy_init(demands, capacities, open_cost, assign_cost, rng, rng.randint(0, 2))
        else:
            a = best_assign.copy()
            k = rng.randint(3, 12)
            idx = rng.sample(range(n), k)
            load = np.bincount(a, minlength=m).astype(float)
            np.add.at(load, a, demands)
            for i in idx:
                d = demands[i]
                feasible = (load + d) <= capacities
                if feasible.any():
                    f = int(rng.choice(np.where(feasible)[0].tolist()))
                    a[i] = f
        a = _local_search_1move(a, demands, capacities, open_cost, assign_cost, rng, iters=50)
        a = _local_search_2swap(a, demands, capacities, open_cost, assign_cost, rng, iters=20)
        a = _local_search_1move(a, demands, capacities, open_cost, assign_cost, rng, iters=50)
        consider(a)
        if time.time() > deadline - 5.0:
            break

    if best_assign is None:
        best_assign = _greedy_init(demands, capacities, open_cost, assign_cost, rng, 0)
        consider(best_assign)

    return best_assign.astype(int)
# EVOLVE-BLOCK-END

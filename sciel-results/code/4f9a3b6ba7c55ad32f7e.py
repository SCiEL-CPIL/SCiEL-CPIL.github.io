# EVOLVE-BLOCK-START
import numpy as np
import time


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    n_cust = len(demands)
    n_fac = len(capacities)

    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before the 150s kill

    # ---------- helpers ----------
    def calc_cost(res):
        used = np.unique(res)
        if len(used) == 0:
            return 0.0
        open_sum = float(np.sum(open_cost[used]))
        assign_sum = float(np.sum(assign_cost[np.arange(n_cust), res]))
        return open_sum + assign_sum

    def loads_of(res):
        load = np.zeros(n_fac, dtype=np.int64)
        np.add.at(load, res, demands)
        return load

    def _greedy(order, allowed):
        """Greedy assign customers (in `order`) to cheapest feasible facility in `allowed`."""
        allowed_arr = np.asarray(allowed, dtype=np.int64)
        n_a = len(allowed_arr)
        if n_a == 0:
            return None
        if int(np.sum(capacities[allowed_arr])) < int(np.sum(demands)):
            return None
        load = np.zeros(n_a, dtype=np.int64)
        res = np.full(n_cust, -1, dtype=np.int64)
        sub_cost = assign_cost[:, allowed_arr]
        sorted_f = np.argsort(sub_cost, axis=1)
        for i in order:
            placed = False
            for fl in sorted_f[i]:
                if load[fl] + demands[i] <= capacities[allowed_arr[fl]]:
                    load[fl] += demands[i]
                    res[i] = allowed_arr[fl]
                    placed = True
                    break
            if not placed:
                return None
        return res

    def local_search(res, load, max_time):
        """Best-improvement single moves + sampled swaps + facility-closing.
        Accounts for open-cost changes (facilities becoming empty / newly opened)."""
        res = res.copy()
        load = load.copy()
        t0 = time.time()
        rng = np.random.default_rng(12345)

        def feasible_targets(i, exclude):
            d_i = demands[i]
            mask = (load + d_i <= capacities)
            mask[exclude] = False
            return np.where(mask)[0]

        improved = True
        while improved and (time.time() - t0) < max_time and time.time() < deadline - 3.0:
            improved = False

            # --- 1. Best-improvement single moves ---
            for i in range(n_cust):
                if time.time() > deadline - 3.0:
                    break
                f_old = res[i]
                d_i = demands[i]
                feasible = feasible_targets(i, f_old)
                if len(feasible) == 0:
                    continue
                base = assign_cost[i, f_old]
                deltas = assign_cost[i, feasible] - base
                if load[f_old] == d_i:          # f_old becomes empty -> save open cost
                    deltas = deltas - open_cost[f_old]
                for idx, f_new in enumerate(feasible):
                    if load[f_new] == 0:        # f_new newly opened -> pay open cost
                        deltas[idx] += open_cost[f_new]
                best_idx = int(np.argmin(deltas))
                if deltas[best_idx] < -1e-9:
                    f_new = feasible[best_idx]
                    res[i] = f_new
                    load[f_old] -= d_i
                    load[f_new] += d_i
                    improved = True

            if not improved:
                # --- 2. Sampled swap moves ---
                for _ in range(1500):
                    if time.time() > deadline - 3.0:
                        break
                    i, j = int(rng.integers(0, n_cust)), int(rng.integers(0, n_cust))
                    if i == j:
                        continue
                    f_i, f_j = res[i], res[j]
                    if f_i == f_j:
                        continue
                    d_i, d_j = demands[i], demands[j]
                    new_load_i = load[f_i] - d_i + d_j
                    new_load_j = load[f_j] - d_j + d_i
                    if new_load_i > capacities[f_i] or new_load_j > capacities[f_j]:
                        continue
                    delta_assign = (assign_cost[i, f_j] - assign_cost[i, f_i]) + \
                                   (assign_cost[j, f_i] - assign_cost[j, f_j])
                    old_open_i = open_cost[f_i] if load[f_i] > 0 else 0.0
                    old_open_j = open_cost[f_j] if load[f_j] > 0 else 0.0
                    new_open_i = open_cost[f_i] if new_load_i > 0 else 0.0
                    new_open_j = open_cost[f_j] if new_load_j > 0 else 0.0
                    delta_open = (new_open_i + new_open_j) - (old_open_i + old_open_j)
                    if delta_assign + delta_open < -1e-9:
                        res[i], res[j] = res[j], res[i]
                        load[f_i] = new_load_i
                        load[f_j] = new_load_j
                        improved = True

                # --- 3. Facility-closing moves ---
                for f_close in range(n_fac):
                    if time.time() > deadline - 3.0:
                        break
                    if load[f_close] == 0:
                        continue
                    custs = np.where(res == f_close)[0]
                    if len(custs) == 0:
                        continue
                    temp_load = load.copy()
                    temp_res = res.copy()
                    temp_load[f_close] = 0
                    success = True
                    for c in np.argsort(-demands[custs]):
                        cc = custs[c]
                        placed = False
                        for f in np.argsort(assign_cost[cc]):
                            if f == f_close:
                                continue
                            if temp_load[f] + demands[cc] <= capacities[f]:
                                temp_load[f] += demands[cc]
                                temp_res[cc] = f
                                placed = True
                                break
                        if not placed:
                            success = False
                            break
                    if success:
                        new_cost = calc_cost(temp_res)
                        old_cost = calc_cost(res)
                        if new_cost < old_cost - 1e-9:
                            res = temp_res
                            load = temp_load
                            improved = True

        return res, load

    # ---------- candidate facility subsets (open/fixed cost trade-off) ----------
    avg_assign = assign_cost.mean(axis=0)
    by_open = np.argsort(open_cost)
    by_assign = np.argsort(avg_assign)
    by_combined = np.argsort(open_cost + avg_assign)

    candidate_subsets = []
    candidate_subsets.append(list(range(n_fac)))
    for k in (5, 8, 10, 12, 15):
        if k <= n_fac:
            candidate_subsets.append(by_open[:k].tolist())
            candidate_subsets.append(by_assign[:k].tolist())
            candidate_subsets.append(by_combined[:k].tolist())

    # de-duplicate
    seen = set()
    unique_subsets = []
    for s in candidate_subsets:
        key = frozenset(s)
        if key not in seen:
            seen.add(key)
            unique_subsets.append(s)

    # ---------- multi-start greedy initialisations (diverse orderings) ----------
    orderings = [
        np.argsort(-demands),                                   # largest demand first
        np.argsort(assign_cost.mean(axis=1)),                   # cheapest-to-serve first
        np.arange(n_cust),                                      # natural
        np.random.default_rng(7).permutation(n_cust),
        np.random.default_rng(13).permutation(n_cust),
    ]

    best_result = None
    best_cost = float('inf')

    # Phase A: diverse greedy starts, each refined with a short local search
    for subset in unique_subsets:
        if time.time() > deadline - 20.0:
            break
        for order in orderings:
            if time.time() > deadline - 8.0:
                break
            res = _greedy(order, subset)
            if res is None:
                continue
            load = loads_of(res)
            res, load = local_search(res, load, max_time=2.0)
            c = calc_cost(res)
            if c < best_cost:
                best_cost = c
                best_result = res.copy()

    # Phase B: spend remaining budget on deeper local search from the best solution
    if best_result is not None and time.time() < deadline - 5.0:
        load = loads_of(best_result)
        res, load = local_search(best_result, load, max_time=deadline - 5.0 - time.time())
        c = calc_cost(res)
        if c < best_cost:
            best_cost = c
            best_result = res.copy()

    # ---------- guaranteed-valid fallback ----------
    if best_result is None:
        best_result = np.zeros(n_cust, dtype=np.int64)
        load = np.zeros(n_fac, dtype=np.int64)
        for i in np.argsort(-demands):
            for f in np.argsort(assign_cost[i]):
                if load[f] + demands[i] <= capacities[f]:
                    load[f] += demands[i]
                    best_result[i] = f
                    break
        # ensure every customer got a valid facility
        for i in range(n_cust):
            if best_result[i] < 0 or best_result[i] >= n_fac:
                for f in range(n_fac):
                    if load[f] + demands[i] <= capacities[f]:
                        load[f] += demands[i]
                        best_result[i] = f
                        break

    return best_result
# EVOLVE-BLOCK-END

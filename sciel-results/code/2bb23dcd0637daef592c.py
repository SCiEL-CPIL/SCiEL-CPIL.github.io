# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_cost(result, demands, open_cost, assign_cost):
    total_assign = np.sum(assign_cost[np.arange(len(demands)), result])
    open_mask = np.zeros(len(open_cost), dtype=bool)
    for f in result:
        open_mask[f] = True
    total_open = np.sum(open_cost[open_mask])
    return total_assign + total_open

def _loads(result, demands):
    load = np.zeros(len(demands) and len(result) and 0 or 0)
    load = np.zeros(len(result) and int(max(result)) + 1 if len(result) else 0)
    # Simpler:
    m = int(max(result)) + 1 if len(result) else 0
    load = np.zeros(m)
    for i in range(len(result)):
        load[result[i]] += demands[i]
    return load

def _repair(result, demands, capacities, assign_cost):
    """Fix capacity violations by moving customers from overloaded facilities to cheapest feasible ones."""
    n = len(demands)
    m = len(capacities)
    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]

    for _ in range(500):
        overloaded = np.where(load > capacities + 1e-6)[0]
        if len(overloaded) == 0:
            break
        for f_over in overloaded:
            excess = load[f_over] - capacities[f_over]
            custs_at_f = [i for i in range(n) if result[i] == f_over]
            custs_at_f.sort(key=lambda i: -demands[i])
            for i in custs_at_f:
                if excess <= 1e-6:
                    break
                d = demands[i]
                best_f = -1
                best_cost = np.inf
                # vectorized search over feasible facilities
                feasible = (load + d) <= (capacities + 1e-6)
                feasible[f_over] = False
                if np.any(feasible):
                    cands = np.where(feasible)[0]
                    cc = assign_cost[i, cands]
                    idx = int(np.argmin(cc))
                    best_f = int(cands[idx])
                    best_cost = cc[idx]
                if best_f >= 0:
                    load[f_over] -= d
                    load[best_f] += d
                    result[i] = best_f
                    excess -= d
                else:
                    rem = capacities - load
                    rem[f_over] = -np.inf
                    best_f = int(np.argmax(rem))
                    load[f_over] -= d
                    load[best_f] += d
                    result[i] = best_f
                    excess -= d
    return result

def _local_search(result, demands, capacities, open_cost, assign_cost,
                  max_iters=200, deadline=None, rng=None, tol=1e-9):
    """Vectorized single-move local search."""
    if rng is None:
        rng = np.random.default_rng()

    n = len(demands)
    m = len(capacities)

    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]

    # Precompute per-customer load contribution matrix L[i, f] = demands[i] if result[i]==f else 0
    L = np.zeros((n, m))
    L[np.arange(n), result] = demands

    current_cost = _compute_cost(result, demands, open_cost, assign_cost)

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        improved = False
        order = rng.permutation(n)

        for i in order:
            cur = result[i]
            d = demands[i]

            # feasibility: load[f] + d <= capacities[f]
            feasible = (load + d) <= (capacities + 1e-6)
            feasible[cur] = False
            if not np.any(feasible):
                continue

            cands = np.where(feasible)[0]
            delta = assign_cost[i, cands] - assign_cost[i, cur]

            # opening cost adjustments
            if load[cur] == d:
                delta = delta - open_cost[cur]
            open_new = (load[cands] == 0)
            delta = delta + open_cost[cands] * open_new

            idx = int(np.argmin(delta))
            if delta[idx] < -tol:
                best_f = int(cands[idx])
                # update
                L[i, cur] = 0.0
                L[i, best_f] = d
                load[cur] -= d
                load[best_f] += d
                result[i] = best_f
                current_cost += delta[idx]
                improved = True

        if not improved:
            break

    return result, current_cost

def _swap_search(result, demands, capacities, open_cost, assign_cost,
                 max_iters=50, deadline=None, rng=None, tol=1e-9):
    """2-opt swaps: move a customer from facility A to B and one from B to A."""
    if rng is None:
        rng = np.random.default_rng()

    n = len(demands)
    m = len(capacities)

    load = np.zeros(m)
    for i in range(n):
        load[result[i]] += demands[i]

    current_cost = _compute_cost(result, demands, open_cost, assign_cost)

    # group customers by facility
    def build_groups():
        groups = {}
        for i in range(n):
            f = result[i]
            groups.setdefault(f, []).append(i)
        return groups

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        groups = build_groups()
        facs = [f for f in groups if len(groups[f]) > 0]
        if len(facs) < 2:
            break

        improved = False
        # sample pairs of facilities
        for _ in range(min(2000, len(facs) * len(facs))):
            if deadline is not None and time.time() > deadline:
                break
            fA = int(rng.integers(0, len(facs)))
            fB = int(rng.integers(0, len(facs)))
            if fA == fB:
                continue
            fA = facs[fA]
            fB = facs[fB]
            gA = groups[fA]
            gB = groups[fB]
            if not gA or not gB:
                continue
            i = gA[int(rng.integers(0, len(gA)))]
            j = gB[int(rng.integers(0, len(gB)))]
            di = demands[i]
            dj = demands[j]

            # after swap: fA loses i gains j; fB loses j gains i
            newA = load[fA] - di + dj
            newB = load[fB] - dj + di
            if newA > capacities[fA] + 1e-6 or newB > capacities[fB] + 1e-6:
                continue

            # cost delta
            delta = (assign_cost[i, fB] + assign_cost[j, fA]) - (assign_cost[i, fA] + assign_cost[j, fB])

            # open cost: fA stays open (had at least i), fB stays open (had at least j)
            # but if a facility becomes empty it loses open cost; here both still have >=1 after swap
            # (fA had >=1 (i) and gains j, fB had >=1 (j) and gains i) so no open change unless
            #  the group had exactly that one customer. Handle:
            if len(gA) == 1:
                # fA becomes empty of i but gains j -> still has j, stays open
                pass
            if len(gB) == 1:
                pass

            if delta < -tol:
                # apply
                load[fA] = newA
                load[fB] = newB
                result[i] = fB
                result[j] = fA
                current_cost += delta
                improved = True

        if not improved:
            break

    return result, current_cost

def _greedy_init(demands, capacities, assign_cost):
    """Greedy initialization: largest demand first."""
    n = len(demands)
    m = len(capacities)
    order = np.argsort(-demands)
    result = np.full(n, -1, dtype=int)
    load = np.zeros(m)
    for i in order:
        d = demands[i]
        feasible = (load + d) <= (capacities + 1e-6)
        if np.any(feasible):
            cands = np.where(feasible)[0]
            cc = assign_cost[i, cands]
            best_f = int(cands[int(np.argmin(cc))])
        else:
            rem = capacities - load
            best_f = int(np.argmax(rem))
        result[i] = best_f
        load[best_f] += d
    return result

def _cheapest_init(demands, capacities, assign_cost):
    n = len(demands)
    result = np.argmin(assign_cost, axis=1).astype(int)
    return result

def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    n = len(demands)
    m = len(capacities)

    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    rng = np.random.default_rng(42)

    best_result = None
    best_cost = np.inf

    def consider(result, do_ls=True, do_swap=True, ls_iters=200, swap_iters=50):
        nonlocal best_result, best_cost
        if deadline is not None and time.time() > deadline:
            return
        result = _repair(result, demands, capacities, assign_cost)
        cost = _compute_cost(result, demands, open_cost, assign_cost)
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
        if do_ls and time.time() < deadline - 5:
            result, cost = _local_search(result, demands, capacities, open_cost, assign_cost,
                                          max_iters=ls_iters, deadline=deadline, rng=rng)
            if cost < best_cost:
                best_cost = cost
                best_result = result.copy()
        if do_swap and time.time() < deadline - 5:
            result, cost = _swap_search(result, demands, capacities, open_cost, assign_cost,
                                         max_iters=swap_iters, deadline=deadline, rng=rng)
            if cost < best_cost:
                best_cost = cost
                best_result = result.copy()

    # 1. Greedy init
    r1 = _greedy_init(demands, capacities, assign_cost)
    consider(r1, do_ls=True, do_swap=True, ls_iters=200, swap_iters=50)

    # 2. Cheapest init
    if time.time() < deadline - 20:
        r2 = _cheapest_init(demands, capacities, assign_cost)
        consider(r2, do_ls=True, do_swap=True, ls_iters=200, swap_iters=50)

    # 3. Randomized greedy (different orderings) for diversity
    for trial in range(3):
        if time.time() < deadline - 15:
            order = np.argsort(-demands * (0.5 + rng.random(n)))
            result = np.full(n, -1, dtype=int)
            load = np.zeros(m)
            for i in order:
                d = demands[i]
                feasible = (load + d) <= (capacities + 1e-6)
                if np.any(feasible):
                    cands = np.where(feasible)[0]
                    cc = assign_cost[i, cands]
                    # add a little randomness
                    cc = cc + rng.random(len(cands)) * 5.0
                    best_f = int(cands[int(np.argmin(cc))])
                else:
                    rem = capacities - load
                    best_f = int(np.argmax(rem))
                result[i] = best_f
                load[best_f] += d
            consider(result, do_ls=True, do_swap=True, ls_iters=150, swap_iters=40)

    # 4. Final high-precision polish on the best result
    if best_result is not None and time.time() < deadline - 3:
        # tighter tolerance, more sweeps
        result, cost = _local_search(best_result, demands, capacities, open_cost, assign_cost,
                                      max_iters=300, deadline=deadline, rng=rng, tol=1e-11)
        if cost < best_cost:
            best_cost = cost
            best_result = result.copy()
        if time.time() < deadline - 3:
            result, cost = _swap_search(best_result, demands, capacities, open_cost, assign_cost,
                                         max_iters=80, deadline=deadline, rng=rng, tol=1e-11)
            if cost < best_cost:
                best_cost = cost
                best_result = result.copy()
        # one more single-move pass
        if time.time() < deadline - 1:
            result, cost = _local_search(best_result, demands, capacities, open_cost, assign_cost,
                                          max_iters=200, deadline=deadline, rng=rng, tol=1e-12)
            if cost < best_cost:
                best_cost = cost
                best_result = result.copy()

    # Final validity check
    if best_result is not None:
        load = np.zeros(m)
        valid = True
        for i in range(n):
            f = best_result[i]
            if f < 0 or f >= m:
                valid = False
                break
            load[f] += demands[i]
        if not valid or np.any(load > capacities + 1e-6):
            best_result = _repair(best_result, demands, capacities, assign_cost)

    if best_result is None:
        best_result = np.zeros(n, dtype=int)
        best_result = _repair(best_result, demands, capacities, assign_cost)

    return best_result
# EVOLVE-BLOCK-END

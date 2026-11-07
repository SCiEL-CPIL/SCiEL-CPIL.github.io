"""Rewritten solver for capacitated facility location (100 customers, 20 facilities)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _cost_of(assign, open_cost, assign_cost):
    """Total cost = open_cost over used facilities + sum assign_cost[i, a[i]]."""
    used = np.unique(assign)
    open_sum = float(open_cost[used].sum()) if used.size else 0.0
    assign_sum = float(assign_cost[np.arange(len(assign)), assign].sum())
    return open_sum + assign_sum


def assign(demands, capacities, open_cost, assign_cost):
    demands = np.asarray(demands, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=np.int64)
    open_cost = np.asarray(open_cost, dtype=np.float64)
    assign_cost = np.asarray(assign_cost, dtype=np.float64)

    n_cust = len(demands)
    n_fac = len(capacities)

    # ---- Construction: cheapest-first greedy with capacity fallback ----
    a = np.empty(n_cust, dtype=np.int64)
    load = np.zeros(n_fac, dtype=np.int64)
    order = np.argsort(assign_cost.min(axis=1))  # easy customers first
    for i in order:
        ci = int(i)
        best_f = -1
        best_val = None
        for f in np.argsort(assign_cost[ci]):
            if load[f] + demands[ci] <= capacities[f]:
                best_f = int(f)
                break
        if best_f < 0:  # fallback: least-loaded feasible
            for f in np.argsort(load):
                if load[f] + demands[ci] <= capacities[f]:
                    best_f = int(f)
                    break
        if best_f < 0:  # truly infeasible (should not happen); pick min load
            best_f = int(np.argmin(load))
        a[ci] = best_f
        load[best_f] += demands[ci]

    best_a = a.copy()
    best_cost = _cost_of(best_a, open_cost, assign_cost)

    # Precompute sorted facility order per customer
    sorted_f = np.argsort(assign_cost, axis=1)  # shape (n_cust, n_fac)

    # ---- Local search helpers ----
    def try_move(i, f):
        """Move customer i to facility f if feasible. Returns True if applied."""
        old = a[i]
        if old == f:
            return False
        if load[f] + demands[i] <= capacities[f]:
            load[old] -= demands[i]
            load[f] += demands[i]
            a[i] = f
            return True
        return False

    def one_move_pass():
        """Greedy 1-move local search until no improvement."""
        improved = True
        while improved:
            improved = False
            for i in range(n_cust):
                old = a[i]
                for f in sorted_f[i]:
                    if f == old:
                        continue
                    if load[f] + demands[i] <= capacities[f]:
                        # delta cost
                        new_c = _cost_of_with_move(a, i, f, open_cost, assign_cost, load)
                        if new_c < best_cost - 1e-9:
                            # commit
                            load[old] -= demands[i]
                            load[f] += demands[i]
                            a[i] = f
                            best_cost = new_c
                            improved = True
                            break
        return improved

    def _cost_of_with_move(a, i, f, open_cost, assign_cost, load):
        # cost delta for moving i to f
        old = a[i]
        delta = assign_cost[i, f] - assign_cost[i, old]
        # open cost changes
        if load[old] - demands[i] == 0 and load[f] + demands[i] > 0:
            delta += open_cost[f] - open_cost[old]
        elif load[old] - demands[i] > 0 and load[f] + demands[i] == 0:
            delta += open_cost[old] - open_cost[f]
        return best_cost + delta

    deadline = time.time() + 130.0  # safety margin under 150s

    # ---- Simulated annealing with 1-move and 2-swap ----
    T = 200.0
    cooling = 0.995
    while time.time() < deadline:
        # one pass of 1-move SA
        for i in range(n_cust):
            old = a[i]
            f = int(sorted_f[i, np.random.randint(n_fac)])
            if f == old:
                continue
            if load[f] + demands[i] <= capacities[f]:
                delta = assign_cost[i, f] - assign_cost[i, old]
                if load[old] - demands[i] == 0 and load[f] + demands[i] > 0:
                    delta += open_cost[f] - open_cost[old]
                elif load[old] - demands[i] > 0 and load[f] + demands[i] == 0:
                    delta += open_cost[old] - open_cost[f]
                if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-9)):
                    load[old] -= demands[i]
                    load[f] += demands[i]
                    a[i] = f
                    c = _cost_of(a, open_cost, assign_cost)
                    if c < best_cost:
                        best_cost = c
                        best_a = a.copy()
        # one pass of 2-swap SA
        i = np.random.randint(n_cust)
        j = np.random.randint(n_cust)
        if i == j:
            continue
        fi = a[i]
        fj = a[j]
        if fi == fj:
            continue
        # swap: i->fj, j->fi
        if load[fj] + demands[i] <= capacities[fj] and load[fi] + demands[j] <= capacities[fi]:
            # compute delta
            delta = (assign_cost[i, fj] + assign_cost[j, fi]) - (assign_cost[i, fi] + assign_cost[j, fj])
            # open cost: facilities fi and fj usage
            new_load_fi = load[fi] - demands[j] + demands[i]
            new_load_fj = load[fj] - demands[i] + demands[j]
            if new_load_fi == 0:
                delta -= open_cost[fi]
            if new_load_fj == 0:
                delta -= open_cost[fj]
            if load[fi] == 0:
                delta += open_cost[fi]
            if load[fj] == 0:
                delta += open_cost[fj]
            if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-9)):
                # apply swap
                load[fi] = new_load_fi
                load[fj] = new_load_fj
                a[i] = fj
                a[j] = fi
                c = _cost_of(a, open_cost, assign_cost)
                if c < best_cost:
                    best_cost = c
                    best_a = a.copy()
        T *= cooling

    # ---- Final polish: greedy 1-move until convergence ----
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_cust):
            old = a[i]
            for f in sorted_f[i]:
                if f == old:
                    continue
                if load[f] + demands[i] <= capacities[f]:
                    c = _cost_of_with_move(a, i, f, open_cost, assign_cost, load)
                    if c < best_cost - 1e-9:
                        load[old] -= demands[i]
                        load[f] += demands[i]
                        a[i] = f
                        best_cost = c
                        improved = True
                        break

    # Ensure we return the best valid assignment
    if _cost_of(a, open_cost, assign_cost) <= best_cost + 1e-9:
        return a
    return best_a
# EVOLVE-BLOCK-END

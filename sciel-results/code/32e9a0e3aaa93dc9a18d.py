"""Rewritten: min-cost flow with facility opening + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_cost_flow


def _valid(result, demands, capacities):
    n = len(capacities)
    if result.shape[0] != len(demands):
        return False
    if np.any(result < 0) or np.any(result >= n):
        return False
    load = np.bincount(result, minlength=n)
    return bool(np.all(load <= capacities + 1e-9))


def _objective(result, demands, capacities, open_cost, assign_cost):
    n = len(capacities)
    load = np.bincount(result, minlength=n)
    used = load > 0
    return float(np.sum(open_cost[used]) + np.sum(assign_cost[np.arange(len(result)), result]))


def _mincost_flow(demands, capacities, open_cost, assign_cost):
    """Solve capacitated facility location via min-cost flow.
    Nodes: 0=source, 1..F=facilities, F+1..F+C=customers, F+C+1=sink.
    Edge source->f capacity=capacities[f] cost=open_cost[f].
    Edge f->c capacity=demands[c] cost=assign_cost[c,f].
    Edge c->sink capacity=demands[c] cost=0.
    """
    F = len(capacities)
    C = len(demands)
    N = F + C + 2
    s = 0
    t = N - 1
    rows, cols, caps, costs = [], [], [], []
    # source -> facility
    for f in range(F):
        rows.append(s); cols.append(1 + f)
        caps.append(int(capacities[f])); costs.append(float(open_cost[f]))
    # facility -> customer
    for c in range(C):
        for f in range(F):
            rows.append(1 + f); cols.append(1 + F + c)
            caps.append(int(demands[c])); costs.append(float(assign_cost[c, f]))
    # customer -> sink
    for c in range(C):
        rows.append(1 + F + c); cols.append(t)
        caps.append(int(demands[c])); costs.append(0.0)
    A = coo_matrix((costs, (rows, cols)), shape=(N, N)).tocsr()
    try:
        flow_value, flow_cost, flow, n_node, n_arc = min_cost_flow(A, demands=[-int(demands.sum()), 0, 0], capacities=caps)
    except Exception:
        return None
    if flow_value < int(demands.sum()) - 1e-6:
        return None
    # result[c] = facility f with flow on edge (1+f, 1+F+c)
    result = np.full(C, -1, dtype=int)
    for f in range(F):
        for c in range(C):
            v = float(flow[1 + f, 1 + F + c])
            if v > 0.5:
                result[c] = f
    if np.any(result < 0):
        return None
    return result


def _local_search(result, demands, capacities, open_cost, assign_cost, deadline):
    """Hill-climb: try moving a customer to another facility if it improves objective."""
    F = len(capacities)
    C = len(demands)
    result = result.copy()
    load = np.bincount(result, minlength=F)
    obj = _objective(result, demands, capacities, open_cost, assign_cost)
    best = obj
    improved = True
    order = np.argsort(-demands)  # big customers first for fewer feasible moves
    while improved:
        improved = False
        for c in order:
            if time.time() > deadline:
                break
            cur = result[c]
            d = demands[c]
            # candidate facilities sorted by assign cost
            for f in np.argsort(assign_cost[c]):
                if f == cur:
                    continue
                # move c from cur to f
                new_load_cur = load[cur] - d
                new_load_f = load[f] + d
                if new_load_f > capacities[f] + 1e-9:
                    continue
                # delta objective
                old_open_cur = open_cost[cur] if new_load_cur > 0 else 0.0
                old_open_f = open_cost[f] if new_load_f > 0 else 0.0
                new_open_cur = open_cost[cur] if (new_load_cur > 0) else 0.0
                new_open_f = open_cost[f] if (new_load_f > 0) else 0.0
                delta_assign = assign_cost[c, f] - assign_cost[c, cur]
                delta_open = (new_open_cur - old_open_cur) + (new_open_f - old_open_f)
                if delta_assign + delta_open < -1e-9:
                    load[cur] = new_load_cur
                    load[f] = new_load_f
                    result[c] = f
                    obj += delta_assign + delta_open
                    best = min(best, obj)
                    improved = True
                    break
        if time.time() > deadline:
            break
    return result, best


def assign(demands, capacities, open_cost, assign_cost):
    start = time.time()
    deadline = start + 130.0  # safety margin under 150s
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    open_cost = np.asarray(open_cost, dtype=float)
    assign_cost = np.asarray(assign_cost, dtype=float)
    F = len(capacities)
    C = len(demands)

    # Greedy fallback (fast, always valid)
    greedy = np.zeros(C, dtype=int)
    load = np.zeros(F)
    for i in np.argsort(-demands):
        placed = False
        for f in np.argsort(assign_cost[i]):
            if load[f] + demands[i] <= capacities[f] + 1e-9:
                load[f] += demands[i]
                greedy[i] = f
                placed = True
                break
        if not placed:
            # assign to facility with most remaining capacity
            f = int(np.argmax(capacities - load))
            load[f] += demands[i]
            greedy[i] = f
    best_result = greedy.copy()
    best_obj = _objective(best_result, demands, capacities, open_cost, assign_cost)

    # Min-cost flow (strong exact solver for the LP/integer structure)
    if time.time() < deadline - 5.0:
        mf = _mincost_flow(demands, capacities, open_cost, assign_cost)
        if mf is not None and _valid(mf, demands, capacities):
            obj = _objective(mf, demands, capacities, open_cost, assign_cost)
            if obj < best_obj:
                best_result = mf.copy()
                best_obj = obj

    # Local search refinement
    if time.time() < deadline - 1.0:
        ls, obj = _local_search(best_result, demands, capacities, open_cost, assign_cost, deadline)
        if _valid(ls, demands, capacities):
            if obj < best_obj:
                best_result = ls
                best_obj = obj

    # Final validity guard
    if not _valid(best_result, demands, capacities):
        return greedy
    return best_result.astype(int)
# EVOLVE-BLOCK-END

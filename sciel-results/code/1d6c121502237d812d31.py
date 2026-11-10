import numpy as np
import time

_STATE = {"start": None, "deadline": None}

def _budget():
    if _STATE["start"] is None:
        _STATE["start"] = time.time()
        _STATE["deadline"] = _STATE["start"] + 145.0
    return max(0.0, _STATE["deadline"] - time.time())


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    cap = float(remaining_capacity)
    if cap <= 0:
        return -1
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    n = len(w)
    if n == 0:
        return -1
    # remove items that individually exceed capacity? Keep them; B&B skips.
    # But drop zero/negative weight oddities.
    # density for sorting; guard zero weight
    with np.errstate(divide='ignore', invalid='ignore'):
        dens = np.where(w > 0, v / np.where(w > 0, w, 1.0), np.where(v > 0, np.inf, 0.0))
    order = np.argsort(-dens, kind='stable')
    ws = w[order]
    vs = v[order]
    # greedy incumbent
    gw, gv, gset = 0.0, 0.0, []
    for i in range(n):
        if gw + ws[i] <= cap:
            gw += ws[i]; gv += vs[i]; gset.append(int(order[i]))
    if not gset:
        return -1
    best_val = gv
    best_set = list(gset)
    budget = _budget()
    if budget <= 0.05:
        return best_set[0]
    node_cap = int(min(400000, max(4000, budget * 40000)))
    nodes = [0]
    stop = [False]
    last_check = [0]
    def time_check():
        if nodes[0] - last_check[0] >= 2000:
            last_check[0] = nodes[0]
            if _budget() < 0.1:
                stop[0] = True
        return stop[0] or nodes[0] > node_cap
    # bound
    def bound(i, cw, cv):
        rem = cap - cw
        b = cv
        j = i
        while j < n and rem > 0:
            if ws[j] <= rem:
                rem -= ws[j]; b += vs[j]
            else:
                b += vs[j] * (rem / ws[j])
                rem = 0
            j += 1
        return b
    sys_setrecursionlimit = __import__('sys').setrecursionlimit
    sys_setrecursionlimit(max(10000, n + 100))
    def dfs(i, cw, cv, chosen):
        if time_check():
            return
        nodes[0] += 1
        if i >= n:
            if cv > best_val:
                best_val = cv
                best_set[:] = list(chosen)
            return
        if bound(i, cw, cv) <= best_val:
            return
        if cw + ws[i] <= cap:
            chosen.append(int(order[i]))
            dfs(i + 1, cw + ws[i], cv + vs[i], chosen)
            chosen.pop()
        dfs(i + 1, cw, cv, chosen)
    dfs(0, 0.0, 0.0, [])
    if best_set:
        return int(best_set[0])
    return -1

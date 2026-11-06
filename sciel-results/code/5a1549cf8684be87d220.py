import numpy as np
import time
from collections import defaultdict

def priority(instance):
    start_time = time.time()
    deadline = start_time + 5.0

    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)

    if n == 0:
        return np.array([], dtype=float)

    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        if predecessors[i]:
            for pred in predecessors[i]:
                successors[pred].append(i)

    # ---- TDW: total downstream work (self + all transitive successors) ----
    tdw = [0.0] * n
    state = [0] * n

    def get_tdw(i):
        if state[i] == 2:
            return tdw[i]
        if state[i] == 1:
            return 0.0
        state[i] = 1
        s = 0.0
        for j in successors[i]:
            if time.time() > deadline:
                state[i] = 2
                tdw[i] = times[i]
                return tdw[i]
            s += get_tdw(j)
        tdw[i] = times[i] + s
        state[i] = 2
        return tdw[i]

    for i in range(n):
        get_tdw(i)

    # ---- LSP: longest successor path (self + max over successors) ----
    lsp = [0.0] * n
    state_lsp = [0] * n

    def get_lsp(i):
        if state_lsp[i] == 2:
            return lsp[i]
        if state_lsp[i] == 1:
            return 0.0
        state_lsp[i] = 1
        mx = 0.0
        for j in successors[i]:
            if time.time() > deadline:
                lsp[i] = times[i]
                state_lsp[i] = 2
                return lsp[i]
            v = get_lsp(j)
            if v > mx:
                mx = v
        lsp[i] = times[i] + mx
        state_lsp[i] = 2
        return lsp[i]

    for i in range(n):
        get_lsp(i)

    # ---- Total (transitive) successor count ----
    tsc = [0] * n
    state_tsc = [0] * n

    def get_tsc(i):
        if state_tsc[i] == 2:
            return tsc[i]
        if state_tsc[i] == 1:
            return 0
        state_tsc[i] = 1
        s = 0
        for j in successors[i]:
            if time.time() > deadline:
                tsc[i] = len(successors[i])
                state_tsc[i] = 2
                return tsc[i]
            s += get_tsc(j)
        tsc[i] = s
        state_tsc[i] = 2
        return tsc[i]

    for i in range(n):
        get_tsc(i)

    # ---- Soft-max over successor LSPs (smooth surrogate for branching constraint) ----
    softmax = [0.0] * n
    state_sm = [0] * n
    beta = 0.5

    def get_softmax(i):
        if state_sm[i] == 2:
            return softmax[i]
        if state_sm[i] == 1:
            return 0.0
        state_sm[i] = 1
        succ_vals = []
        for j in successors[i]:
            if time.time() > deadline:
                softmax[i] = 0.0
                state_sm[i] = 2
                return softmax[i]
            succ_vals.append(get_lsp(j))
        if succ_vals:
            arr = np.asarray(succ_vals, dtype=float)
            m = float(arr.max())
            if m > 0:
                softmax[i] = m + np.log(np.sum(np.exp(beta * (arr - m)))) / beta
            else:
                softmax[i] = 0.0
        else:
            softmax[i] = 0.0
        state_sm[i] = 2
        return softmax[i]

    for i in range(n):
        get_softmax(i)

    # ---- Assemble normalized components ----
    tdw_arr = np.asarray(tdw, dtype=float)
    lsp_arr = np.asarray(lsp, dtype=float)
    tsc_arr = np.asarray(tsc, dtype=float)
    sm_arr = np.asarray(softmax, dtype=float)
    ct = float(instance.get("cycle_time", 40.0))
    if ct <= 0:
        ct = 40.0
    bottleneck = times / ct

    def norm(a):
        a = np.asarray(a, dtype=float)
        rng = float(a.max() - a.min())
        if rng > 1e-12:
            return (a - a.min()) / rng
        return np.zeros_like(a)

    tdw_n = norm(tdw_arr)
    lsp_n = norm(lsp_arr)
    tsc_n = norm(tsc_arr)
    sm_n = norm(sm_arr)
    bn_n = norm(bottleneck)

    # Blend: downstream work dominates, then critical path, then branching,
    # then smooth branching constraint, then cycle-time bottleneck.
    priorities = (1.0 * tdw_n
                  + 0.4 * lsp_n
                  + 0.3 * tsc_n
                  + 0.15 * sm_n
                  + 0.1 * bn_n
                  + times * 1e-6)

    return priorities

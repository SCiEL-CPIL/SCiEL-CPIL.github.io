"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0


def _time_ok(margin=2.0):
    return (time.time() - _START_TIME) < (_DEADLINE - margin)


def _greedy_fill(ow, ov, cap, skip_mask=None):
    total = 0.0
    n = len(ow)
    if skip_mask is None:
        for k in range(n):
            w = ow[k]
            if w <= cap:
                cap -= w
                total += ov[k]
    else:
        for k in range(n):
            if skip_mask[k]:
                continue
            w = ow[k]
            if w <= cap:
                cap -= w
                total += ov[k]
    return total


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    C = float(remaining_capacity)
    if C <= 0:
        return -1
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    n = len(w)
    if n == 0:
        return -1

    feasible = w <= C
    if not np.any(feasible):
        return -1

    feas_idx = np.where(feasible)[0]
    fw = w[feas_idx]
    fv = v[feas_idx]
    m = len(fw)
    if m == 0:
        return -1

    with np.errstate(divide='ignore', invalid='ignore'):
        dens = np.where(fw > 0, fv / np.where(fw > 0, fw, 1.0), np.inf)
    order = np.argsort(-dens, kind='stable')
    ow = fw[order]
    ov = fv[order]
    pos = np.empty(m, dtype=int)
    pos[order] = np.arange(m)

    if m <= 2:
        best = -1
        best_val = -np.inf
        for i in range(m):
            if fw[i] <= C:
                if fv[i] > best_val:
                    best_val = fv[i]
                    best = i
        if best == -1:
            return -1
        return int(feas_idx[best])

    # 1-step lookahead
    scores1 = np.empty(m)
    for i in range(m):
        cap1 = C - fw[i]
        if cap1 < 0:
            scores1[i] = -np.inf
        else:
            skip_mask = np.zeros(m, dtype=bool)
            skip_mask[pos[i]] = True
            scores1[i] = fv[i] + _greedy_fill(ow, ov, cap1, skip_mask)

    if not _time_ok(margin=5.0):
        best = int(np.argmax(scores1))
        return int(feas_idx[best])

    # 2-step lookahead for top candidates
    if m <= 40:
        cand = list(range(m))
    else:
        B = min(30, m)
        top = np.argsort(-scores1, kind='stable')[:B]
        cand = [int(t) for t in top]

    best_score = -np.inf
    best_i = cand[0]

    for i in cand:
        cap1 = C - fw[i]
        if cap1 < 0:
            continue
        pi = pos[i]
        base = fv[i]

        skip_mask_i = np.zeros(m, dtype=bool)
        skip_mask_i[pi] = True

        best2 = base + _greedy_fill(ow, ov, cap1, skip_mask_i)

        for j in range(m):
            if j == i:
                continue
            wj = fw[j]
            if wj > cap1:
                continue
            cap2 = cap1 - wj
            if cap2 < 0:
                continue
            skip_mask_ij = np.zeros(m, dtype=bool)
            skip_mask_ij[pi] = True
            skip_mask_ij[pos[j]] = True
            val = base + fv[j] + _greedy_fill(ow, ov, cap2, skip_mask_ij)
            if val > best2:
                best2 = val

        if best2 > best_score:
            best_score = best2
            best_i = i

        if not _time_ok(margin=2.0):
            break

    return int(feas_idx[best_i])
# EVOLVE-BLOCK-END

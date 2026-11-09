"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 110.0  # safety margin before 120 s evaluator limit


def _time_ok(margin=3.0):
    """True if we still have time before the deadline (with margin)."""
    return (time.time() - _START_TIME) < (_DEADLINE - margin)


def _greedy_value(ow, ov, skip, cap):
    """Total value of density-order greedy fill of (ow, ov), skipping positions in `skip`,
    with starting capacity `cap`. Items that don't fit are skipped (scan continues)."""
    total = 0.0
    n = len(ow)
    if skip:
        # local vars for speed
        s0 = skip[0] if len(skip) > 0 else -1
        s1 = skip[1] if len(skip) > 1 else -1
        s2 = skip[2] if len(skip) > 2 else -1
        for k in range(n):
            if k == s0 or k == s1 or k == s2:
                continue
            w = ow[k]
            if w <= cap:
                cap -= w
                total += ov[k]
    else:
        for k in range(n):
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

    # Feasible items: weight <= C (and weight > 0 to avoid div issues; w<=0 -> free, keep)
    feasible = w <= C
    if not np.any(feasible):
        return -1

    # Work in original-index space but only consider feasible items.
    # Map feasible items to a compact list.
    feas_idx = np.where(feasible)[0]
    fw = w[feas_idx]
    fv = v[feas_idx]
    m = len(fw)
    if m == 0:
        return -1

    # Density (guard zero/negative weight: treat as +inf density so picked first)
    with np.errstate(divide='ignore', invalid='ignore'):
        dens = np.where(fw > 0, fv / np.where(fw > 0, fw, 1.0), np.inf)
    order = np.argsort(-dens, kind='stable')
    ow = fw[order]
    ov = fv[order]
    # pos_in_order: position in sorted order for each compact index
    pos = np.empty(m, dtype=int)
    pos[order] = np.arange(m)

    # 1-step lookahead score for each compact candidate
    # score[i] = fv[i] + greedy(ow, ov, {pos[i]}, C - fw[i])
    scores1 = np.empty(m)
    for i in range(m):
        cap1 = C - fw[i]
        if cap1 < 0:
            scores1[i] = -np.inf
        else:
            scores1[i] = fv[i] + _greedy_value(ow, ov, (pos[i],), cap1)

    # Choose candidate set for 2-step
    # If time is tight, fall back to 1-step.
    if not _time_ok(margin=5.0):
        # fall back to best 1-step (or pure density if even that is risky)
        best = int(np.argmax(scores1))
        return int(feas_idx[best])

    if m <= 50:
        cand = list(range(m))
    else:
        # top B by 1-step score
        B = 25
        top = np.argsort(-scores1, kind='stable')[:B]
        cand = [int(t) for t in top]

    # 2-step lookahead for each candidate in cand
    best_score = -np.inf
    best_i = cand[0]
    for i in cand:
        cap1 = C - fw[i]
        if cap1 < 0:
            continue
        pi = pos[i]
        base = fv[i]
        best2 = base + _greedy_value(ow, ov, (pi,), cap1)  # no second pick
        for j in range(m):
            if j == i:
                continue
            wj = fw[j]
            if wj > cap1:
                continue
            cap2 = cap1 - wj
            val = base + fv[j] + _greedy_value(ow, ov, (pi, pos[j]), cap2)
            if val > best2:
                best2 = val
        if best2 > best_score:
            best_score = best2
            best_i = i
        if not _time_ok(margin=2.0):
            break

    return int(feas_idx[best_i])
# EVOLVE-BLOCK-END

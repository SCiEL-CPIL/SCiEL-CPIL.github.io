"""Combined structure-driven heuristic: Lagrangian + greedy contribution +
soft-slack bottleneck + local swap polish + regret (leave-one-out) blocking."""
import time
import numpy as np

T_START = time.perf_counter()
LIMIT = 300.0
MARGIN = 8.0


def _deadline_left():
    return (T_START + LIMIT) - time.perf_counter()


def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if prizes.ndim == 2:
        p = prizes.sum(axis=0)
    else:
        p = prizes.ravel()
    if weights.ndim == 1:
        w = weights.reshape(1, -1)
    elif weights.ndim == 2 and weights.shape[0] != weights.shape[1]:
        w = weights
    elif weights.ndim == 2 and weights.shape[1] == p.shape[0]:
        w = weights
    else:
        w = weights.reshape(1, -1)

    n = int(p.shape[0])
    m = int(w.shape[0])
    cap = 25.0
    C = np.full(m, cap, dtype=float)

    def _safe(a):
        a = np.where(np.isfinite(a), a, 1.0)
        a = np.maximum(a, 1e-9)
        return a

    if n == 0:
        return np.ones(0)

    # ------------------------------------------------------------------
    # 1) Lagrangian subgradient for shadow prices -> reduced cost
    # ------------------------------------------------------------------
    lam = np.zeros(m, dtype=float)
    best_rc = p.copy()
    best_rc_val = -1.0

    n_sub = 25
    for it in range(n_sub):
        if _deadline_left() < MARGIN + 2.0:
            break
        shadow = lam @ w
        rc = p - shadow
        order = np.argsort(-rc, kind="stable")
        rem = C.copy()
        sel = []
        val = 0.0
        for i in order:
            if rc[i] <= 0.0:
                break
            if np.all(w[:, i] <= rem):
                sel.append(i)
                rem -= w[:, i]
                val += p[i]
        if val > best_rc_val:
            best_rc_val = val
            best_rc = rc.copy()
        used = np.zeros(m)
        for i in sel:
            used += w[:, i]
        slack = cap - used
        step = (4.0 / (1.0 + it * 0.15))
        lam = np.maximum(0.0, lam + step * slack)

    rmin = best_rc.min()
    rmax = best_rc.max()
    if rmax > rmin:
        rc_norm = (best_rc - rmin) / (rmax - rmin)
    else:
        rc_norm = np.ones(n)
    rc_norm = np.clip(rc_norm, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 2) Structural weight metrics + soft-slack bottleneck (log-sum-exp)
    # ------------------------------------------------------------------
    sum_w = w.sum(axis=0)
    max_w = w.max(axis=0)
    min_w = w.min(axis=0)
    avg_w = w.mean(axis=0)
    safe_max = np.maximum(max_w, 1e-9)

    slack_ratios = (cap - w) / np.maximum(w, 1e-9)
    bottleneck_slack = np.maximum(slack_ratios.min(axis=0), 0.0)

    # Soft-max tightness: a smooth proxy for the most constraining dimension
    tightness = w / cap
    t_clip = np.clip(tightness, 0.0, 1.0)
    t_shift = t_clip - t_clip.max(axis=0, keepdims=True)
    log_sum_exp = t_shift.max(axis=0) + np.log(np.sum(np.exp(t_shift), axis=0))
    soft_max_tightness = t_clip.max(axis=0, keepdims=True)[0] + log_sum_exp - np.log(m)
    soft_slack = np.clip(1.0 - soft_max_tightness, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 3) Greedy simulations with diverse orderings
    # ------------------------------------------------------------------
    contrib = np.zeros(n, dtype=float)
    selection_freq = np.zeros(n, dtype=float)
    best_found = 0.0
    best_sel_set = set()

    def _greedy(order):
        nonlocal best_found, best_sel_set
        rem = C.copy()
        val = 0.0
        sel = []
        for i in order:
            if np.all(w[:, i] <= rem):
                sel.append(i)
                rem -= w[:, i]
                val += p[i]
        if val > 0:
            for i in sel:
                contrib[i] += val
                selection_freq[i] += 1.0
        if val > best_found:
            best_found = val
            best_sel_set = set(sel)
        return val

    orderings = []
    orderings.append(np.argsort(-rc_norm, kind="stable"))
    orderings.append(np.argsort(-(p / (sum_w + 1e-9)), kind="stable"))
    orderings.append(np.argsort(-(p / (safe_max)), kind="stable"))
    orderings.append(np.argsort(-(p / (safe_max ** 2 + 1e-9)), kind="stable"))
    orderings.append(np.argsort(-(p / (avg_w + 1e-9)), kind="stable"))
    orderings.append(np.argsort(-(p / (min_w + 1e-9)), kind="stable"))
    orderings.append(np.argsort(-(p * (bottleneck_slack + 1.0) / (safe_max + 1.0)), kind="stable"))
    orderings.append(np.argsort(-p, kind="stable"))
    orderings.append(np.argsort(-(p * (soft_slack + 0.1) / (safe_max + 1.0)), kind="stable"))
    orderings.append(np.argsort(-soft_slack, kind="stable"))

    for seed in (7, 13, 42, 99, 123):
        rng = np.random.RandomState(seed)
        orderings.append(np.array(rng.permutation(n)))

    max_rounds = 300
    for r in range(max_rounds):
        if _deadline_left() < MARGIN:
            break
        for order in orderings:
            if _deadline_left() < MARGIN:
                break
            _greedy(order)

    # ------------------------------------------------------------------
    # 3b) Local swap polish on the best pack (captures discrete blocking)
    # ------------------------------------------------------------------
    if best_found > 0 and _deadline_left() > MARGIN + 5.0:
        best_items = list(best_sel_set)
        if len(best_items) > 1:
            used = np.zeros(m)
            for i in best_items:
                used += w[:, i]
            cur_val = best_found
            in_pack = set(best_items)
            candidates = [i for i in range(n) if i not in in_pack]
            best_swap_val = cur_val
            best_swap_items = best_items[:]

            for a in best_items:
                if _deadline_left() < MARGIN:
                    break
                rem_after_remove = C - (used - w[:, a])
                val_after_remove = cur_val - p[a]
                for b in candidates:
                    if np.all(w[:, b] <= rem_after_remove):
                        new_val = val_after_remove + p[b]
                        if new_val > best_swap_val:
                            best_swap_val = new_val
                            best_swap_items = [x for x in best_items if x != a] + [b]

            if best_swap_val > cur_val:
                for i in best_swap_items:
                    contrib[i] += best_swap_val
                    selection_freq[i] += 2.0
                best_found = best_swap_val

                # Second polish round
                if _deadline_left() > MARGIN + 3.0:
                    best_items2 = best_swap_items[:]
                    in_pack2 = set(best_items2)
                    used2 = np.zeros(m)
                    for i in best_items2:
                        used2 += w[:, i]
                    cur_val2 = best_swap_val
                    candidates2 = [i for i in range(n) if i not in in_pack2]
                    for a in best_items2:
                        if _deadline_left() < MARGIN:
                            break
                        rem_after_remove2 = C - (used2 - w[:, a])
                        val_after_remove2 = cur_val2 - p[a]
                        for b in candidates2:
                            if np.all(w[:, b] <= rem_after_remove2):
                                new_val2 = val_after_remove2 + p[b]
                                if new_val2 > best_swap_val:
                                    best_swap_val = new_val2
                                    best_swap_items = [x for x in best_items2 if x != a] + [b]
                    if best_swap_val > best_found:
                        for i in best_swap_items:
                            contrib[i] += best_swap_val
                            selection_freq[i] += 1.0

    # ------------------------------------------------------------------
    # 3c) Regret (leave-one-out) blocking score from the baseline greedy
    # ------------------------------------------------------------------
    baseline_order = orderings[0]
    baseline_rem = C.copy()
    baseline_val = 0.0
    for i in baseline_order:
        if np.all(w[:, i] <= baseline_rem):
            baseline_rem -= w[:, i]
            baseline_val += p[i]

    regret = np.zeros(n, dtype=float)
    for i in range(n):
        if _deadline_left() < MARGIN + 1.0:
            break
        rem = C.copy()
        val = 0.0
        for j in baseline_order:
            if j == i:
                continue
            if np.all(w[:, j] <= rem):
                rem -= w[:, j]
                val += p[j]
        regret[i] = max(0.0, baseline_val - val)

    # ------------------------------------------------------------------
    # 4) Combine signals into the final attractiveness score.
    #    Dominant: greedy contribution (captures blocking + value).
    #    Secondary: reduced cost, prize, soft-slack bottleneck, frequency,
    #    and inverse regret (non-blocking items favored).
    # ------------------------------------------------------------------
    cmax = contrib.max()
    if cmax > 0:
        contrib_norm = contrib / cmax
    else:
        contrib_norm = np.ones(n)

    fmax = selection_freq.max()
    if fmax > 0:
        freq_norm = selection_freq / fmax
    else:
        freq_norm = np.ones(n)

    pmax = p.max()
    if pmax > 0:
        p_norm = p / pmax
    else:
        p_norm = np.ones(n)

    ssmax = soft_slack.max()
    if ssmax > 0:
        ss_norm = soft_slack / ssmax
    else:
        ss_norm = np.ones(n)

    rmax = regret.max()
    if rmax > 0:
        regret_norm = 1.0 - (regret / rmax)
    else:
        regret_norm = np.ones(n)

    score = (
        (contrib_norm ** 2.0)
        * (rc_norm ** 1.0)
        * (p_norm ** 1.0)
        * (ss_norm ** 0.5)
        * (freq_norm ** 0.5)
        * (regret_norm ** 1.0)
    )
    score = score + 1e-9

    # ------------------------------------------------------------------
    # 5) Calibrate magnitude so the heuristic's own greedy total matches
    #    the best achievable pack value (pure scaling, ordering preserved).
    # ------------------------------------------------------------------
    if best_found > 0:
        order = np.argsort(-score, kind="stable")
        rem = C.copy()
        val = 0.0
        for i in order:
            if np.all(w[:, i] <= rem):
                rem -= w[:, i]
                val += p[i]
        if val > 0:
            score = score * (best_found / val)

    score = _safe(score)
    smax = score.max()
    if smax > 0:
        score = score / smax
    score = _safe(score)
    return score

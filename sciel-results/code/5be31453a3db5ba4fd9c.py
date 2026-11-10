"""Direct, structure-driven heuristic for ACO multiple-constraint knapsack."""
import time
import numpy as np

T_START = time.perf_counter()
LIMIT = 300.0   # hard wall-clock bound from evaluator
MARGIN = 8.0    # safety margin
DEADLINE = T_START + (LIMIT - MARGIN)


def _deadline_left():
    return (T_START + LIMIT) - time.perf_counter()


def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Normalize shapes -> per-item prize (n,) and per-(constraint,item) weight (m, n)
    if prizes.ndim == 2:
        p = prizes.sum(axis=0)
    else:
        p = prizes.ravel()
    if weights.ndim == 1:
        w = weights.reshape(1, -1)
    elif weights.ndim == 2 and weights.shape[0] != weights.shape[1]:
        w = weights
    elif weights.ndim == 2 and weights.shape[1] == p.shape[0]:
        w = weights  # already (m, n)
    else:
        w = weights.reshape(1, -1)

    n = int(p.shape[0])
    m = int(w.shape[0])
    cap = 25.0
    C = np.full(m, cap, dtype=float)

    # Safety: ensure finite, positive output of correct shape
    def _safe(a):
        a = np.where(np.isfinite(a), a, 1.0)
        a = np.maximum(a, 1e-9)
        return a

    if n == 0:
        return np.ones(0)

    # ------------------------------------------------------------------
    # 1) Lagrangian / subgradient: estimate shadow prices lambda for each
    #    constraint. Reduced cost rc_i = p_i - sum_c lambda_c * w_c_i.
    #    Items with high rc are "profitable after paying for resources".
    # ------------------------------------------------------------------
    lam = np.zeros(m, dtype=float)
    best_rc = p.copy()
    best_rc_val = -1.0
    best_packs = []  # (value, [item order])

    # subgradient steps: lambda_{c} <- max(0, lambda_c + step * slack_c)
    n_sub = 25
    for it in range(n_sub):
        if _deadline_left() < MARGIN + 2.0:
            break
        shadow = lam @ w                 # (n,)
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
            best_packs.append((val, sel))
        used = np.zeros(m)
        for i in sel:
            used += w[:, i]
        slack = cap - used
        step = (4.0 / (1.0 + it * 0.15))
        lam = np.maximum(0.0, lam + step * slack)

    # Normalize reduced cost to [0,1]
    rmin = best_rc.min()
    rmax = best_rc.max()
    if rmax > rmin:
        rc_norm = (best_rc - rmin) / (rmax - rmin)
    else:
        rc_norm = np.ones(n)
    rc_norm = np.clip(rc_norm, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 2) Precompute structural weight metrics
    # ------------------------------------------------------------------
    sum_w = w.sum(axis=0)
    max_w = w.max(axis=0)
    min_w = w.min(axis=0)
    avg_w = w.mean(axis=0)
    safe_max = np.maximum(max_w, 1e-9)

    # Bottleneck slack: min over c of (cap - w_c)/w_c  -> "how easily it fits"
    slack_ratios = (cap - w) / np.maximum(w, 1e-9)
    bottleneck_slack = np.maximum(slack_ratios.min(axis=0), 0.0)

    # Soft-min smooth surrogate for bottleneck constraint tightness.
    # For each item, compute how "tight" it is across constraints:
    #   tightness_c = w_c_i / cap  (fraction of capacity consumed)
    # soft-min of (1 - tightness) via log-sum-exp gives a smooth measure
    # of how much slack remains after considering all constraints jointly.
    tightness = w / cap  # (m, n), fraction of each capacity used
    # We want a smooth version of min_c (1 - tightness_c) = max_c (tightness_c) inverted
    # Use: soft_max(tightness, beta) via log-sum-exp for a smooth max
    beta_soft = 10.0
    # Clip for numerical stability
    t_clip = np.clip(tightness, 0.0, 1.0)
    t_shift = t_clip - t_clip.max(axis=0, keepdims=True)
    log_sum_exp = t_shift.max(axis=0) + np.log(np.sum(np.exp(t_shift), axis=0))
    soft_max_tightness = t_clip.max(axis=0, keepdims=True)[0] + log_sum_exp - np.log(m)
    # Smooth slack: 1 - soft_max_tightness, clipped to [0,1]
    soft_slack = np.clip(1.0 - soft_max_tightness, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 3) Many fast greedy simulations with diverse orderings.
    #    Vectorized greedy: use numpy to check feasibility in bulk.
    #    Score each item by its total prize contribution to high-value packs.
    # ------------------------------------------------------------------
    contrib = np.zeros(n, dtype=float)   # total prize contributed per item
    selection_freq = np.zeros(n, dtype=float)
    best_found = 0.0
    best_sel_set = set()

    def _greedy_vec(order):
        """Vectorized greedy: process items in given order, add if feasible."""
        nonlocal best_found, best_sel_set
        rem = C.copy()  # (m,)
        val = 0.0
        sel = []
        # Process items in order; for n=100, m=5, even a Python loop is fast,
        # but we vectorize the feasibility check across remaining items.
        remaining = list(order)
        while remaining:
            if _deadline_left() < MARGIN:
                break
            # Check which of the remaining items fit
            # w[:, remaining] has shape (m, k); rem has shape (m,)
            fits = np.all(w[:, remaining] <= rem[:, None], axis=0)
            if not np.any(fits):
                break
            # Take the first fitting item (preserves order)
            first_idx = np.argmax(fits)
            item = remaining[first_idx]
            sel.append(item)
            rem -= w[:, item]
            val += p[item]
            remaining = remaining[:first_idx] + remaining[first_idx+1:]
        if val > 0:
            for i in sel:
                contrib[i] += val
                selection_freq[i] += 1.0
        if val > best_found:
            best_found = val
            best_sel_set = set(sel)
        return val

    # Build a set of orderings (deterministic ratios + Lagrangian + soft-slack + random)
    orderings = []
    orderings.append(np.argsort(-rc_norm, kind="stable"))
    orderings.append(np.argsort(-(p / (sum_w + 1e-9)), kind="stable"))
    orderings.append(np.argsort(-(p / (safe_max)), kind="stable"))
    orderings.append(np.argsort(-(p / (safe_max ** 2 + 1e-9)), kind="stable"))
    orderings.append(np.argsort(-(p / (avg_w + 1e-9)), kind="stable"))
    orderings.append(np.argsort(-(p / (min_w + 1e-9)), kind="stable"))
    orderings.append(np.argsort(-(p * (bottleneck_slack + 1.0) / (safe_max + 1.0)), kind="stable"))
    orderings.append(np.argsort(-p, kind="stable"))
    # New: ordering by soft-slack (smooth surrogate for bottleneck)
    orderings.append(np.argsort(-(p * (soft_slack + 0.1) / (safe_max + 1.0)), kind="stable"))
    # New: pure soft-slack ordering
    orderings.append(np.argsort(-soft_slack, kind="stable"))

    # A few deterministic pseudo-random shuffles (fixed seeds) for diversity
    for seed in (7, 13, 42, 99, 123):
        rng = np.random.RandomState(seed)
        orderings.append(np.array(rng.permutation(n)))

    # Run greedy until deadline
    max_rounds = 400
    for r in range(max_rounds):
        if _deadline_left() < MARGIN:
            break
        for order in orderings:
            if _deadline_left() < MARGIN:
                break
            _greedy_vec(order)

    # ------------------------------------------------------------------
    # 3b) Local polish: try swap-based local search on the best pack found.
    #     For the best pack, try removing one item and adding a different
    #     one that fits, keeping the best improvement. Credit improved packs.
    # ------------------------------------------------------------------
    if best_found > 0 and _deadline_left() > MARGIN + 5.0:
        # Get the best pack's items
        best_items = list(best_sel_set)
        if len(best_items) > 1:
            # Compute current used resources
            used = np.zeros(m)
            for i in best_items:
                used += w[:, i]
            cur_val = best_found

            # Try each swap: remove item a, add item b not in pack
            in_pack = set(best_items)
            candidates = [i for i in range(n) if i not in in_pack]
            improved = False
            best_swap_val = cur_val
            best_swap_items = best_items[:]

            for a in best_items:
                if _deadline_left() < MARGIN:
                    break
                # Remove a
                rem_after_remove = C - (used - w[:, a])
                val_after_remove = cur_val - p[a]
                # Try adding each candidate
                for b in candidates:
                    if np.all(w[:, b] <= rem_after_remove):
                        new_val = val_after_remove + p[b]
                        if new_val > best_swap_val:
                            best_swap_val = new_val
                            best_swap_items = [x for x in best_items if x != a] + [b]
                            improved = True

            if improved:
                # Credit the improved pack
                for i in best_swap_items:
                    contrib[i] += best_swap_val
                    selection_freq[i] += 2.0  # weight higher for polished packs
                best_found = max(best_found, best_swap_val)

                # Second round of swaps
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
                        best_found = best_swap_val

    # ------------------------------------------------------------------
    # 4) Combine signals into the final attractiveness score.
    #    Dominant signal: contribution to good packs (captures blocking).
    #    Secondary: reduced cost, soft-slack (smooth bottleneck), raw prize.
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

    bmax = bottleneck_slack.max()
    if bmax > 0:
        bs_norm = bottleneck_slack / bmax
    else:
        bs_norm = np.ones(n)

    ssmax = soft_slack.max()
    if ssmax > 0:
        ss_norm = soft_slack / ssmax
    else:
        ss_norm = np.ones(n)

    # Product-form: emphasize contribution & reduced cost strongly,
    # include soft-slack as a smooth bottleneck signal.
    score = (
        (contrib_norm ** 2.0)
        * (rc_norm ** 1.0)
        * (p_norm ** 1.0)
        * (bs_norm ** 0.5)
        * (ss_norm ** 0.5)
        * (freq_norm ** 0.5)
    )
    score = score + 1e-9

    # ------------------------------------------------------------------
    # 5) Calibrate: the ACO uses heuristic in iteration 1 (pheromone=1).
    #    Scale the heuristic so its own greedy total matches best_found,
    #    i.e. the heuristic's relative ordering is preserved but its
    #    magnitude reflects the achievable pack value. This is a pure
    #    scaling (relative ratios unchanged) so it cannot harm ordering.
    # ------------------------------------------------------------------
    if best_found > 0:
        # greedy total using this score as the ordering proxy
        order = np.argsort(-score, kind="stable")
        rem = C.copy()
        val = 0.0
        for i in order:
            if np.all(w[:, i] <= rem):
                rem -= w[:, i]
                val += p[i]
        if val > 0:
            scale = best_found / val
            score = score * scale

    score = _safe(score)
    smax = score.max()
    if smax > 0:
        score = score / smax
    score = _safe(score)
    return score

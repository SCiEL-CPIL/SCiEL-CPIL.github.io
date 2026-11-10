import time
import numpy as np

T_START = time.perf_counter()
LIMIT = 300.0
MARGIN = 10.0

def _deadline_left():
    return (T_START + LIMIT) - time.perf_counter()

def _greedy_value(order, weights, prizes, C):
    """
    Simulate greedy packing given an order of items.
    Returns the total prize of the feasible set.
    """
    n_constraints = weights.shape[0]
    rem = C.copy()
    val = 0.0
    for i in order:
        if np.all(weights[:, i] <= rem):
            rem -= weights[:, i]
            val += prizes[i]
    return val

def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Ensure shapes
    if prizes.ndim == 2:
        p = prizes.sum(axis=0)
    else:
        p = prizes.ravel()
    
    if weights.ndim == 1:
        w = weights.reshape(1, -1)
    elif weights.ndim == 2:
        w = weights
    else:
        w = weights.reshape(5, -1)

    n = p.shape[0]
    m = w.shape[0]
    
    if n == 0:
        return np.ones(0)

    cap = 25.0
    C = np.full(m, cap, dtype=float)

    # 1. Compute Shadow Prices (Lagrangian Dual)
    # Use subgradient method to find lambda
    lam = np.zeros(m)
    best_dual_val = -1.0
    best_lam = lam.copy()
    
    # Step size parameters
    n_sub = 50 # Fast enough
    step_init = 10.0
    
    for it in range(n_sub):
        if _deadline_left() < MARGIN + 5.0:
            break
        # Reduced costs
        shadow_cost = w.T @ lam # shape (n,)
        rc = p - shadow_cost
        
        # Greedy pick based on rc
        order = np.argsort(-rc, kind='stable')
        val = _greedy_value(order, w, p, C)
        
        # Dual bound is sum of positive reduced costs? No, that's for LP.
        # For heuristic fitness, we care about the primal value found.
        if val > best_dual_val:
            best_dual_val = val
            best_lam = lam.copy()
        
        # Update lambda
        # Subgradient of dual function is slack = C - used_capacity
        # We need to re-calc used for the update to be precise.
        rem = C.copy()
        used = np.zeros(m)
        for i in order:
            if rc[i] <= 0:
                continue
            if np.all(w[:, i] <= rem):
                rem -= w[:, i]
                used += w[:, i]
        
        slack = C - used
        # Step size decay
        step = step_init / (1.0 + it * 0.5)
        lam = np.maximum(0.0, lam + step * slack)

    # Use the best lam found
    lam = best_lam
    shadow_cost = w.T @ lam
    rc = p - shadow_cost
    
    # 2. Prepare Base Signals
    # Signal 1: Reduced Cost (RC)
    # Non-positive RCs are bad. We will threshold them.
    rc_pos = np.where(rc > 0, rc, 0.0)
    rc_max = rc_pos.max()
    if rc_max > 0:
        rc_norm = rc_pos / rc_max
    else:
        rc_norm = np.ones(n)
        
    # Signal 2: Density (Prize / Weight)
    # Use sum of weights as a proxy for total resource cost
    sum_w = w.sum(axis=0)
    safe_sum_w = np.maximum(sum_w, 1e-9)
    density = p / safe_sum_w
    d_max = density.max()
    if d_max > 0:
        d_norm = density / d_max
    else:
        d_norm = np.ones(n)

    # Signal 3: Bottleneck Density (Prize / Max Weight in any constraint)
    max_w = w.max(axis=0)
    safe_max_w = np.maximum(max_w, 1e-9)
    bot_density = p / safe_max_w
    b_max = bot_density.max()
    if b_max > 0:
        b_norm = bot_density / b_max
    else:
        b_norm = np.ones(n)
        
    # Signal 4: Bottleneck Slack (Feasibility Margin)
    # From Inspiration 1: min_c (capacity - w_ic) / w_ic
    # This measures how easily an item fits into the tightest constraint.
    # High slack means the item is "small" relative to the capacity in its worst dimension.
    safe_weights = np.maximum(w, 1e-9)
    slack_ratios = (cap - w) / safe_weights
    slack_ratios = np.maximum(slack_ratios, 0.0)
    bottleneck_slack = slack_ratios.min(axis=0)
    bs_max = bottleneck_slack.max()
    if bs_max > 0:
        bs_norm = bottleneck_slack / bs_max
    else:
        bs_norm = np.ones(n)
        
    # Signal 5: Multi-Start Greedy Frequency
    # From Inspiration 1: Items frequently selected in high-value greedy packs are "safe".
    # We run a few quick greedy passes with different sorting criteria and accumulate
    # a "weighted count" of how often each item appears in the selected set.
    # This captures discrete blocking effects better than static ratios.
    
    weighted_counts = np.zeros(n, dtype=float)
    
    # Define strategies for multi-start greedy
    strategies = []
    
    # Strategy 1: Prize / Sum Weight (Standard Density)
    strategies.append(np.argsort(-density, kind='stable'))
    
    # Strategy 2: Prize / Max Weight (Bottleneck Density)
    strategies.append(np.argsort(-bot_density, kind='stable'))
    
    # Strategy 3: Prize only
    strategies.append(np.argsort(-p, kind='stable'))
    
    # Strategy 4: Reduced Cost
    strategies.append(np.argsort(-rc, kind='stable'))
    
    # Strategy 5: Combined Slack and Prize
    # Prioritize items that fit easily and have high prize
    combo_slack = (bottleneck_slack + 1e-9) * p
    strategies.append(np.argsort(-combo_slack, kind='stable'))
    
    # Strategy 6: Random (Seeded for reproducibility)
    rng = np.random.RandomState(42)
    strategies.append(rng.permutation(n))
    
    # Run greedy simulations
    for order in strategies:
        if _deadline_left() < MARGIN + 2.0:
            break
        val = _greedy_value(order, w, p, C)
        if val > 0:
            # Accumulate weighted counts
            # We weight the count by the total value of the pack to emphasize
            # items that appear in *good* packs, not just any pack.
            # To do this efficiently without re-simulating, we can just add 1 for each selected item.
            # But re-simulating to get the set is O(N). We already did the sim to get val.
            # Let's re-run to get the set for counting. It's fast for N=100.
            rem = C.copy()
            for i in order:
                if np.all(w[:, i] <= rem):
                    rem -= w[:, i]
                    weighted_counts[i] += val # Weight by total pack value
    
    wc_max = weighted_counts.max()
    if wc_max > 0:
        wc_norm = weighted_counts / wc_max
    else:
        wc_norm = np.ones(n)

    # 3. Fit/Select Best Heuristic Combination
    # We want to maximize the quality of the greedy solution (proxy for ACO success)
    # We combine the static signals (RC, Density, Slack) with the dynamic signal (Weighted Counts).
    
    # Candidate functions:
    # We test a few combinations of the normalized signals.
    # The goal is to find a heuristic that guides the ACO ants to find high-value feasible sets.
    
    candidates = []
    
    # H1: Pure RC
    candidates.append(rc_norm)
    
    # H2: Pure Density
    candidates.append(d_norm)
    
    # H3: Pure Bottleneck Slack
    candidates.append(bs_norm)
    
    # H4: Pure Weighted Counts
    candidates.append(wc_norm)
    
    # H5: RC * WC
    candidates.append(rc_norm * wc_norm)
    
    # H6: RC * WC * BS
    candidates.append(rc_norm * wc_norm * bs_norm)
    
    # H7: (RC + WC) * BS
    candidates.append((rc_norm + wc_norm) * bs_norm)
    
    # H8: RC^1.5 * WC^1.0 * BS^0.5
    candidates.append((rc_norm ** 1.5) * (wc_norm ** 1.0) * (bs_norm ** 0.5))
    
    # H9: WC^1.5 * BS^1.0 (Emphasize safety and frequency)
    candidates.append((wc_norm ** 1.5) * (bs_norm ** 1.0))
    
    # H10: RC * D_norm * BS
    candidates.append(rc_norm * d_norm * bs_norm)
    
    # H11: RC * WC (Previous winner candidate, keep it)
    # H12: Soft-max combination of RC and WC
    # H13: Geometric mean of RC and WC
    candidates.append(np.sqrt(rc_norm * wc_norm + 1e-9))
    
    best_score = -1.0
    best_heur = candidates[0]
    
    # Fast greedy evaluation to pick best heuristic
    # We simulate a simple greedy: sort by H, pick if fits.
    # We do this for all candidates.
    for h in candidates:
        if _deadline_left() < MARGIN + 2.0:
            break
        order = np.argsort(-h, kind='stable')
        val = _greedy_value(order, w, p, C)
        if val > best_score:
            best_score = val
            best_heur = h
            
    # Ensure output is valid: positive, finite, correct shape
    final_heur = np.asarray(best_heur, dtype=float)
    final_heur = np.where(np.isfinite(final_heur), final_heur, 1.0)
    final_heur = np.maximum(final_heur, 1e-9)
    
    # Normalize to [0, 1] to ensure stability in ACO probability calculation
    # The ACO uses pheromone * heuristic. If heuristic is very large, it might dominate.
    # If it's very small, it might be ignored.
    # Normalizing to [0, 1] is a safe standard practice.
    f_max = final_heur.max()
    if f_max > 0:
        final_heur = final_heur / f_max
    
    # Ensure minimum value is 1e-9 as per task description
    final_heur = np.maximum(final_heur, 1e-9)
    
    return final_heur

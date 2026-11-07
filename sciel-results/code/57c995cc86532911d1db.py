"""Heuristic for ACO bin packing: adaptive selection within tightness+scarcity family."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit

    # Precompute common quantities
    pair_sum = demands[:, None] + demands[None, :]
    fits = pair_sum <= capacity
    waste = capacity - pair_sum
    degree = fits.sum(axis=0).astype(float)
    degree_safe = np.maximum(degree, 1.0)
    scarcity = 1.0 / degree_safe  # shape (n,)

    eps = 1e-6

    # Size-based priority: larger items are harder to place, give them higher weight
    # Normalized so that the largest item gets weight 1.0 and smallest gets a small floor
    d_min = demands.min()
    d_max = demands.max()
    if d_max > d_min:
        size_norm = (demands - d_min) / (d_max - d_min)  # in [0, 1]
    else:
        size_norm = np.ones(n)
    size_weight = 0.5 + 0.5 * size_norm  # in [0.5, 1.0], larger items get more weight

    # Combined scarcity + size priority: blend scarcity with size-based priority
    # Larger items have fewer compatible partners AND are harder to fit, so emphasize both
    combined_weight = 0.7 * scarcity + 0.3 * size_weight  # shape (n,)

    # Candidate heuristics within the tightness+scarcity family
    candidates = []

    # 1. Base: tightness * combined_weight
    tightness = np.where(fits, 1.0 / (waste + eps), 0.0)
    h1 = tightness * combined_weight[None, :]
    candidates.append(h1)

    # 2. Sharper tightness: (1/waste)^2 * combined_weight
    tightness2 = np.where(fits, 1.0 / (waste + eps)**2, 0.0)
    h2 = tightness2 * combined_weight[None, :]
    candidates.append(h2)

    # 3. Exponential tightness with tight scale (sharper discrimination)
    for scale in [5.0, 8.0, 12.0]:
        h_exp = np.where(fits, np.exp(-waste / scale), 0.0)
        h = h_exp * combined_weight[None, :]
        candidates.append(h)

    # 3b. Soft-max-inspired smooth discrimination: use log-sum-exp style
    #     to create a smooth surrogate for the tightest-fit preference
    #     This avoids the discontinuity of hard argmax while still emphasizing tight fits
    waste_safe = np.where(fits, waste + eps, np.inf)
    # Compute a smooth "tightness rank" via softmax over waste values per row
    # Lower waste = higher attractiveness; use negative waste for softmax
    neg_waste = np.where(fits, -waste, -np.inf)
    # Softmax with temperature for smooth discrimination
    for temp in [3.0, 5.0, 10.0]:
        scaled = neg_waste / temp
        # Numerically stable softmax
        row_max = np.max(scaled, axis=1, keepdims=True)
        row_max = np.where(np.isfinite(row_max), row_max, 0)
        exp_scaled = np.exp(scaled - row_max)
        exp_scaled = np.where(np.isfinite(exp_scaled), exp_scaled, 0.0)
        softmax_vals = exp_scaled / np.maximum(exp_scaled.sum(axis=1, keepdims=True), eps)
        # Convert to heuristic: higher softmax probability = more attractive
        h_sm = np.where(fits, softmax_vals * n, 0.0)  # scale by n to be comparable
        h = h_sm * combined_weight[None, :]
        candidates.append(h)

    # 4. Tightness * combined_weight with size-similarity bonus
    d_i = demands[:, None]
    d_j = demands[None, :]
    denom = np.maximum(d_i * d_j, eps)
    sim = (d_i * d_j) / (denom + 1.0)  # in (0,1), ~1 when sizes similar
    h4 = tightness * combined_weight[None, :] * (0.5 + 0.5 * sim)
    candidates.append(h4)

    # 5. Tightness * combined_weight with fill-ratio bonus
    fill_ratio = pair_sum / capacity
    fill_bonus = 1.0 / (1.0 + np.abs(fill_ratio - 1.0) + eps)
    h5 = tightness * combined_weight[None, :] * fill_bonus
    candidates.append(h5)

    # 6. Power-law tightness: (1/waste)^p * combined_weight
    for p in [1.5, 2.5]:
        h_pow = np.where(fits, 1.0 / (waste + eps)**p, 0.0)
        h = h_pow * combined_weight[None, :]
        candidates.append(h)

    # 6b. Hybrid: blend sharp tightness with smooth softmax for robustness
    #     Use the best power-law (p=2.0) blended with medium-temperature softmax
    h_pow2 = np.where(fits, 1.0 / (waste + eps)**2, 0.0)
    # Reuse temp=5.0 softmax from above (recompute for clarity)
    neg_waste_5 = np.where(fits, -waste, -np.inf)
    scaled_5 = neg_waste_5 / 5.0
    row_max_5 = np.max(scaled_5, axis=1, keepdims=True)
    row_max_5 = np.where(np.isfinite(row_max_5), row_max_5, 0)
    exp_5 = np.exp(scaled_5 - row_max_5)
    exp_5 = np.where(np.isfinite(exp_5), exp_5, 0.0)
    sm_5 = exp_5 / np.maximum(exp_5.sum(axis=1, keepdims=True), eps)
    sm_scaled = np.where(fits, sm_5 * n, 0.0)
    # Normalize both to [0,1] range for blending
    h_pow2_norm = h_pow2 / np.maximum(h_pow2.max(), eps)
    sm_norm = sm_scaled / np.maximum(sm_scaled.max(), eps)
    for alpha in [0.3, 0.5, 0.7]:
        h_blend = alpha * h_pow2_norm + (1 - alpha) * sm_norm
        h = h_blend * combined_weight[None, :]
        candidates.append(h)

    # Clean all candidates
    clean_candidates = []
    for h in candidates:
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        clean_candidates.append(h)

    # If time is tight, just return the base heuristic
    if time.time() > deadline - 30:
        return clean_candidates[0]

    # Fast ACO simulation to evaluate candidates
    eval_seeds = [42, 123, 999, 7, 2024]
    best_h = clean_candidates[0]
    best_score = float('inf')

    # Use more ants/iters for final evaluation of top candidates
    for idx, H in enumerate(clean_candidates):
        if time.time() > deadline - 20:
            break

        scores = []
        for seed in eval_seeds:
            if time.time() > deadline - 15:
                break
            bins_used = _simulate_aco_fast(demands, capacity, H, seed=seed, ants=15, iters=15)
            scores.append(bins_used)

        if len(scores) == len(eval_seeds):
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_h = H

    # Polish: if we have time, re-evaluate the best with more iterations
    if time.time() < deadline - 30 and best_h is not clean_candidates[0]:
        polish_scores = []
        for seed in eval_seeds:
            if time.time() > deadline - 25:
                break
            bins_used = _simulate_aco_fast(demands, capacity, best_h, seed=seed, ants=20, iters=20)
            polish_scores.append(bins_used)
        if len(polish_scores) == len(eval_seeds):
            polish_avg = np.mean(polish_scores)
            # Also re-evaluate the base candidate with polish settings for fair comparison
            base_polish = []
            for seed in eval_seeds:
                if time.time() > deadline - 25:
                    break
                b = _simulate_aco_fast(demands, capacity, clean_candidates[0], seed=seed, ants=20, iters=20)
                base_polish.append(b)
            if len(base_polish) == len(eval_seeds) and polish_avg > np.mean(base_polish):
                best_h = clean_candidates[0]

    # Final sanity check
    best_h = np.where(np.isfinite(best_h) & (best_h > 0), best_h, 1e-9)
    return best_h


def _simulate_aco_fast(demands, capacity, H, seed=42, ants=15, iters=15):
    """Fast ACO simulation for heuristic evaluation."""
    rng = np.random.RandomState(seed)
    n = len(demands)
    W = demands

    tau = np.ones((n, n))
    min_bins_global = float('inf')

    for it in range(iters):
        min_bins_this_iter = float('inf')

        for ant in range(ants):
            used = np.zeros(n, dtype=bool)
            bins_used = 0
            items_left = n

            while items_left > 0:
                bins_used += 1
                unused_indices = np.where(~used)[0]
                if len(unused_indices) == 0:
                    break

                # Random fitting item to start
                start_idx = unused_indices[rng.randint(len(unused_indices))]
                used[start_idx] = True
                items_left -= 1
                current = start_idx
                current_load = W[current]

                while True:
                    remaining = capacity - current_load
                    fits = (~used) & (W <= remaining)

                    if not np.any(fits):
                        break

                    fit_indices = np.where(fits)[0]
                    probs = tau[current, fit_indices] * H[current, fit_indices]
                    probs = np.maximum(probs, 1e-12)
                    probs /= probs.sum()

                    next_idx = fit_indices[rng.choice(len(fit_indices), p=probs)]
                    used[next_idx] = True
                    items_left -= 1
                    current_load += W[next_idx]
                    current = next_idx

            min_bins_this_iter = min(min_bins_this_iter, bins_used)

        # Pheromone update
        tau *= 0.9
        tau += 1.0 / min_bins_this_iter

        min_bins_global = min(min_bins_global, min_bins_this_iter)

    return min_bins_global
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 295.0  # safety margin before 300s limit


def heuristics(prizes, weights):
    t0 = time.time()
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("Expected shape (5, 100)")

    n_items = 100
    n_constraints = 5
    capacity = 25.0

    # Ensure finite and positive
    prizes = np.where(np.isfinite(prizes), prizes, 1e-9)
    weights = np.where(np.isfinite(weights), weights, 1e-9)
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)

    # Spectral centrality: build weight co-occurrence matrix
    # W is (5, 100), so W.T @ W is (100, 100)
    W = weights
    co_occurrence = W.T @ W  # (100, 100)

    # Normalize: use row-normalized version to get a similarity-like matrix
    row_sums = co_occurrence.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    normalized = co_occurrence / row_sums  # (100, 100)

    # Principal eigenvector of the normalized matrix
    # Use power iteration for speed
    v = np.ones(n_items) / np.sqrt(n_items)
    for _ in range(50):
        v_new = normalized @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-15:
            break
        v_new /= norm
        if np.allclose(v, v_new, atol=1e-10):
            break
        v = v_new
    centrality = np.abs(v)  # (100,)

    # Per-constraint efficiency ratios
    ratios = prizes / weights  # (5, 100)

    # Soft-min across constraints using log-sum-exp trick
    # softmin(x_1,...,x_k) = -logsumexp(-x_1,...,-x_k)
    neg_ratios = -ratios  # (5, 100)
    max_neg = neg_ratios.max(axis=0, keepdims=True)
    exp_neg = np.exp(neg_ratios - max_neg)  # (5, 100)
    logsumexp_neg = max_neg.squeeze(0) + np.log(exp_neg.sum(axis=0))  # (100,)
    softmin = -logsumexp_neg  # (100,)

    # Combine: prize-weighted efficiency with centrality
    # Items that are efficient on all constraints AND structurally central
    base_heuristic = softmin * centrality  # (100,)

    # Also consider direct prize/total_weight as a baseline
    total_weight = weights.sum(axis=0)  # (100,)
    direct_efficiency = prizes.mean(axis=0) / (total_weight + 1e-9)  # (100,)

    # Blend spectral centrality heuristic with direct efficiency
    # Normalize both to [0,1] range for blending
    base_min, base_max = base_heuristic.min(), base_heuristic.max()
    if base_max > base_min:
        base_norm = (base_heuristic - base_min) / (base_max - base_min)
    else:
        base_norm = np.ones(n_items) / n_items

    eff_min, eff_max = direct_efficiency.min(), direct_efficiency.max()
    if eff_max > eff_min:
        eff_norm = (direct_efficiency - eff_min) / (eff_max - eff_min)
    else:
        eff_norm = np.ones(n_items) / n_items

    # Weighted blend: favor the spectral/soft-min approach
    heuristic = 0.7 * base_norm + 0.3 * eff_norm

    # Ensure all positive
    heuristic = np.maximum(heuristic, 1e-9)

    # Deadline check
    if time.time() - t0 > _DEADLINE:
        return heuristic

    # Optional: quick stochastic refinement if time permits
    elapsed = time.time() - t0
    if elapsed < 250.0:
        # Simulate a few quick ACO-like trials to refine heuristic weights
        rng = np.random.default_rng(42)
        trial_scores = np.zeros(n_items)
        n_trials = 20
        for _ in range(n_trials):
            remaining_cap = np.full(n_constraints, capacity)
            total_prize = 0.0
            selected = set()
            # Greedy with stochastic perturbation
            order = np.argsort(-heuristic + rng.normal(0, 0.1, n_items))
            for item in order:
                fits = np.all(weights[:, item] <= remaining_cap)
                if fits:
                    remaining_cap -= weights[:, item]
                    total_prize += prizes[:, item].sum()
                    selected.add(item)
            # Credit items selected in good trials
            if total_prize > capacity * 0.5:  # decent pack
                for item in selected:
                    trial_scores[item] += 1.0

        # Blend trial-based scores
        if trial_scores.max() > 0:
            trial_norm = trial_scores / trial_scores.max()
            heuristic = 0.8 * heuristic + 0.2 * trial_norm
            heuristic = np.maximum(heuristic, 1e-9)

    # Final safety: ensure finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END

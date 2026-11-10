import numpy as np
import time
from scipy.optimize import minimize

def _compute_ratio_fast(w, diff_bins, h0_bins, h1_bins, h2_bins,
                        n_bins_diff, n_bins_h0, n_bins_h1, n_bins_h2):
    """Fast ratio computation using precomputed bin indices."""
    def entropy_from_bins(bins, weights, n_bins):
        counts = np.zeros(n_bins)
        np.add.at(counts, bins, weights)
        probs = counts[counts > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))

    D = entropy_from_bins(diff_bins, w, n_bins_diff)
    H0 = entropy_from_bins(h0_bins, w, n_bins_h0)
    H1 = entropy_from_bins(h1_bins, w, n_bins_h1)
    H2 = entropy_from_bins(h2_bins, w, n_bins_h2)

    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0
    return D / M

def _soft_max_entropy_ratio(w, diff_bins, h0_bins, h1_bins, h2_bins,
                            n_bins_diff, n_bins_h0, n_bins_h1, n_bins_h2, beta=50.0):
    """Compute D/M with soft-max for the M term, suitable for gradient-based optimization."""
    def entropy_from_bins(bins, weights, n_bins):
        counts = np.zeros(n_bins)
        np.add.at(counts, bins, weights)
        probs = counts[counts > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))

    D = entropy_from_bins(diff_bins, w, n_bins_diff)
    H0 = entropy_from_bins(h0_bins, w, n_bins_h0)
    H1 = entropy_from_bins(h1_bins, w, n_bins_h1)
    H2 = entropy_from_bins(h2_bins, w, n_bins_h2)

    H_vals = np.array([H0, H1, H2])
    H_max = np.max(H_vals)
    if H_max <= 1e-9:
        return 0.0

    # Soft-max: smooth approximation of max(H0, H1, H2)
    scaled = beta * H_vals
    scaled_max = np.max(scaled)
    exp_scaled = np.exp(scaled - scaled_max)
    soft_max = (np.log(np.sum(exp_scaled)) + scaled_max) / beta

    if soft_max <= 1e-9:
        return 0.0
    return D / soft_max

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    y_idx, x_idx = np.indices((n, n))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()

    start_time = time.time()
    timeout = 600
    safety_margin = 40

    # Precompute bin indices
    diff_vals = x_flat - y_flat
    diff_bins = diff_vals + 9
    n_bins_diff = 19

    h0_vals = x_flat
    h0_bins = h0_vals
    n_bins_h0 = 10

    h1_vals = x_flat + y_flat
    h1_bins = h1_vals
    n_bins_h1 = 19

    h2_vals = x_flat + 2 * y_flat
    h2_bins = h2_vals
    n_bins_h2 = 28

    def compute_ratio(w):
        return _compute_ratio_fast(w, diff_bins, h0_bins, h1_bins, h2_bins,
                                   n_bins_diff, n_bins_h0, n_bins_h1, n_bins_h2)

    def compute_ratio_soft(w, beta=50.0):
        return _soft_max_entropy_ratio(w, diff_bins, h0_bins, h1_bins, h2_bins,
                                       n_bins_diff, n_bins_h0, n_bins_h1, n_bins_h2, beta)

    # Generate diverse initial candidates
    candidates = []

    # Uniform
    candidates.append(np.ones((n, n), dtype=np.float64) / 100.0)

    # Anti-diagonals
    for k in range(19):
        w = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                w[i, j] = 1.0
        if w.sum() > 0:
            w /= w.sum()
            candidates.append(w)

    # Middle anti-diagonals
    w3 = np.zeros((n, n), dtype=np.float64)
    for k in [8, 9, 10]:
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                w3[i, j] = 1.0
    w3 /= w3.sum()
    candidates.append(w3)

    # Weighted anti-diagonals by x-y spread
    w4 = np.zeros((n, n), dtype=np.float64)
    for k in range(19):
        diag_cells = []
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                diag_cells.append(j - i)
        if len(diag_cells) > 0:
            unique_diffs = len(set(diag_cells))
            weight = unique_diffs / len(diag_cells)
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    w4[i, j] = weight
    w4 /= w4.sum()
    candidates.append(w4)

    # Concentrate x
    for x_target in [3, 4, 5, 6]:
        w5 = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            w5[i, x_target] = 1.0
        w5 /= w5.sum()
        candidates.append(w5)

    # Exponential weighting on anti-diagonals
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
        w6 = np.zeros((n, n), dtype=np.float64)
        for k in range(19):
            center_dist = abs(k - 9)
            weight = np.exp(-alpha * center_dist / 10.0)
            for i in range(n):
                j = k - i
                if 0 <= j < n:
                    w6[i, j] = weight
        w6 /= w6.sum()
        candidates.append(w6)

    # 2D Gaussian patterns
    for cx in [2, 4, 5, 6, 8]:
        for cy in [2, 4, 5, 6, 8]:
            w_g = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    w_g[i, j] = np.exp(-((i - cy)**2 + (j - cx)**2) / 8.0)
            if w_g.sum() > 1e-9:
                w_g /= w_g.sum()
                candidates.append(w_g)

    # Evaluate all candidates
    best_score = -1
    best_P = candidates[0]
    for c in candidates:
        sc = compute_ratio(c.flatten())
        if sc > best_score:
            best_score = sc
            best_P = c.copy()

    # Phase 1: Multi-cell local search with restarts
    np.random.seed(42)
    current_P = best_P.copy()
    current_score = best_score

    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 200
    phase1_end = start_time + 150

    while time.time() < phase1_end:
        if no_improve_count > max_no_improve:
            noise_scale = 0.2 + 0.1 * (no_improve_count // max_no_improve)
            noise = np.random.normal(0, noise_scale, (n, n))
            current_P = best_P + noise
            current_P = np.clip(current_P, 0, None)
            s = current_P.sum()
            if s > 1e-9:
                current_P /= s
            else:
                current_P = best_P.copy()
            current_score = compute_ratio(current_P.flatten())
            no_improve_count = 0
            step_size = 0.15
            max_no_improve = 100
            continue

        n_cells = np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.3, 0.25, 0.15, 0.1])
        indices = np.random.randint(0, n * n, n_cells)
        deltas = np.random.uniform(-step_size, step_size, n_cells)

        new_P = current_P.copy()
        new_P_flat = new_P.flatten()
        new_P_flat[indices] += deltas
        new_P_flat = np.clip(new_P_flat, 0, None)
        s = new_P_flat.sum()
        if s <= 1e-9:
            continue
        new_P_flat /= s
        new_P = new_P_flat.reshape(n, n)

        new_score = compute_ratio(new_P_flat)

        if new_score > current_score:
            current_P = new_P
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_P = current_P.copy()
            step_size *= 0.97
        else:
            no_improve_count += 1
            step_size = max(0.01, step_size * 1.03)

    # Phase 2: Gradient-based optimization with soft-max surrogate
    P_for_log = best_P.copy()
    P_for_log[P_for_log < 1e-12] = 1e-12
    log_P_init = np.log(P_for_log.flatten())

    def neg_soft_ratio_from_log(log_p_flat, beta=50.0):
        log_p = log_p_flat.reshape(n, n)
        log_p_max = np.max(log_p)
        p = np.exp(log_p - log_p_max)
        p_sum = p.sum()
        if p_sum <= 1e-15:
            return 1e10
        p = p / p_sum
        w = p.flatten()
        ratio = compute_ratio_soft(w, beta)
        return -ratio

    def neg_ratio_from_log(log_p_flat):
        log_p = log_p_flat.reshape(n, n)
        log_p_max = np.max(log_p)
        p = np.exp(log_p - log_p_max)
        p_sum = p.sum()
        if p_sum <= 1e-15:
            return 1e10
        p = p / p_sum
        w = p.flatten()
        ratio = compute_ratio(w)
        return -ratio

    phase2_end = start_time + timeout - safety_margin

    # Gradient optimization with soft-max surrogate
    gradient_starts = [log_P_init]
    for _ in range(8):
        perturbed = log_P_init + np.random.normal(0, 0.8, n * n)
        gradient_starts.append(perturbed)

    for c in candidates[:10]:
        c_log = c.copy()
        c_log[c_log < 1e-12] = 1e-12
        gradient_starts.append(np.log(c_log.flatten()))

    best_gradient_score = best_score
    best_gradient_P = best_P.copy()

    for log_start in gradient_starts:
        if time.time() > phase2_end - 90:
            break

        try:
            result = minimize(
                neg_soft_ratio_from_log,
                log_start,
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )

            log_p_opt = result.x.reshape(n, n)
            log_p_max = np.max(log_p_opt)
            p_opt = np.exp(log_p_opt - log_p_max)
            p_opt = p_opt / p_opt.sum()

            # Evaluate with exact ratio
            score_opt = compute_ratio(p_opt.flatten())
            if score_opt > best_gradient_score:
                best_gradient_score = score_opt
                best_gradient_P = p_opt.copy()
        except Exception:
            continue

    if best_gradient_score > best_score:
        best_score = best_gradient_score
        best_P = best_gradient_P.copy()

    # Phase 2b: Refine with exact ratio
    log_P_init2 = np.log(np.clip(best_P, 1e-12, None).flatten())
    for _ in range(3):
        if time.time() > phase2_end - 60:
            break
        try:
            result = minimize(
                neg_ratio_from_log,
                log_P_init2,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            log_p_opt = result.x.reshape(n, n)
            log_p_max = np.max(log_p_opt)
            p_opt = np.exp(log_p_opt - log_p_max)
            p_opt = p_opt / p_opt.sum()
            score_opt = compute_ratio(p_opt.flatten())
            if score_opt > best_score:
                best_score = score_opt
                best_P = p_opt.copy()
                log_P_init2 = np.log(np.clip(best_P, 1e-12, None).flatten())
        except Exception:
            break

    # Phase 3: Final high-precision polish
    current_P = best_P.copy()
    current_score = best_score
    step_size = 0.003
    no_improve_count = 0
    max_no_improve = 150

    while time.time() < phase2_end - 15:
        if no_improve_count > max_no_improve:
            break

        move_type = np.random.choice(['single', 'pair', 'diag'], p=[0.4, 0.35, 0.25])

        if move_type == 'single':
            i, j = np.random.randint(0, n, 2)
            delta = np.random.uniform(-step_size, step_size)
            new_P = current_P.copy()
            new_P[i, j] += delta
        elif move_type == 'pair':
            i1, j1 = np.random.randint(0, n, 2)
            i2, j2 = np.random.randint(0, n, 2)
            delta = np.random.uniform(-step_size, step_size)
            new_P = current_P.copy()
            new_P[i1, j1] += delta
            new_P[i2, j2] -= delta
        else:
            i, j = np.random.randint(0, n, 2)
            delta = np.random.uniform(-step_size, step_size)
            new_P = current_P.copy()
            k = i + j
            cells_on_diag = [(a, b) for a in range(n) for b in range(n) if a + b == k]
            if len(cells_on_diag) >= 2:
                other_idx = np.random.choice(len(cells_on_diag) - 1)
                if other_idx >= cells_on_diag.index((i, j)):
                    other_idx += 1
                oi, oj = cells_on_diag[other_idx]
                new_P[i, j] += delta
                new_P[oi, oj] -= delta

        new_P = np.clip(new_P, 0, None)
        s = new_P.sum()
        if s <= 1e-9:
            continue
        new_P = new_P / s

        new_score = compute_ratio(new_P.flatten())

        if new_score > current_score:
            current_P = new_P
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_P = current_P.copy()
            step_size *= 0.94
        else:
            no_improve_count += 1
            step_size = max(0.0003, step_size * 1.04)

    # Final validation and normalization
    result = np.clip(best_P, 0, None)
    s = result.sum()
    if s <= 1e-9:
        result = np.ones((n, n)) / 100.0
    else:
        result = result / s

    return result

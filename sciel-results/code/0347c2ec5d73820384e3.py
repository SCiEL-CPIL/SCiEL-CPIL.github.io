import numpy as np
import time

def simulate_aco(demands, capacity, H, seed, n_ants=20, n_iters=30):
    np.random.seed(seed)
    n = len(demands)
    tau = np.ones((n, n))
    best_overall = float('inf')
    
    for _ in range(n_iters):
        bins_counts = []
        for _ in range(n_ants):
            remaining = list(range(n))
            bins = 0
            current_load = 0
            last_item = -1
            
            while remaining:
                if last_item == -1:
                    idx = np.random.randint(len(remaining))
                    i = remaining[idx]
                    remaining.pop(idx)
                    current_load = demands[i]
                    last_item = i
                    bins += 1
                else:
                    feasible_indices = []
                    weights = []
                    for k, j in enumerate(remaining):
                        if current_load + demands[j] <= capacity:
                            feasible_indices.append(k)
                            w = tau[last_item, j] * H[last_item, j]
                            if w <= 0:
                                w = 1e-9
                            weights.append(w)
                    
                    if not feasible_indices:
                        current_load = 0
                        last_item = -1
                        continue
                    
                    total_w = sum(weights)
                    r = np.random.rand() * total_w
                    cumulative = 0
                    chosen_k = feasible_indices[0]
                    for k_idx in range(len(feasible_indices)):
                        cumulative += weights[k_idx]
                        if r <= cumulative:
                            chosen_k = feasible_indices[k_idx]
                            break
                    
                    j = remaining[chosen_k]
                    remaining.pop(chosen_k)
                    current_load += demands[j]
                    last_item = j
            
            bins_counts.append(bins)
        
        best_this_iter = min(bins_counts)
        if best_this_iter < best_overall:
            best_overall = best_this_iter
        
        tau *= 0.9
        tau += 1.0 / best_this_iter
    
    return best_overall


def simulate_aco_fast(demands, capacity, H, seed, n_ants=5, n_iters=10):
    """Faster simulation for candidate screening"""
    np.random.seed(seed)
    n = len(demands)
    tau = np.ones((n, n))
    best_overall = float('inf')
    
    for _ in range(n_iters):
        bins_counts = []
        for _ in range(n_ants):
            remaining = list(range(n))
            bins = 0
            current_load = 0
            last_item = -1
            
            while remaining:
                if last_item == -1:
                    idx = np.random.randint(len(remaining))
                    i = remaining[idx]
                    remaining.pop(idx)
                    current_load = demands[i]
                    last_item = i
                    bins += 1
                else:
                    feasible_indices = []
                    weights = []
                    for k, j in enumerate(remaining):
                        if current_load + demands[j] <= capacity:
                            feasible_indices.append(k)
                            w = tau[last_item, j] * H[last_item, j]
                            if w <= 0:
                                w = 1e-9
                            weights.append(w)
                    
                    if not feasible_indices:
                        current_load = 0
                        last_item = -1
                        continue
                    
                    total_w = sum(weights)
                    r = np.random.rand() * total_w
                    cumulative = 0
                    chosen_k = feasible_indices[0]
                    for k_idx in range(len(feasible_indices)):
                        cumulative += weights[k_idx]
                        if r <= cumulative:
                            chosen_k = feasible_indices[k_idx]
                            break
                    
                    j = remaining[chosen_k]
                    remaining.pop(chosen_k)
                    current_load += demands[j]
                    last_item = j
            
            bins_counts.append(bins)
        
        best_this_iter = min(bins_counts)
        if best_this_iter < best_overall:
            best_overall = best_this_iter
        
        tau *= 0.9
        tau += 1.0 / best_this_iter
    
    return best_overall


def heuristics(demands, capacity):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin

    n = len(demands)
    d = np.array(demands, dtype=np.float64)
    
    # Precompute pair sums and ratios
    pair_sum = d[:, None] + d[None, :]
    feasible_mask = pair_sum <= capacity
    ratio = pair_sum / capacity
    residual = capacity - pair_sum
    
    candidates = []
    
    # Candidate 0: Baseline (Ratio^1)
    h0 = np.where(feasible_mask, ratio, 1e-9)
    candidates.append(h0)
    
    # Candidate 1: Ratio^2
    h1 = np.where(feasible_mask, ratio**2, 1e-9)
    candidates.append(h1)
    
    # Candidate 2: Ratio^4
    h2 = np.where(feasible_mask, ratio**4, 1e-9)
    candidates.append(h2)
    
    # Candidate 3: Inverse Waste (1 / (capacity - pair_sum + eps))
    waste = capacity - pair_sum
    h3 = np.where(feasible_mask, 1.0 / (waste + 1.0), 1e-9)
    candidates.append(h3)
    
    # Candidate 4: Best Fit Style (1 / waste^2)
    h4 = np.where(feasible_mask, 1.0 / (waste**2 + 1.0), 1e-9)
    candidates.append(h4)
    
    # Candidate 5: Exp(-waste)
    h5 = np.where(feasible_mask, np.exp(-waste / 5.0), 1e-9)
    candidates.append(h5)
    
    # Residual spectrum parameters
    median_demand = np.median(d)
    q25 = np.percentile(d, 25)
    q75 = np.percentile(d, 75)
    sigma = (q75 - q25) / 2.0 + 1.0
    
    # Candidate 6: Residual spectrum - Gaussian around median
    ideal_residual = median_demand
    residual_score = np.exp(-((residual - ideal_residual)**2) / (2 * sigma**2))
    h6 = np.where(feasible_mask, residual_score, 1e-9)
    candidates.append(h6)
    
    # Candidate 7: Residual spectrum with stronger tight-fit preference
    skew_factor = 1.5
    residual_score_skew = np.exp(-((residual - ideal_residual * skew_factor)**2) / (2 * sigma**2))
    h7 = np.where(feasible_mask, residual_score_skew * (1.0 + 0.5 * residual_score), 1e-9)
    candidates.append(h7)
    
    # Candidate 8: Combined residual spectrum and inverse waste
    h8 = np.where(feasible_mask, 0.6 * residual_score + 0.4 * (1.0 / (waste + 1.0)), 1e-9)
    candidates.append(h8)
    
    # Candidate 9: Multi-scale residual spectrum
    h9 = np.zeros((n, n))
    for scale in [0.5, 1.0, 1.5, 2.0]:
        ideal_s = median_demand * scale
        sigma_s = (q75 - q25) / 2.0 * scale + 1.0
        score_s = np.exp(-((residual - ideal_s)**2) / (2 * sigma_s**2))
        h9 += score_s
    h9 = np.where(feasible_mask, h9 / 4.0, 1e-9)
    candidates.append(h9)
    
    # Candidate 10: Residual with penalty for very large residuals
    max_useful_residual = q75 * 1.5
    h10 = np.where(feasible_mask, residual_score * np.exp(-np.maximum(0, residual - max_useful_residual) / 10.0), 1e-9)
    candidates.append(h10)
    
    # === NEW: Complementarity Graph / Structural Fit Candidates ===
    # Reward pairs where residual is a near-integer multiple of the smallest item size
    min_demand = np.min(d)
    
    # For each pair, compute how close the residual is to a multiple of min_demand
    # residual_mod = residual mod min_demand
    # distance_to_multiple = min(residual_mod, min_demand - residual_mod)
    # Score is high when distance_to_multiple is small (residual aligns with item grid)
    
    residual_mod = np.mod(residual, min_demand)
    distance_to_multiple = np.minimum(residual_mod, min_demand - residual_mod)
    
    # Normalize distance: 0 means perfect alignment, 1 means worst case
    norm_dist = distance_to_multiple / (min_demand / 2.0)
    norm_dist = np.clip(norm_dist, 0, 1)
    
    # Structural fit score: high when residual is near integer multiple of min item
    # Use exponential decay based on normalized distance
    struct_score = np.exp(-norm_dist * 3.0)
    
    # Candidate 11: Pure structural fit
    h11 = np.where(feasible_mask, struct_score, 1e-9)
    candidates.append(h11)
    
    # Candidate 12: Structural fit blended with residual spectrum
    h12 = np.where(feasible_mask, 0.5 * struct_score + 0.5 * residual_score, 1e-9)
    candidates.append(h12)
    
    # Candidate 13: Structural fit with inverse waste
    h13 = np.where(feasible_mask, 0.4 * struct_score + 0.3 * residual_score + 0.3 * (1.0 / (waste + 1.0)), 1e-9)
    candidates.append(h13)
    
    # Candidate 14: Structural fit with stronger emphasis
    struct_score_strong = np.exp(-norm_dist * 5.0)
    h14 = np.where(feasible_mask, 0.6 * struct_score_strong + 0.4 * residual_score, 1e-9)
    candidates.append(h14)
    
    # Candidate 15: Structural fit with variance-based weighting
    # The "variance of remaining space" interpretation:
    # If residual is close to a multiple of min item, the remaining space is more "predictable"
    # Use a softer version that also considers the absolute residual size
    abs_penalty = np.exp(-residual / (capacity * 0.3))
    h15 = np.where(feasible_mask, struct_score * abs_penalty, 1e-9)
    candidates.append(h15)
    
    # Candidate 16: Structural fit with quantile-aware scoring
    # Weight structural fit by how well the residual matches the demand distribution
    frac_below_residual = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if feasible_mask[i, j]:
                frac_below_residual[i, j] = np.sum(d <= residual[i, j]) / n
    
    # Ideal fraction: 0.5 to 0.75
    ideal_frac = 0.625
    quantile_score = np.exp(-((frac_below_residual - ideal_frac)**2) / 0.05)
    h16 = np.where(feasible_mask, 0.5 * struct_score + 0.5 * quantile_score, 1e-9)
    candidates.append(h16)
    
    # FFD-based candidates
    sorted_indices = np.argsort(-d)
    bins = []
    bin_loads = []
    for idx in sorted_indices:
        placed = False
        for b in range(len(bins)):
            if bin_loads[b] + d[idx] <= capacity:
                bins[b].append(idx)
                bin_loads[b] += d[idx]
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_loads.append(d[idx])
    
    h_ffd = np.full((n, n), 1e-9)
    for b in bins:
        for i in b:
            for j in b:
                if i != j and feasible_mask[i, j]:
                    h_ffd[i, j] = 1.0
    
    # Candidate 17: FFD with structural fit modulation
    h17 = np.where(h_ffd > 0.5, 50.0 * struct_score + 10.0, np.where(feasible_mask, residual_score, 1e-9))
    candidates.append(h17)
    
    # Candidate 18: Pure FFD (strong)
    h18 = np.where(h_ffd > 0.5, 100.0, np.where(feasible_mask, ratio, 1e-9))
    candidates.append(h18)
    
    # Candidate 19: FFD with ratio scaling
    h19 = np.full((n, n), 1e-9)
    for b in bins:
        for i in b:
            for j in b:
                if i != j and feasible_mask[i, j]:
                    h19[i, j] = 50.0 + ratio[i, j]
    candidates.append(h19)
    
    # Candidate 20: Adaptive ideal residual
    conservative_ideal = (median_demand + q75) / 2.0
    sigma2 = (q75 - q25) / 3.0 + 1.0
    residual_score2 = np.exp(-((residual - conservative_ideal)**2) / (2 * sigma2**2))
    h20 = np.where(feasible_mask, residual_score2, 1e-9)
    candidates.append(h20)
    
    # Ensure all candidates have valid values
    for idx in range(len(candidates)):
        candidates[idx] = np.where(np.isfinite(candidates[idx]) & (candidates[idx] > 0), candidates[idx], 1e-9)
    
    best_score = float('inf')
    best_h = h0  # Fallback to baseline
    
    # Use multiple seeds for robustness
    seeds = [42, 123, 7, 999]
    
    # Phase 1: Quick screening with fast simulation (fewer ants/iterations)
    if time.time() > deadline - 120:
        # Very tight on time, just return best guess
        return h0
    
    quick_scores = []
    for idx, h in enumerate(candidates):
        if time.time() > deadline - 100:
            break
        score = simulate_aco_fast(d, capacity, h, seeds[0], n_ants=5, n_iters=10)
        quick_scores.append((idx, score))
    
    # Sort by quick score and take top candidates for detailed evaluation
    quick_scores.sort(key=lambda x: x[1])
    top_indices = [idx for idx, _ in quick_scores[:10]]
    
    # Phase 2: Detailed evaluation with full simulation and multiple seeds
    for idx in top_indices:
        h = candidates[idx]
        
        # Check time
        if time.time() > deadline - 30:
            break
        
        # Evaluate candidate with all seeds
        scores = []
        for seed in seeds:
            if time.time() > deadline - 20:
                break
            score = simulate_aco(d, capacity, h, seed, n_ants=20, n_iters=30)
            scores.append(score)
        
        if scores:
            avg_score = np.mean(scores)
            
            if avg_score < best_score:
                best_score = avg_score
                best_h = h
            
            # Early stopping if we hit near-optimal
            lower_bound = np.ceil(np.sum(d) / capacity)
            if best_score <= lower_bound + 1:
                break
    
    return best_h

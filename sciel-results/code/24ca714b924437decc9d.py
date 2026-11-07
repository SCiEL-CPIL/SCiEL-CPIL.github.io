import numpy as np
import time

def _run_aco_correct(demands, capacity, H, n_ants=20, n_iter=30, seed=0):
    """
    Correct ACO simulation for bin packing.
    Tracks the actual sum of items in the current bin.
    """
    rng = np.random.default_rng(seed)
    n = len(demands)
    W = np.asarray(demands, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    
    # Clamp non-positive to 1e-9
    H = np.where(H > 0, H, 1e-9)
    
    # Pheromone matrix
    tau = np.ones((n, n), dtype=np.float64)
    
    best_fewest = n
    iter_best_fewest = n
    
    for it in range(n_iter):
        iter_fewest = n
        for _ in range(n_ants):
            remaining = np.ones(n, dtype=bool)
            bins = 0
            cur = None
            bin_sum = 0.0
            
            while remaining.any():
                if cur is None:
                    # Open new bin logic is handled by the 'continue' below if no fit, 
                    # but here we start a bin.
                    # Actually, the protocol says: "opens a bin, places a random fitting item"
                    # So if cur is None, we are opening a bin.
                    fit = remaining & (W <= capacity)
                    if not fit.any():
                        # This shouldn't happen if W[i] <= capacity always
                        break 
                    idx = np.nonzero(fit)[0]
                    pick = idx[rng.integers(len(idx))]
                    remaining[pick] = False
                    cur = pick
                    bin_sum = W[pick]
                    bins += 1
                else:
                    # Check fit against current bin sum
                    rem_cap = capacity - bin_sum
                    fit = remaining & (W <= rem_cap)
                    
                    if not fit.any():
                        # Open new bin
                        bins += 1
                        cur = None
                        bin_sum = 0.0
                        # The loop will continue, cur is None, so it will place a random item
                        continue
                    
                    # Select item j
                    probs = tau[cur, :] * H[cur, :]
                    probs = np.where(fit, probs, 0.0)
                    total = probs.sum()
                    
                    if total <= 0 or not np.isfinite(total):
                        idx = np.nonzero(fit)[0]
                        pick = idx[rng.integers(len(idx))]
                    else:
                        p = probs / total
                        pick = rng.choice(n, p=p)
                    
                    remaining[pick] = False
                    bin_sum += W[pick]
                    cur = pick
            
        iter_fewest = min(iter_fewest, bins)
        best_fewest = min(best_fewest, iter_fewest)
        
        # Pheromone update
        tau = tau * 0.9 + 1.0 / iter_fewest
        
    return best_fewest

def heuristics(demands, capacity):
    start = time.time()
    deadline = start + 280  # Safety margin
    
    W = np.asarray(demands, dtype=np.float64)
    n = len(W)
    
    # Precompute pair sums and gaps
    pair_sum = W[:, None] + W[None, :]
    fit = pair_sum <= capacity
    
    # Candidate Heuristics
    cands = []
    
    # 1. Inverse Gap: High when pair fills bin tightly
    # gap = capacity - pair_sum
    # H = 1 / (gap + eps)
    with np.errstate(divide='ignore', invalid='ignore'):
        gap = capacity - pair_sum
        h1 = np.where(fit, 1.0 / (gap + 1e-9), 0.0)
    cands.append(h1)
    
    # 2. Inverse Gap Squared
    with np.errstate(divide='ignore', invalid='ignore'):
        h2 = np.where(fit, 1.0 / ((gap + 1e-9) ** 2), 0.0)
    cands.append(h2)
    
    # 3. Fill Ratio: pair_sum / capacity (prefer pairs that fill more of the bin)
    fill_ratio = np.where(fit, pair_sum / capacity, 0.0)
    cands.append(fill_ratio)
    
    # 4. Fill Ratio Squared (stronger preference for high fill)
    h4 = np.where(fit, fill_ratio ** 2, 0.0)
    cands.append(h4)
    
    # 5. Blend: inverse-gap * fill_ratio^2 (best-fit + smooth fill signal)
    with np.errstate(divide='ignore', invalid='ignore'):
        h5 = np.where(fit, (1.0 / (gap + 1.0)) * (fill_ratio ** 2), 0.0)
    cands.append(h5)
    
    # 6. Blend: inverse-gap + alpha * fill_ratio^2
    for alpha in [0.5, 1.0, 2.0]:
        with np.errstate(divide='ignore', invalid='ignore'):
            h_blend = np.where(fit, 1.0 / (gap + 1.0) + alpha * (fill_ratio ** 2), 0.0)
        cands.append(h_blend)
    
    # 7. Log Inverse Gap
    with np.errstate(divide='ignore', invalid='ignore'):
        h7 = np.where(fit, np.log1p(1.0 / (gap + 1e-9)), 0.0)
    cands.append(h7)
    
    # 8. Exponential Decay of Gap
    h8 = np.where(fit, np.exp(-gap / 10.0), 0.0)
    cands.append(h8)
    
    # 9. Inverse Gap cubed (very strong tight-fit preference)
    with np.errstate(divide='ignore', invalid='ignore'):
        h9 = np.where(fit, 1.0 / ((gap + 1e-9) ** 3), 0.0)
    cands.append(h9)
    
    # 10. Soft-min surrogate: prefer pairs where min(remaining_cap_after, next_item) is large
    #    i.e., the second item leaves room for more items
    #    H[i,j] = (capacity - W[i] - W[j])^0.5 for fits (prefer moderate gaps)
    h10 = np.where(fit, np.sqrt(gap + 1e-9), 0.0)
    cands.append(h10)
    
    # 11. Combined: (1/gap) * sqrt(fill_ratio)
    with np.errstate(divide='ignore', invalid='ignore'):
        h11 = np.where(fit, (1.0 / (gap + 1.0)) * np.sqrt(fill_ratio + 1e-9), 0.0)
    cands.append(h11)
    
    # 12. fill_ratio^3
    h12 = np.where(fit, fill_ratio ** 3, 0.0)
    cands.append(h12)
    
    # 13. Scarcity-based: rank j by inverse degree in compatibility graph
    #     For each item j, count how many items i can fit with j (i.e., W[i] + W[j] <= capacity)
    #     Higher scarcity (fewer partners) -> higher heuristic value
    #     H[i,j] = 1 / (degree[j] + 1) for fits
    degree = np.sum(fit, axis=0)  # number of items that can fit with each item j
    h13 = np.where(fit, 1.0 / (degree[None, :] + 1.0), 0.0)
    cands.append(h13)
    
    # 14. Combined: inverse_gap * scarcity
    with np.errstate(divide='ignore', invalid='ignore'):
        h14 = np.where(fit, (1.0 / (gap + 1.0)) * (1.0 / (degree[None, :] + 1.0)), 0.0)
    cands.append(h14)
    
    # 15. Size-based: prefer larger items (First Fit Decreasing principle)
    h15 = np.where(fit, W[None, :], 0.0)
    cands.append(h15)
    
    # 16. Combined: inverse_gap * size
    with np.errstate(divide='ignore', invalid='ignore'):
        h16 = np.where(fit, (1.0 / (gap + 1.0)) * W[None, :], 0.0)
    cands.append(h16)
    
    # 17. Combined: size * scarcity
    h17 = np.where(fit, W[None, :] * (1.0 / (degree[None, :] + 1.0)), 0.0)
    cands.append(h17)
    
    # 18. Combined: fill_ratio * scarcity
    h18 = np.where(fit, fill_ratio * (1.0 / (degree[None, :] + 1.0)), 0.0)
    cands.append(h18)
    
    # 19. Combined: inverse_gap * size * scarcity
    with np.errstate(divide='ignore', invalid='ignore'):
        h19 = np.where(fit, (1.0 / (gap + 1.0)) * W[None, :] * (1.0 / (degree[None, :] + 1.0)), 0.0)
    cands.append(h19)
    
    # 20. Combined: fill_ratio^2 * scarcity
    h20 = np.where(fit, (fill_ratio ** 2) * (1.0 / (degree[None, :] + 1.0)), 0.0)
    cands.append(h20)
    
    # Evaluate candidates
    best_H = cands[0]
    best_score = float('inf')
    
    # Use seeds for robustness; use 2 seeds for speed to evaluate more candidates
    eval_seeds = [0, 42]
    
    candidate_scores = []
    for idx, H in enumerate(cands):
        if time.time() > deadline - 15:
            break
            
        # Ensure H is valid
        H_valid = np.where(H > 0, H, 1e-9)
        
        avg_score = 0
        valid = True
        for seed in eval_seeds:
            try:
                score = _run_aco_correct(W, capacity, H_valid, n_ants=20, n_iter=30, seed=seed)
                avg_score += score
            except Exception:
                valid = False
                break
        
        if valid:
            avg_score /= len(eval_seeds)
            candidate_scores.append((avg_score, idx, H_valid))
            if avg_score < best_score:
                best_score = avg_score
                best_H = H_valid
    
    # If we have time, re-evaluate top candidates with more seeds for robustness
    if time.time() < deadline - 30 and len(candidate_scores) > 0:
        candidate_scores.sort(key=lambda x: x[0])
        top_k = min(5, len(candidate_scores))
        for avg_score, idx, H_valid in candidate_scores[:top_k]:
            if time.time() > deadline - 15:
                break
            # Re-evaluate with 3 seeds
            avg3 = 0
            valid = True
            for seed in [0, 42, 123]:
                try:
                    score = _run_aco_correct(W, capacity, H_valid, n_ants=20, n_iter=30, seed=seed)
                    avg3 += score
                except Exception:
                    valid = False
                    break
            if valid:
                avg3 /= 3.0
                if avg3 < best_score:
                    best_score = avg3
                    best_H = H_valid

    # Final safety checks
    best_H = np.asarray(best_H, dtype=np.float64)
    if best_H.shape != (n, n):
        best_H = np.ones((n, n)) * 1e-9
    if not np.all(np.isfinite(best_H)):
        best_H = np.ones((n, n)) * 1e-9
    best_H = np.where(best_H > 0, best_H, 1e-9)
    
    return best_H

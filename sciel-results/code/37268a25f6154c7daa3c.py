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
                    # Open new bin
                    fit = remaining & (W <= capacity)
                    if not fit.any():
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
    deadline = start + 290  # Safety margin before 300s limit
    
    W = np.asarray(demands, dtype=np.float64)
    n = len(W)
    
    # Precompute pair sums and gaps
    pair_sum = W[:, None] + W[None, :]
    fit = pair_sum <= capacity
    
    # Global max demand
    max_demand = np.max(W)
    
    # Compute inverse degree (scarcity) for each item
    # deg[j] = number of items k != j such that W[j] + W[k] <= capacity
    # We can compute this from the fit matrix
    # fit is n x n. fit[i, j] is True if i and j fit.
    # deg[j] = sum(fit[:, j]) - 1 (exclude self if fit[j,j] is true, but W[j]+W[j] might be > cap)
    # Actually, fit[j, j] is W[j]*2 <= capacity.
    # Let's just count valid partners.
    deg = np.sum(fit, axis=0)
    # If fit[j, j] is True, it counts itself. We want partners k != j.
    # So deg[j] -= 1 if fit[j, j] else 0
    self_fit = np.diag(fit)
    deg = deg - self_fit.astype(np.float64)
    
    # Inverse degree: higher is scarcer
    inv_deg = 1.0 / (deg + 1.0)
    
    # Candidate Heuristics
    cands = []
    cand_names = []
    
    # 1. Inverse Gap: High when pair fills bin tightly
    with np.errstate(divide='ignore', invalid='ignore'):
        gap = capacity - pair_sum
        h1 = np.where(fit, 1.0 / (gap + 1e-9), 0.0)
    cands.append(h1)
    cand_names.append("inv_gap")
    
    # 2. Inverse Gap Squared
    with np.errstate(divide='ignore', invalid='ignore'):
        h2 = np.where(fit, 1.0 / ((gap + 1e-9) ** 2), 0.0)
    cands.append(h2)
    cand_names.append("inv_gap_sq")
    
    # 3. Structural Bottleneck Signal (Deadness)
    dead = np.where(fit, np.maximum(0, max_demand - gap), 0)
    h9 = np.where(fit, 1.0 / (1.0 + dead), 0.0)
    cands.append(h9)
    cand_names.append("deadness")
    
    # 4. Blend: Inv Gap * Inv Degree
    # H[i, j] = (1/gap(i,j)) * inv_deg[j]
    h10 = h1 * inv_deg[None, :]
    cands.append(h10)
    cand_names.append("inv_gap_inv_deg")
    
    # 5. Blend: Inv Gap Squared * Inv Degree
    h11 = h2 * inv_deg[None, :]
    cands.append(h11)
    cand_names.append("inv_gap_sq_inv_deg")
    
    # 6. Blend: Inv Gap + Inv Degree
    h12 = h1 + inv_deg[None, :]
    cands.append(h12)
    cand_names.append("inv_gap_plus_inv_deg")
    
    # 7. Pure Inv Degree (Row-independent)
    h13 = np.tile(inv_deg[None, :], (n, 1))
    # Apply fit mask? If not fit, it shouldn't be chosen anyway, but H must be positive.
    # The ACO loop masks with `fit`. So H can be anything for non-fit, but must be > 0.
    # So we can just use the raw inv_deg.
    cands.append(h13)
    cand_names.append("pure_inv_deg")
    
    # 8. Sharpened Gap with Deadness
    h14 = h1 * h9
    cands.append(h14)
    cand_names.append("inv_gap_deadness")
    
    # Evaluate candidates
    best_H = cands[0]
    best_score = float('inf')
    
    # Use a few seeds to ensure robustness
    eval_seeds = [0, 42, 123]
    
    for idx, H in enumerate(cands):
        if time.time() > deadline - 10:
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
            if avg_score < best_score:
                best_score = avg_score
                best_H = H_valid
        
    # Final safety checks
    best_H = np.asarray(best_H, dtype=np.float64)
    if best_H.shape != (n, n):
        best_H = np.ones((n, n)) * 1e-9
    if not np.all(np.isfinite(best_H)):
        best_H = np.ones((n, n)) * 1e-9
    best_H = np.where(best_H > 0, best_H, 1e-9)
    
    return best_H

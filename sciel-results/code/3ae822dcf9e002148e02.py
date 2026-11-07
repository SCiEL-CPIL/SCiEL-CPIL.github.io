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
    deadline = start + 290  # Safety margin before 300s limit
    
    W = np.asarray(demands, dtype=np.float64)
    n = len(W)
    
    # Precompute pair sums and gaps
    pair_sum = W[:, None] + W[None, :]
    fit = pair_sum <= capacity
    
    # Global max demand
    max_demand = np.max(W)
    
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
    
    # 3. Pair Sum (Prefer larger pairs? Often bad, but let's see)
    h3 = np.where(fit, pair_sum, 0.0)
    cands.append(h3)
    
    # 4. Complement (Prefer smaller pairs? Leaves more room)
    h4 = np.where(fit, gap, 0.0)
    cands.append(h4)
    
    # 5. Product of Sizes
    h5 = np.where(fit, W[:, None] * W[None, :], 0.0)
    cands.append(h5)
    
    # 6. Ratio of Sizes (Balance?)
    with np.errstate(divide='ignore', invalid='ignore'):
        h6 = np.where(fit, np.minimum(W[:, None], W[None, :]) / (np.maximum(W[:, None], W[None, :]) + 1e-9), 0.0)
    cands.append(h6)
        
    # 7. Log Inverse Gap
    with np.errstate(divide='ignore', invalid='ignore'):
        h7 = np.where(fit, np.log1p(1.0 / (gap + 1e-9)), 0.0)
    cands.append(h7)
    
    # 8. Exponential Decay of Gap
    h8 = np.where(fit, np.exp(-gap / 10.0), 0.0)
    cands.append(h8)
    
    # 9. Structural Bottleneck Signal
    # After placing i and j, residual = capacity - (W[i] + W[j])
    # If residual < max_demand, it's a dead end for the largest item.
    # We want to penalize this. So H should be low if residual is small but non-zero?
    # Or H should be high if residual is large enough to fit max_demand?
    # Let's try: H = 1 if residual >= max_demand, else 1/residual
    # Actually, let's try a smooth penalty.
    # residual = capacity - pair_sum
    # If residual < max_demand, we lose the option of placing the biggest item.
    # Let's define a "deadness" score.
    dead = np.where(fit, np.maximum(0, max_demand - gap), 0)
    # High deadness is bad. So H = 1 / (1 + dead)
    h9 = np.where(fit, 1.0 / (1.0 + dead), 0.0)
    cands.append(h9)
    
    # 10. Blend of Inverse Gap and Structural
    h10 = h1 * h9
    cands.append(h10)

    # 11. Gap Squared Penalty (Prefer large gaps? No, usually tight is better)
    # Let's try a mix of tightness and feasibility
    # H = (1/gap) * (gap/max_demand) = 1/max_demand (constant) -> bad
    # H = (gap/max_demand) / (1/gap) = gap^2 / max_demand
    h11 = np.where(fit, (gap / max_demand) ** 2, 0.0)
    cands.append(h11)
    
    # 12. Soft Min of Gap and Max Demand
    # We want gap to be either 0 or >= max_demand.
    # Penalty for gap in (0, max_demand)
    # H = exp(- (gap * (max_demand - gap)) / (max_demand * max_demand))
    # This peaks at gap=0 and gap=max_demand
    h12 = np.where(fit, np.exp(- (gap * (max_demand - gap)) / (max_demand * max_demand)), 0.0)
    cands.append(h12)

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
        
    # If we have time, try combinations of top 2
    if time.time() < deadline - 20:
        # Sort candidates by performance? We didn't store them. Let's just mix top few.
        # For simplicity, stick with the best single candidate unless we have a very strong reason.
        pass

    # Final safety checks
    best_H = np.asarray(best_H, dtype=np.float64)
    if best_H.shape != (n, n):
        best_H = np.ones((n, n)) * 1e-9
    if not np.all(np.isfinite(best_H)):
        best_H = np.ones((n, n)) * 1e-9
    best_H = np.where(best_H > 0, best_H, 1e-9)
    
    return best_H

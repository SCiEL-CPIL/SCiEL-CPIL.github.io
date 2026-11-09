"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _simulate_aco(prizes, weights, heuristic, n_ants=20, n_iters=30, capacity=25):
    """
    Simulate the ACO process and return the best total prize found.
    prizes: (5, 100)
    weights: (5, 100)
    heuristic: (100,) - one score per item
    """
    n_constraints, n_items = prizes.shape
    # Ensure heuristic is valid
    h = np.asarray(heuristic, dtype=np.float64).ravel()
    if h.shape != (n_items,):
        raise ValueError("wrong shape")
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    pheromone = np.ones(n_items, dtype=np.float64)
    best_total_prize = 0.0
    
    for it in range(n_iters):
        iter_best = 0.0
        for _ in range(n_ants):
            remaining_cap = np.full(n_constraints, capacity, dtype=np.float64)
            selected = []
            total_prize = 0.0
            
            # Greedy probabilistic selection
            # We pick items one by one until none fit.
            # Probability proportional to pheromone[item] * heuristic[item]
            # among items that still fit.
            
            available = np.arange(n_items)
            while len(available) > 0:
                # Filter available items that fit
                fits_mask = np.all(weights[:, available] <= remaining_cap, axis=0)
                if not np.any(fits_mask):
                    break
                
                avail_indices = available[fits_mask]
                probs = pheromone[avail_indices] * h[avail_indices]
                total_prob = probs.sum()
                if total_prob <= 0:
                    break
                probs = probs / total_prob
                
                # Sample one item
                idx = np.random.choice(avail_indices, p=probs)
                
                # Add it
                selected.append(idx)
                total_prize += prizes[:, idx].sum()
                for c in range(n_constraints):
                    remaining_cap[c] -= weights[c, idx]
                
                # Remove from available
                available = available[available != idx]
            
        if total_prize > iter_best:
            iter_best = total_prize
        
        if iter_best > best_total_prize:
            best_total_prize = iter_best
        
        # Update pheromone
        pheromone *= 0.9
        pheromone += iter_best / 10.0
    
    return best_total_prize


def _evaluate_heuristic(prizes, weights, heuristic):
    """
    Run ACO simulation with the given heuristic and return the best prize found.
    Uses a fixed seed for reproducibility within the heuristic selection process.
    """
    np.random.seed(42)  # Fixed seed for heuristic evaluation
    try:
        return _simulate_aco(prizes, weights, heuristic)
    except Exception:
        return 0.0


def heuristics(prizes, weights):
    """
    Returns an array of one attractiveness score per item.
    Uses a diverse set of heuristic candidates and selects the best one
    via a local ACO simulation with a fixed seed.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s limit
    
    n_constraints, n_items = prizes.shape
    
    # Generate diverse heuristic candidates
    candidates = []
    
    # 1. Original: prize / (sum of weights)
    w_sum = weights.sum(axis=0)
    candidates.append(prizes / (w_sum + 1e-9))
    
    # 2. Prize / (max weight across constraints)
    w_max = weights.max(axis=0)
    candidates.append(prizes / (w_max + 1e-9))
    
    # 3. Prize / (min weight across constraints)
    w_min = weights.min(axis=0)
    candidates.append(prizes / (w_min + 1e-9))
    
    # 4. Prize / (mean weight)
    w_mean = weights.mean(axis=0)
    candidates.append(prizes / (w_mean + 1e-9))
    
    # 5. Sum of prizes / (sum of weights) - already similar to 1 but let's try prize sum per item
    # Actually prizes is (5,100), so prizes[:,i] is the prize vector for item i.
    # The total prize for item i is prizes[:,i].sum()
    prize_total = prizes.sum(axis=0)
    candidates.append(prize_total / (w_sum + 1e-9))
    
    # 6. Geometric mean of (prize_c / weight_c) for each constraint
    # prize_c / weight_c for each c, then geometric mean
    ratios = prizes / (weights + 1e-9)
    log_ratios = np.log(ratios)
    geo_mean_ratio = np.exp(log_ratios.mean(axis=0))
    candidates.append(geo_mean_ratio)
    
    # 7. Min of (prize_c / weight_c) across constraints - bottleneck heuristic
    min_ratio = ratios.min(axis=0)
    candidates.append(min_ratio)
    
    # 8. Max of (prize_c / weight_c) across constraints
    max_ratio = ratios.max(axis=0)
    candidates.append(max_ratio)
    
    # 9. Prize / (weighted sum with constraint tightness)
    # Constraint tightness: how much of the capacity is used if we take all items
    # This is a rough measure: sum of weights / capacity
    tightness = weights.sum(axis=1) / 25.0  # (5,)
    w_tight = weights.T @ tightness  # (100,)
    candidates.append(prize_total / (w_tight + 1e-9))
    
    # 10. Random-ish but structured: prize / (sum of weights squared)
    w_sq = (weights ** 2).sum(axis=0)
    candidates.append(prize_total / (w_sq + 1e-9))
    
    # Evaluate each candidate and pick the best
    best_score = -1
    best_heuristic = candidates[0]
    
    for i, h in enumerate(candidates):
        if time.time() > deadline - 10:
            break
        score = _evaluate_heuristic(prizes, weights, h)
        if score > best_score:
            best_score = score
            best_heuristic = h
    
    # Ensure the returned heuristic is valid
    result = np.asarray(best_heuristic, dtype=np.float64).ravel()
    result = np.where(np.isfinite(result) & (result > 0), result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END

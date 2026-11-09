import numpy as np
import time

def _ffd_assignment(demands, capacity):
    """
    Perform First-Fit Decreasing bin packing.
    Returns an array of bin indices for each item.
    """
    n = len(demands)
    sorted_indices = np.argsort(-demands)
    
    bins = []
    bin_loads = []
    item_bin = np.full(n, -1, dtype=int)
    
    for idx in sorted_indices:
        size = demands[idx]
        placed = False
        for b_idx in range(len(bins)):
            if bin_loads[b_idx] + size <= capacity:
                bins[b_idx].append(idx)
                bin_loads[b_idx] += size
                item_bin[idx] = b_idx
                placed = True
                break
        if not placed:
            bins.append([idx])
            bin_loads.append(size)
            item_bin[idx] = len(bins) - 1
            
    return item_bin

def _build_ffd_prior(demands, capacity):
    """
    Build a heuristic matrix based on FFD bin assignments.
    High values for pairs in the same bin.
    """
    n = len(demands)
    item_bin = _ffd_assignment(demands, capacity)
    
    same_bin = (item_bin[:, None] == item_bin[None, :]).astype(float)
    np.fill_diagonal(same_bin, 0)
    
    return same_bin

def _build_complementarity_prior(demands, capacity):
    """
    Build a heuristic matrix rewarding pairs where the remaining capacity
    after placing i and j is a near-integer multiple of the smallest item size.
    This encourages structural fit patterns.
    """
    n = len(demands)
    min_size = np.min(demands)
    
    pair_sum = demands[:, None] + demands[None, :]
    remaining = capacity - pair_sum
    
    # Only consider pairs that fit
    fits = (pair_sum <= capacity)
    
    # Calculate how close remaining is to an integer multiple of min_size
    # remaining = k * min_size + r, where 0 <= r < min_size
    # We want r to be close to 0 (or close to min_size, which means remaining is close to (k+1)*min_size)
    # Actually, if remaining is close to k * min_size, it means we can fit exactly k more smallest items.
    # But we want the "waste" to be minimal in a structural sense.
    # Let's define distance to nearest multiple of min_size.
    
    multiples = np.round(remaining / min_size) * min_size
    dist = np.abs(remaining - multiples)
    
    # Heuristic: 1 / (dist + epsilon) for fitting pairs
    eps = 1e-9
    comp = np.where(fits, 1.0 / (dist + eps), 1e-9)
    
    # Also, we should only reward if the remaining space is actually "useful"
    # i.e., not too small to fit anything else, but the prompt says "placing j right after i".
    # The "structural fit" idea is that if remaining is a multiple of min_size, it might be fillable.
    
    np.fill_diagonal(comp, 0)
    
    return comp

def _fast_aco_eval(demands, capacity, heuristic, n_ants=5, n_iters=5, seed=42):
    """
    Fast evaluation of a heuristic matrix using a simplified ACO.
    Returns the best number of bins found.
    """
    n = len(demands)
    rng = np.random.default_rng(seed)
    pheromone = np.ones((n, n))
    best_bins = n
    
    # Precompute fit mask for speed
    pair_sum = demands[:, None] + demands[None, :]
    fits_mask = (pair_sum <= capacity)
    
    for _ in range(n_iters):
        iteration_best = n
        for _ in range(n_ants):
            placed = np.zeros(n, dtype=bool)
            bins = 0
            current = -1
            
            while not placed.all():
                if current == -1:
                    # Open new bin: pick random fitting item
                    available = np.where(~placed)[0]
                    if len(available) == 0: break
                    current = available[rng.integers(len(available))]
                    placed[current] = True
                    bins += 1
                else:
                    # Pick next item
                    available = np.where(~placed)[0]
                    if len(available) == 0: break
                    
                    # Filter by fit
                    fit_indices = available[fits_mask[current, available]]
                    if len(fit_indices) == 0:
                        current = -1
                        continue
                    
                    # Calculate probabilities
                    # pheromone[current, j] * heuristic[current, j]
                    probs = pheromone[current, fit_indices] * heuristic[current, fit_indices]
                    probs = np.where(probs > 0, probs, 1e-9)
                    
                    total = probs.sum()
                    if total <= 0:
                        current = fit_indices[rng.integers(len(fit_indices))]
                    else:
                        r = rng.random()
                        cumsum = np.cumsum(probs) / total
                        idx = np.searchsorted(cumsum, r)
                        if idx >= len(fit_indices):
                            idx = len(fit_indices) - 1
                        current = fit_indices[idx]
                    placed[current] = True
                    
            iteration_best = min(iteration_best, bins)
            
        # Update pheromone
        pheromone *= 0.9
        if iteration_best > 0:
            pheromone += 1.0 / iteration_best
            
        best_bins = min(best_bins, iteration_best)
        
    return best_bins

def heuristics(demands, capacity):
    start = time.time()
    deadline = 300.0
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # 1. FFD Prior
    ffd_prior = _build_ffd_prior(demands, capacity)
    
    # 2. Tightness Prior
    pair_sum = demands[:, None] + demands[None, :]
    fits = (pair_sum <= capacity)
    remaining = capacity - pair_sum
    eps = 1e-9
    tightness = np.where(fits, 1.0 / (remaining + eps), 1e-9)
    np.fill_diagonal(tightness, 0)
    
    # 3. Complementarity Prior (Structural Fit)
    comp_prior = _build_complementarity_prior(demands, capacity)
    
    # Normalize priors to have max value 1.0 for fair combination
    def normalize(h):
        h_max = np.max(h)
        if h_max > 0:
            h_norm = h / h_max
        else:
            h_norm = h
        h_norm = np.where(np.isfinite(h_norm), h_norm, 1e-9)
        return h_norm
    
    ffd_norm = normalize(ffd_prior)
    tight_norm = normalize(tightness)
    comp_norm = normalize(comp_prior)
    
    candidates = []
    
    # Candidate 1: FFD only
    candidates.append(ffd_norm)
    
    # Candidate 2: Tightness only
    candidates.append(tight_norm)
    
    # Candidate 3: Complementarity only
    candidates.append(comp_norm)
    
    # Candidate 4: FFD * Tightness
    candidates.append(ffd_norm * tight_norm)
    
    # Candidate 5: FFD * Complementarity
    candidates.append(ffd_norm * comp_norm)
    
    # Candidate 6: Tightness * Complementarity
    candidates.append(tight_norm * comp_norm)
    
    # Candidate 7: FFD * Tightness * Complementarity
    candidates.append(ffd_norm * tight_norm * comp_norm)
    
    # Candidate 8: Weighted Sum (0.5 FFD + 0.3 Tight + 0.2 Comp)
    candidates.append(0.5 * ffd_norm + 0.3 * tight_norm + 0.2 * comp_norm)
    
    # Candidate 9: Weighted Sum (0.4 FFD + 0.4 Tight + 0.2 Comp)
    candidates.append(0.4 * ffd_norm + 0.4 * tight_norm + 0.2 * comp_norm)
    
    # Candidate 10: Weighted Sum (0.3 FFD + 0.3 Tight + 0.4 Comp)
    candidates.append(0.3 * ffd_norm + 0.3 * tight_norm + 0.4 * comp_norm)
    
    # Normalize candidates to have max value 1.0
    normalized_candidates = [normalize(h) for h in candidates]
    
    # Evaluate candidates
    best_heuristic = normalized_candidates[0]
    best_score = float('inf')
    
    for idx, h in enumerate(normalized_candidates):
        if time.time() > deadline - 60:
            break
        score = _fast_aco_eval(demands, capacity, h, n_ants=5, n_iters=5, seed=42)
        if score < best_score:
            best_score = score
            best_heuristic = h
            
    # If we have time, try a few more with different seeds to be robust
    if time.time() < deadline - 30:
        # Re-evaluate the top 3 candidates with different seeds
        # This is a bit complex, so let's just stick with the best found
        # or do a simple local search around the best combination weights
        pass

    # Final check: ensure the returned matrix is valid (finite, correct shape)
    if not np.all(np.isfinite(best_heuristic)):
        best_heuristic = np.where(np.isfinite(best_heuristic), best_heuristic, 1e-9)
        
    return best_heuristic

import numpy as np
import time

def _bfd_assignment(demands, capacity):
    n = len(demands)
    sorted_indices = np.argsort(-demands)
    
    bins = [] # List of lists of indices
    bin_loads = [] # List of current loads
    item_bin = np.full(n, -1, dtype=int)
    
    for idx in sorted_indices:
        size = demands[idx]
        best_bin = -1
        best_remaining = float('inf')
        
        for b_idx in range(len(bins)):
            remaining = capacity - bin_loads[b_idx]
            if remaining >= size:
                if remaining < best_remaining:
                    best_remaining = remaining
                    best_bin = b_idx
        
        if best_bin != -1:
            bins[best_bin].append(idx)
            bin_loads[best_bin] += size
            item_bin[idx] = best_bin
        else:
            bins.append([idx])
            bin_loads.append(size)
            item_bin[idx] = len(bins) - 1
            
    return item_bin

def _ffd_assignment(demands, capacity):
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

def _fast_aco_eval(demands, capacity, heuristic, n_ants=10, n_iters=10, seed=42):
    n = len(demands)
    rng = np.random.default_rng(seed)
    pheromone = np.ones((n, n))
    best_bins = n
    
    # Precompute fit mask
    pair_sum = demands[:, None] + demands[None, :]
    fits_mask = (pair_sum <= capacity)
    
    # Ensure heuristic is positive
    h = np.maximum(heuristic, 1e-9)
    
    for _ in range(n_iters):
        iteration_best = n
        for _ in range(n_ants):
            placed = np.zeros(n, dtype=bool)
            bins = 0
            current = -1
            
            while not placed.all():
                if current == -1:
                    # Open new bin
                    available = np.where(~placed)[0]
                    if len(available) == 0: break
                    current = available[rng.integers(len(available))]
                    placed[current] = True
                    bins += 1
                else:
                    available = np.where(~placed)[0]
                    if len(available) == 0: break
                    
                    # Filter by fit
                    fit_indices = available[fits_mask[current, available]]
                    if len(fit_indices) == 0:
                        current = -1
                        continue
                    
                    # Calculate probabilities
                    probs = pheromone[current, fit_indices] * h[current, fit_indices]
                    total = probs.sum()
                    if total <= 0:
                        current = fit_indices[rng.integers(len(fit_indices))]
                    else:
                        # Normalize and sample
                        probs_norm = probs / total
                        r = rng.random()
                        cumsum = np.cumsum(probs_norm)
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
    deadline = 280.0 # Safety margin
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # 1. Compute Structural Priors
    item_bin_bfd = _bfd_assignment(demands, capacity)
    item_bin_ffd = _ffd_assignment(demands, capacity)
    
    same_bin_bfd = (item_bin_bfd[:, None] == item_bin_bfd[None, :]).astype(float)
    same_bin_ffd = (item_bin_ffd[:, None] == item_bin_ffd[None, :]).astype(float)
    
    np.fill_diagonal(same_bin_bfd, 0)
    np.fill_diagonal(same_bin_ffd, 0)
    
    # 2. Compute Tightness Prior
    pair_sum = demands[:, None] + demands[None, :]
    fits = (pair_sum <= capacity)
    remaining = capacity - pair_sum
    eps = 1e-9
    # High tightness for small remaining space
    tightness = np.where(fits, 1.0 / (remaining + eps), 0.0)
    np.fill_diagonal(tightness, 0)
    
    # 3. Generate Candidates
    candidates = []
    names = []
    
    # C1: BFD Structure * Tightness
    h1 = same_bin_bfd * tightness
    candidates.append(h1); names.append("BFD_Tight")
    
    # C2: FFD Structure * Tightness
    h2 = same_bin_ffd * tightness
    candidates.append(h2); names.append("FFD_Tight")
    
    # C3: (BFD + FFD) * Tightness
    h3 = (same_bin_bfd + same_bin_ffd) * tightness
    candidates.append(h3); names.append("Mix_Tight")
    
    # C4: Pure Tightness
    h4 = tightness.copy()
    candidates.append(h4); names.append("Tight")
    
    # C5: Pure BFD Structure
    h5 = same_bin_bfd.copy()
    candidates.append(h5); names.append("BFD")
    
    # C6: Pure FFD Structure
    h6 = same_bin_ffd.copy()
    candidates.append(h6); names.append("FFD")
    
    # C7: BFD Structure + Tightness (Additive)
    h7 = same_bin_bfd + tightness
    candidates.append(h7); names.append("BFD_Add_Tight")
    
    # C8: Soft BFD (1 if same bin, 0.1 if not) * Tightness
    soft_bfd = np.where(same_bin_bfd > 0, 1.0, 0.1)
    h8 = soft_bfd * tightness
    candidates.append(h8); names.append("Soft_BFD_Tight")
    
    # C9: BFD Structure * (1 + Tightness)
    h9 = same_bin_bfd * (1.0 + tightness)
    candidates.append(h9); names.append("BFD_1_Tight")
    
    # C10: FFD Structure * (1 + Tightness)
    h10 = same_bin_ffd * (1.0 + tightness)
    candidates.append(h10); names.append("FFD_1_Tight")
    
    # Normalize candidates to max 1.0 for fair initial comparison?
    # The ACO scales by pheromone (starts at 1).
    # If H is very large, it dominates.
    # Let's normalize each to [0, 1] range for the search phase.
    normalized_candidates = []
    for h in candidates:
        h_max = np.max(h)
        if h_max > 0:
            h_norm = h / h_max
        else:
            h_norm = h
        h_norm = np.where(np.isfinite(h_norm), h_norm, 1e-9)
        normalized_candidates.append(h_norm)
        
    best_heuristic = normalized_candidates[0]
    best_score = float('inf')
    
    for idx, h in enumerate(normalized_candidates):
        if time.time() > deadline:
            break
            
        score = _fast_aco_eval(demands, capacity, h, n_ants=10, n_iters=10, seed=42)
        if score < best_score:
            best_score = score
            best_heuristic = h
            
    # Final sanity check
    if not np.all(np.isfinite(best_heuristic)):
        best_heuristic = np.where(np.isfinite(best_heuristic), best_heuristic, 1e-9)
        
    # Ensure positive
    best_heuristic = np.maximum(best_heuristic, 1e-9)
    
    return best_heuristic

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _simulate_aco(demands, capacity, H, n_ants=20, n_iters=30, seed=42):
    """
    Run ACO with given heuristic matrix, return fewest bins found.
    Optimized for speed using precomputed compatibility masks where possible, 
    though the dynamic nature of 'used' items prevents full precomputation of probabilities.
    """
    rng = np.random.RandomState(seed)
    n = len(demands)
    W = np.array(demands, dtype=float)
    cap = float(capacity)
    
    # Pheromone matrix
    tau = np.ones((n, n))
    
    best_bins = float('inf')
    
    # Precompute which items can follow which items (static constraint)
    # fit_mask[i, j] is True if W[i] + W[j] <= capacity
    # Note: In the problem, we place j after i. So we need W[i] + W[j] <= capacity?
    # Actually, the bin contains items. If we place j after i, the bin already has i.
    # So we need W[i] + W[j] <= capacity.
    # However, the bin might contain other items too?
    # The problem states: "placing item j right after item i in the same bin".
    # "An ant opens a bin, places a random fitting item, then repeatedly picks a fitting item..."
    # This implies the bin accumulates items. The "current" item is the last one placed.
    # The constraint is that the *total* sum of items in the bin <= capacity.
    # The heuristic H[current, j] is used.
    # The fitting condition is: sum(bin_items) + W[j] <= capacity.
    
    for it in range(n_iters):
        iteration_best = float('inf')
        
        for ant in range(n_ants):
            bins = 0
            used = np.zeros(n, dtype=bool)
            current = None
            current_sum = 0.0
            
            while not used.all():
                if current is None:
                    # Pick random fitting item to start bin
                    # Any item fits in an empty bin since W[i] <= 99 and cap=150
                    fitting = np.where(~used)[0]
                    if len(fitting) == 0:
                        break
                    current = fitting[rng.randint(len(fitting))]
                    used[current] = True
                    current_sum = W[current]
                    bins += 1
                else:
                    # Find items that fit after current
                    # Remaining capacity in this bin: capacity - current_sum
                    remaining_cap = cap - current_sum
                    # Items j such that W[j] <= remaining_cap and not used
                    fitting = np.where((~used) & (W <= remaining_cap))[0]
                    
                    if len(fitting) == 0:
                        # Open new bin
                        current = None
                        current_sum = 0.0
                        continue
                    
                    # Compute probabilities
                    # P(j) proportional to tau[current, j] * H[current, j]
                    probs = tau[current, fitting] * H[current, fitting]
                    total = probs.sum()
                    if total <= 0 or not np.isfinite(total):
                        probs = np.ones(len(fitting))
                        total = probs.sum()
                    
                    probs /= total
                    next_idx = rng.choice(fitting, p=probs)
                    used[next_idx] = True
                    current_sum += W[next_idx]
                    current = next_idx
            
            if bins < iteration_best:
                iteration_best = bins
        
        if iteration_best < best_bins:
            best_bins = iteration_best
        
        # Update pheromone
        tau *= 0.9
        tau += 1.0 / iteration_best
    
    return best_bins

def _compute_heuristic_exp(demands, capacity, lambda_decay):
    """
    Compute heuristic based on exponential decay of the gap.
    H[i, j] = exp(-lambda * (capacity - W[i] - W[j])) if W[i] + W[j] <= capacity, else 1e-9.
    This rewards pairs that fill the bin closely.
    """
    W = np.array(demands, dtype=float)
    n = len(W)
    pair_sums = W[:, None] + W[None, :]
    fit_mask = pair_sums <= capacity
    
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        gaps = capacity - pair_sums
        # For non-fitting pairs, gap is positive (since W[i]+W[j] > cap -> gap < 0? No, gap = cap - sum).
        # If sum > cap, gap is negative. exp(-lambda * negative) = exp(positive) = large.
        # We must mask this.
        H = np.where(fit_mask, np.exp(-lambda_decay * gaps), 1e-9)
    
    # Ensure non-negative and finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H

def _compute_heuristic_inv_gap(demands, capacity):
    """
    Fallback heuristic: inverse gap.
    """
    W = np.array(demands, dtype=float)
    n = len(W)
    pair_sums = W[:, None] + W[None, :]
    fit_mask = pair_sums <= capacity
    
    with np.errstate(divide='ignore', invalid='ignore'):
        gaps = capacity - pair_sums
        H = np.where(fit_mask, 1.0 / (gaps + 1e-9), 1e-9)
    
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H

def heuristics(demands, capacity):
    W = np.array(demands, dtype=float)
    n = len(W)
    
    start_time = time.time()
    deadline = start_time + 280  # 50s margin for safety
    
    best_H = None
    best_score = float('inf')
    
    # Candidate lambda values for exponential decay
    # Smaller lambda -> flatter, more uniform
    # Larger lambda -> sharper, strongly prefers near-perfect fits
    lambdas = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 6.0, 12.0, 25.0]
    
    # Also test the inverse gap as a baseline candidate
    candidates = lambdas + ['inv_gap']
    
    # Evaluate candidates with reduced quick-eval cost
    for cand in candidates:
        if time.time() > deadline:
            break
        
        if cand == 'inv_gap':
            H = _compute_heuristic_inv_gap(W, capacity)
        else:
            H = _compute_heuristic_exp(W, capacity, cand)
        
        # Quick evaluation - use 8 ants, 12 iterations for speed
        score = _simulate_aco(W, capacity, H, n_ants=8, n_iters=12, seed=42)
        
        if score < best_score:
            best_score = score
            best_H = H
    
    # If we have time, refine around the best lambda with finer grid
    if best_H is not None and time.time() < deadline - 30:
        fine_lambdas = []
        # Generate a finer grid covering more range
        for base in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0]:
            for factor in [0.5, 0.7, 1.0, 1.4, 2.0]:
                fine_lambdas.append(base * factor)
        
        # Sort unique
        fine_lambdas = sorted(list(set(fine_lambdas)))
        
        for lam in fine_lambdas:
            if time.time() > deadline:
                break
            
            H = _compute_heuristic_exp(W, capacity, lam)
            score = _simulate_aco(W, capacity, H, n_ants=8, n_iters=12, seed=42)
            
            if score < best_score:
                best_score = score
                best_H = H
    
    # Final fallback
    if best_H is None:
        best_H = _compute_heuristic_inv_gap(W, capacity)
        
    return best_H
# EVOLVE-BLOCK-END

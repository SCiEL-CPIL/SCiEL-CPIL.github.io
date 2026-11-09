import numpy as np
import time
import random

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    Uses a structural change: instead of a fixed parametric form, we optimize 
    the weights of a set of basis heuristic functions using a fast proxy evaluator 
    (reduced ACO simulation) and derivative-free optimization.
    
    Basis functions capture different aspects of the problem:
    1. Prize-to-total-cost ratio
    2. Prize-to-step-cost ratio
    3. Exponential penalty on step cost
    4. Exponential penalty on total cost
    5. Prize-to-product-cost ratio
    6. Feasibility-based boost
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = distance_matrix.shape[0]
    if n < 2:
        return np.ones((n, n), dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Precompute useful arrays
    d = distance_matrix
    d_to_depot = d[:, 0]
    
    eps = 1e-9
    
    # Define Basis Heuristic Matrices
    # All bases are n x n
    
    # 1. Standard: Prize / (d_ij + d_j0)
    cost_total = d + d_to_depot[None, :]
    base1 = np.where(cost_total > eps, prizes[None, :] / cost_total, 0)
    
    # 2. Prize / d_ij
    base2 = np.where(d > eps, prizes[None, :] / d, 0)
    
    # 3. Prize * exp(-d_ij / max_len)
    base3 = prizes[None, :] * np.exp(-d / max_len)
    
    # 4. Prize * exp(-(d_ij + d_j0) / max_len)
    base4 = prizes[None, :] * np.exp(-cost_total / max_len)
    
    # 5. Prize / (d_ij * d_j0)
    prod_cost = d * d_to_depot[None, :]
    base5 = np.where(prod_cost > eps, prizes[None, :] / prod_cost, 0)
    
    # 6. Feasibility: 1 if d_ij + d_j0 <= max_len * 0.8 else 0.5
    # This encourages moves that leave plenty of budget
    feas_mask = (cost_total <= max_len * 0.8).astype(np.float64)
    base6 = feas_mask * prizes[None, :]
    
    bases = np.stack([base1, base2, base3, base4, base5, base6], axis=0)
    # bases shape: (6, n, n)
    
    # Ensure no NaNs or Infs in bases
    bases = np.nan_to_num(bases, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Fast Proxy Evaluator
    def evaluate_weights(weights, n_ants=10, n_iter=5):
        """
        Evaluates a linear combination of bases using a reduced ACO simulation.
        Returns the best total prize found.
        """
        if time.time() > deadline:
            return -np.inf
        
        # Construct heuristic
        # weights shape: (6,)
        # H = sum(w_k * B_k)
        H = np.tensordot(weights, bases, axes=1) # shape (n, n)
        
        # Ensure positive
        H = np.where(H > 0, H, 1e-9)
        
        best_total = 0
        pheromone = np.ones((n, n))
        
        for _ in range(n_iter):
            iter_best = 0
            for _ in range(n_ants):
                visited = set([0])
                current = 0
                length = 0.0
                total_prize = prizes[0]
                
                while True:
                    candidates = []
                    for j in range(1, n):
                        if j in visited:
                            continue
                        new_length = length + d[current, j] + d[j, 0]
                        if new_length <= max_len:
                            candidates.append(j)
                    
                    if not candidates:
                        break
                    
                    # Compute probabilities
                    probs = np.zeros(len(candidates))
                    for idx, j in enumerate(candidates):
                        probs[idx] = pheromone[current, j] * H[current, j]
                    
                    total_prob = probs.sum()
                    if total_prob < 1e-15:
                        probs = np.ones(len(candidates))
                        total_prob = len(candidates)
                    else:
                        probs /= total_prob
                    
                    # Sample
                    r = np.random.random()
                    cumsum = 0
                    chosen_idx = len(candidates) - 1
                    for idx in range(len(candidates)):
                        cumsum += probs[idx]
                        if r <= cumsum:
                            chosen_idx = idx
                            break
                    
                    j = candidates[chosen_idx]
                    length += d[current, j]
                    total_prize += prizes[j]
                    visited.add(j)
                    current = j
                
                if total_prize > iter_best:
                    iter_best = total_prize
                
                # Update pheromone
                pheromone *= 0.9
                pheromone += iter_best / 10.0
            
            best_total = max(best_total, iter_best)
            
            if time.time() > deadline:
                return best_total
        
        return best_total

    # Optimization
    # We want to maximize evaluate_weights
    # Use Random Search with Local Refinement (Hill Climbing)
    # since Nelder-Mead might be slow or unstable with noisy proxy.
    
    best_weights = None
    best_score = -np.inf
    
    # Initialize with some random positive weights
    # Normalize weights to sum to 1 to constrain the search space
    def normalize_weights(w):
        w = np.abs(w)
        s = w.sum()
        if s < 1e-9:
            return np.ones(6) / 6.0
        return w / s
    
    # Try a few random starts
    num_starts = 5
    steps_per_start = 20
    
    for _ in range(num_starts):
        if time.time() > deadline:
            break
            
        # Random initial weights
        w = np.random.rand(6)
        w = normalize_weights(w)
        
        current_score = evaluate_weights(w)
        if current_score > best_score:
            best_score = current_score
            best_weights = w.copy()
        
        # Local search: perturb one weight at a time
        for step in range(steps_per_start):
            if time.time() > deadline:
                break
                
            improved = False
            for k in range(6):
                # Try increasing and decreasing weight k
                for delta in [0.1, -0.1, 0.05, -0.05]:
                    w_new = w.copy()
                    w_new[k] += delta
                    w_new = normalize_weights(w_new)
                    
                    score_new = evaluate_weights(w_new)
                    if score_new > current_score:
                        w = w_new
                        current_score = score_new
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_weights = w.copy()
                        break # Move to next k if improved
                
                if time.time() > deadline:
                    break
            
            if not improved:
                # Try a larger jump or restart local search?
                # For now, just break if no small improvement
                # Could add random restart here
                pass

    # If optimization failed or didn't run, use a default robust heuristic
    if best_weights is None:
        # Default: mix of base1 and base4
        best_weights = np.array([0.5, 0.0, 0.0, 0.5, 0.0, 0.0])
    
    # Final Heuristic Construction
    H = np.tensordot(best_weights, bases, axes=1)
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H

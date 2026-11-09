import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes the Arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    
    Strategy:
    1. Deterministic Structured Annealing: 
       - Start with a broad support (e.g., all cells or a checkerboard) to maximize M.
       - Iteratively refine the support set (active cells) using a greedy heuristic:
         Try adding/removing cells to see if the ratio D/M improves.
       - This explicitly navigates the discrete structure of the support, 
         avoiding the "flat" regions of the probability simplex where 
         gradient-based methods might struggle to find the right sparsity pattern.
    2. High-Precision Refinement:
       - Once a good support structure is found, use L-BFGS-B on the log-probabilities 
         of the active cells to fine-tune the weights.
    3. Multiple Structured Starts:
       - Try several different initial support patterns (full, diagonal, anti-diagonal, 
         random sparse) to ensure robustness.
    """
    n = 10
    N = n * n
    start_time = time.time()
    deadline = start_time + 590  # 10s buffer for overhead

    # Pre-compute indices for entropy calculation
    ys = np.repeat(np.arange(n), n)
    xs = np.tile(np.arange(n), n)
    
    # Indices for binning
    idx_dm1 = (xs - ys + n - 1).astype(np.int32)       # x - y + 9 -> 0..18
    idx_x   = xs.astype(np.int32)                      # x -> 0..9
    idx_d1  = (xs + ys).astype(np.int32)               # x + y -> 0..18
    idx_d2  = (xs + 2 * ys).astype(np.int32)           # x + 2y -> 0..27
    
    bin_sizes = {
        'dm1': 2 * n - 1,
        'x': n,
        'd1': 2 * n - 1,
        'd2': 3 * n - 1
    }

    def entropy_from_counts(counts):
        """Calculate Shannon entropy (bits) from a count/probability vector."""
        mask = counts > 1e-15
        if not np.any(mask):
            return 0.0
        p = counts[mask]
        p_sum = p.sum()
        if p_sum < 1e-15:
            return 0.0
        p = p / p_sum
        return float(-np.sum(p * np.log2(p)))

    def evaluate_P(P_flat):
        """
        Evaluate the score for a flattened probability vector P_flat.
        """
        if not np.all(np.isfinite(P_flat)):
            return -np.inf
        
        # Clip and normalize
        P_norm = np.clip(P_flat, 0, None)
        s = P_norm.sum()
        if s < 1e-9:
            return -np.inf
        P_norm /= s

        c_dm1 = np.bincount(idx_dm1, weights=P_norm, minlength=bin_sizes['dm1'])
        c_x   = np.bincount(idx_x,   weights=P_norm, minlength=bin_sizes['x'])
        c_d1  = np.bincount(idx_d1,  weights=P_norm, minlength=bin_sizes['d1'])
        c_d2  = np.bincount(idx_d2,  weights=P_norm, minlength=bin_sizes['d2'])

        H_dm1 = entropy_from_counts(c_dm1)
        H_x   = entropy_from_counts(c_x)
        H_d1  = entropy_from_counts(c_d1)
        H_d2  = entropy_from_counts(c_d2)

        D = H_dm1
        M = max(H_x, H_d1, H_d2)
        
        if M < 1e-9:
            return -np.inf
            
        return D / M

    def score_from_log_probs(log_p, active_mask):
        """
        Objective function for minimization on active cells.
        Input: log_p (unnormalized log-probabilities for all N cells, but only active ones matter)
        Output: Negative score
        """
        # Only consider active cells
        p_active = np.exp(log_p[active_mask] - np.max(log_p[active_mask]))
        s = p_active.sum()
        if s < 1e-15:
            return 1e9 
        p_active = p_active / s
        
        # Reconstruct full P
        P_full = np.zeros(N)
        P_full[active_mask] = p_active
        
        score = evaluate_P(P_full)
        if not np.isfinite(score):
            return 1e9
        return -score

    def optimize_given_support(active_mask):
        """
        Given a boolean mask of active cells, optimize the weights.
        """
        num_active = np.sum(active_mask)
        if num_active == 0:
            return -np.inf, np.zeros(N)
        
        # Initial log-probs: uniform on active cells
        log_p_init = np.zeros(N)
        log_p_init[active_mask] = 0.0
        
        # Bounds
        lb = np.full(N, -50)
        ub = np.full(N, 50)
        
        # Only optimize active variables? L-BFGS-B doesn't support sparse variables directly easily,
        # so we optimize all but fix inactive ones to a very low value.
        # Actually, better to just optimize the active subset.
        
        # Let's map active indices to a smaller vector
        active_indices = np.where(active_mask)[0]
        
        def obj(x):
            # x is length num_active
            log_p_full = np.full(N, -50.0)
            log_p_full[active_indices] = x
            return score_from_log_probs(log_p_full, active_mask)
        
        x0 = np.zeros(num_active)
        
        try:
            res = minimize(
                obj,
                x0,
                method='L-BFGS-B',
                bounds=[(-30, 30)] * num_active,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            best_score = -res.fun
            log_p_best = np.full(N, -50.0)
            log_p_best[active_indices] = res.x
            
            # Convert to P
            p_active = np.exp(log_p_best[active_indices] - np.max(log_p_best[active_indices]))
            p_active /= p_active.sum()
            P_full = np.zeros(N)
            P_full[active_indices] = p_active
            
            return best_score, P_full
        except Exception:
            return -np.inf, np.zeros(N)

    def greedy_support_search(initial_mask, max_steps=500):
        """
        Greedy local search on the support set.
        """
        current_mask = initial_mask.copy()
        current_P, current_score = optimize_given_support(current_mask)
        
        if current_score == -np.inf:
            return current_mask, current_P, current_score
            
        improved = True
        step = 0
        while improved and step < max_steps and time.time() < deadline - 10:
            improved = False
            step += 1
            
            # Try adding a cell
            inactive_indices = np.where(~current_mask)[0]
            if len(inactive_indices) > 0:
                # Randomly sample a subset of inactive indices to try to save time
                # or try all if N is small. N=100 is small, so try all.
                for idx in inactive_indices:
                    new_mask = current_mask.copy()
                    new_mask[idx] = True
                    new_P, new_score = optimize_given_support(new_mask)
                    if new_score > current_score + 1e-6:
                        current_mask = new_mask
                        current_P = new_P
                        current_score = new_score
                        improved = True
                        break # Restart search with new mask
            
            # Try removing a cell
            if not improved:
                active_indices = np.where(current_mask)[0]
                if len(active_indices) > 1: # Keep at least 1 cell
                    for idx in active_indices:
                        new_mask = current_mask.copy()
                        new_mask[idx] = False
                        new_P, new_score = optimize_given_support(new_mask)
                        if new_score > current_score + 1e-6:
                            current_mask = new_mask
                            current_P = new_P
                            current_score = new_score
                            improved = True
                            break # Restart search with new mask
                            
        return current_mask, current_P, current_score

    # --- Phase 1: Structured Starts ---
    best_score_overall = -np.inf
    best_P_overall = np.ones(N) / N
    
    # Define several initial support structures
    # 1. Full support
    mask_full = np.ones(N, dtype=bool)
    
    # 2. Diagonal support
    mask_diag = np.zeros(N, dtype=bool)
    for i in range(n):
        mask_diag[i*n + i] = True
        
    # 3. Anti-diagonal support
    mask_anti_diag = np.zeros(N, dtype=bool)
    for i in range(n):
        mask_anti_diag[i*n + (n-1-i)] = True
        
    # 4. Checkerboard (high entropy potential)
    mask_checker = np.zeros(N, dtype=bool)
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                mask_checker[i*n + j] = True
                
    # 5. Random sparse (e.g., 20% density)
    np.random.seed(42)
    mask_sparse = np.random.rand(N) < 0.2
    if not np.any(mask_sparse):
        mask_sparse[0] = True
        
    # 6. Random medium (e.g., 50% density)
    mask_medium = np.random.rand(N) < 0.5
    if not np.any(mask_medium):
        mask_medium[0] = True

    starts = [
        ("full", mask_full),
        ("diag", mask_diag),
        ("anti_diag", mask_anti_diag),
        ("checker", mask_checker),
        ("sparse", mask_sparse),
        ("medium", mask_medium)
    ]
    
    for name, mask in starts:
        if time.time() > deadline - 20:
            break
            
        # Greedy search from this start
        # Limit steps to ensure we finish in time. 
        # Each step involves an optimization which is fast (<10ms usually).
        # 100 steps * 10ms = 1s per start. 6 starts = 6s. Safe.
        final_mask, final_P, final_score = greedy_support_search(mask, max_steps=100)
        
        if final_score > best_score_overall:
            best_score_overall = final_score
            best_P_overall = final_P
            
    # --- Phase 2: Final Refinement on Best Support ---
    # The greedy search already optimizes weights at each step.
    # But we can do one more high-precision polish on the best P found.
    
    if best_score_overall > -np.inf:
        # Determine the effective support of best_P_overall
        active_mask = best_P_overall > 1e-8
        if np.any(active_mask):
            _, polished_P = optimize_given_support(active_mask)
            polished_score = evaluate_P(polished_P)
            if polished_score > best_score_overall:
                best_P_overall = polished_P
                best_score_overall = polished_score

    # --- Final Validation ---
    if not np.all(np.isfinite(best_P_overall)) or best_P_overall.shape != (n, n):
        return np.ones((n, n)) / N
        
    # Clip and normalize
    P_final = np.clip(best_P_overall, 0, None)
    s = P_final.sum()
    if s > 1e-9:
        P_final /= s
    else:
        P_final = np.ones(N) / N
        
    return P_final.reshape((n, n))

if __name__ == "__main__":
    pass

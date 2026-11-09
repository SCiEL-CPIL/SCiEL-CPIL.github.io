import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes the Arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    Uses L-BFGS-B optimization on log-probabilities with structured initializations.
    """
    n = 10
    N = n * n
    deadline = time.time() + 590  # Leave 10s buffer for safety
    
    # Pre-compute indices for entropy calculation
    # P[y, x] -> flat index y*n + x
    ys = np.repeat(np.arange(n), n) 
    xs = np.tile(np.arange(n), n)   
    
    idx_dm1 = (xs - ys + n - 1).astype(np.int32)       # x - y + 9, range 0..18
    idx_x   = xs.astype(np.int32)                      # x, range 0..9
    idx_d1  = (xs + ys).astype(np.int32)               # x + y, range 0..18
    idx_d2  = (xs + 2 * ys).astype(np.int32)           # x + 2y, range 0..27
    
    bin_sizes = {
        'dm1': 2 * n - 1,
        'x': n,
        'd1': 2 * n - 1,
        'd2': 3 * n - 1
    }

    def compute_entropies(P_flat):
        """
        Compute D and M from a probability vector.
        Returns (D, M).
        """
        # Ensure P is valid and normalized
        P = np.clip(P_flat, 0, None)
        s = P.sum()
        if s < 1e-15:
            return 0.0, 0.0
        P = P / s
        
        c_dm1 = np.bincount(idx_dm1, weights=P, minlength=bin_sizes['dm1'])
        c_x   = np.bincount(idx_x,   weights=P, minlength=bin_sizes['x'])
        c_d1  = np.bincount(idx_d1,  weights=P, minlength=bin_sizes['d1'])
        c_d2  = np.bincount(idx_d2,  weights=P, minlength=bin_sizes['d2'])
        
        def H(counts):
            mask = counts > 1e-15
            if not np.any(mask):
                return 0.0
            p = counts[mask]
            p = p / p.sum()
            return -np.sum(p * np.log2(p))
            
        D = H(c_dm1)
        H_x = H(c_x)
        H_d1 = H(c_d1)
        H_d2 = H(c_d2)
        M = max(H_x, H_d1, H_d2)
        
        return D, M

    def objective(z):
        """
        Objective function for minimization.
        z is the unconstrained log-probability vector.
        Returns -D/M (since we want to maximize D/M).
        """
        # Softmax transformation to ensure P is a valid probability distribution
        # Subtract max for numerical stability
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        P = exp_z / np.sum(exp_z)
        
        D, M = compute_entropies(P)
        
        if M < 1e-9:
            return 1e6  # Large penalty for invalid M
        return -D / M

    def get_P_from_z(z):
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        P = exp_z / np.sum(exp_z)
        return P

    # --- Generate Structured Initial Candidates ---
    
    candidates_z = []
    
    # 1. Uniform
    P_uniform = np.ones(N) / N
    z_uniform = np.log(P_uniform)
    candidates_z.append(z_uniform)
    
    # 2. Row (y=0)
    P_row = np.zeros(N)
    P_row[0:n] = 1.0 / n
    # Add small epsilon to avoid log(0)
    P_row_safe = P_row + 1e-10
    P_row_safe /= P_row_safe.sum()
    candidates_z.append(np.log(P_row_safe))
    
    # 3. Column (x=0)
    P_col = np.zeros(N)
    P_col[0::n] = 1.0 / n
    P_col_safe = P_col + 1e-10
    P_col_safe /= P_col_safe.sum()
    candidates_z.append(np.log(P_col_safe))
    
    # 4. Main Diagonal (x=y)
    P_diag = np.zeros(N)
    for i in range(n):
        P_diag[i*n + i] = 1.0 / n
    P_diag_safe = P_diag + 1e-10
    P_diag_safe /= P_diag_safe.sum()
    candidates_z.append(np.log(P_diag_safe))
    
    # 5. Anti-Diagonal (x+y = n-1)
    P_adiag = np.zeros(N)
    for i in range(n):
        P_adiag[i*n + (n-1-i)] = 1.0 / n
    P_adiag_safe = P_adiag + 1e-10
    P_adiag_safe /= P_adiag_safe.sum()
    candidates_z.append(np.log(P_adiag_safe))
    
    # 6. Random Log-Normal (Softmax of Gaussian)
    rng = np.random.default_rng(42)
    for _ in range(20):
        z_rand = rng.standard_normal(N) * rng.uniform(0.5, 5.0)
        candidates_z.append(z_rand)
        
    # 7. Sparse Log-Normal
    for _ in range(20):
        z_sparse = np.full(N, -10.0) # Very small base
        k = rng.integers(5, 20)
        idx = rng.choice(N, size=k, replace=False)
        z_sparse[idx] = rng.standard_normal(k) * 2.0
        candidates_z.append(z_sparse)

    best_score = -np.inf
    best_P = None
    best_z = None
    
    # Evaluate initial candidates to find a good starting point
    for z_cand in candidates_z:
        P_cand = get_P_from_z(z_cand)
        D, M = compute_entropies(P_cand)
        if M < 1e-9:
            continue
        score = D / M
        if score > best_score:
            best_score = score
            best_P = P_cand.copy()
            best_z = z_cand.copy()

    if best_P is None:
        # Fallback to uniform
        best_P = np.ones(N) / N
        best_z = np.log(best_P)
        best_score = 1.0 # Uniform score is roughly 1.0 (D~3.17, M~3.17)

    # --- Optimization Phase ---
    
    # We will run L-BFGS-B from several of the best initial points
    # Sort candidates by initial score
    cand_scores = []
    for z_cand in candidates_z:
        P_cand = get_P_from_z(z_cand)
        D, M = compute_entropies(P_cand)
        if M < 1e-9:
            s = -np.inf
        else:
            s = D / M
        cand_scores.append((s, z_cand))
        
    cand_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Select top 5 distinct starting points for optimization
    top_starts = []
    seen_signatures = set()
    for s, z in cand_scores:
        if s == -np.inf:
            continue
        # Simple signature: rounded log-probs
        sig = tuple(np.round(z, 2))
        if sig not in seen_signatures:
            top_starts.append(z)
            seen_signatures.add(sig)
            if len(top_starts) >= 5:
                break
    
    # If we didn't get 5, add more random ones
    while len(top_starts) < 5:
        top_starts.append(rng.standard_normal(N) * 2.0)

    # Run optimization
    start_time = time.time()
    remaining_budget = deadline - start_time
    
    # Split budget among optimizations
    time_per_opt = max(5.0, remaining_budget / (len(top_starts) + 2))
    
    for i, z0 in enumerate(top_starts):
        if time.time() > deadline - 30:
            break
            
        # L-BFGS-B parameters
        # maxiter is limited by time, but we also set a hard timeout via callback
        stop_flag = [False]
        
        def callback(xk):
            if time.time() > deadline - 30:
                stop_flag[0] = True
            return stop_flag[0]
            
        try:
            res = minimize(
                objective, 
                z0, 
                method='L-BFGS-B', 
                options={
                    'maxiter': 500,
                    'ftol': 1e-9,
                    'gtol': 1e-6,
                    'maxfun': 1000,
                    'callback': callback
                }
            )
            
            if res.success or res.fun < -best_score:
                z_opt = res.x
                P_opt = get_P_from_z(z_opt)
                D, M = compute_entropies(P_opt)
                if M > 1e-9:
                    score_opt = D / M
                    if score_opt > best_score:
                        best_score = score_opt
                        best_P = P_opt
                        best_z = z_opt
                        
        except Exception as e:
            # If optimization fails, continue with next start
            pass
            
    # Final Cleanup
    P_final = best_P.copy()
    P_final[P_final < 0] = 0
    s = P_final.sum()
    if s < 1e-9:
        P_final = np.ones(N) / N
    else:
        P_final /= s
        
    return P_final.reshape((n, n))

if __name__ == "__main__":
    pass

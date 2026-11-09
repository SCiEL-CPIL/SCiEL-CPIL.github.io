import numpy as np
import time
import itertools

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes the Arithmetic Kakeya entropy ratio D/M on a 10x10 grid.
    Uses a hybrid approach:
    1. Analytical construction of sparse distributions (atoms) to find good baselines.
    2. Discrete coordinate descent to refine the support and weights.
    
    This avoids the pitfalls of continuous gradient methods on non-smooth objectives 
    (max of entropies) by directly manipulating the discrete support set.
    """
    n = 10
    N = n * n
    deadline = time.time() + 580  # Safety margin
    
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
        Returns (D, M, score).
        """
        # Ensure P is valid and normalized
        P = np.clip(P_flat, 0, None)
        s = P.sum()
        if s < 1e-15:
            return 0.0, 0.0, 0.0
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
        
        if M < 1e-9:
            return D, M, 0.0
        return D, M, D / M

    def get_P_from_weights(support_indices, weights):
        """Construct a probability vector from support indices and weights."""
        P = np.zeros(N)
        if len(support_indices) == 0:
            return P
        P[support_indices] = weights
        return P

    def refine_distribution(P_init):
        """
        Perform discrete coordinate descent.
        1. Identify active support.
        2. Try adding new atoms, removing atoms, or adjusting weights.
        """
        P = P_init.copy()
        P[P < 1e-12] = 0
        support = np.where(P > 1e-12)[0]
        weights = P[support]
        
        best_score, _, _ = compute_entropies(P)
        current_score = best_score
        
        max_iters = 5000
        iter_count = 0
        
        while iter_count < max_iters:
            if time.time() > deadline - 5:
                break
            iter_count += 1
            improved = False
            
            # Strategy A: Try adding a new atom
            if len(support) < 50: # Limit support size for speed
                # Candidate atoms: all indices not in support
                candidates = np.setdiff1d(np.arange(N), support)
                if len(candidates) > 0:
                    # Heuristic: try atoms that maximize entropy increase in D
                    # or minimize increase in M.
                    # Simple random sampling of candidates to keep it fast
                    num_trials = min(20, len(candidates))
                    trial_indices = np.random.choice(candidates, size=num_trials, replace=False)
                    
                    for idx in trial_indices:
                        new_support = np.concatenate([support, [idx]])
                        new_weights = np.concatenate([weights, [0.1]]) # Initial weight
                        P_trial = get_P_from_weights(new_support, new_weights)
                        D, M, score = compute_entropies(P_trial)
                        if score > current_score:
                            current_score = score
                            support = new_support
                            weights = new_weights
                            P = P_trial
                            improved = True
                            break
            
            # Strategy B: Try removing an atom
            if len(support) > 2:
                # Try removing the atom with the smallest weight or one that hurts M most
                # Simple: try removing random atoms
                num_trials = min(10, len(support))
                trial_indices = np.random.choice(len(support), size=num_trials, replace=False)
                
                for i in trial_indices:
                    new_support = np.delete(support, i)
                    new_weights = np.delete(weights, i)
                    P_trial = get_P_from_weights(new_support, new_weights)
                    D, M, score = compute_entropies(P_trial)
                    if score > current_score:
                        current_score = score
                        support = new_support
                        weights = new_weights
                        P = P_trial
                        improved = True
                        break
            
            # Strategy C: Adjust weights
            # Weights should be proportional to the "value" of the cell.
            # For a fixed support, the optimal weights often equalize the marginal contributions
            # or follow a pattern. Let's try a few heuristic weight distributions.
            if len(support) > 0:
                heuristics = [
                    np.ones(len(support)) / len(support), # Uniform
                    np.sort(weights)[::-1], # Decaying
                    np.array([1.0 if i % 2 == 0 else 0.5 for i in range(len(support))]) # Alternating
                ]
                
                for h in heuristics:
                    h_norm = h / h.sum()
                    P_trial = get_P_from_weights(support, h_norm)
                    D, M, score = compute_entropies(P_trial)
                    if score > current_score:
                        current_score = score
                        weights = h_norm
                        P = P_trial
                        improved = True
                        break
                
                # Fine-tune: Nudge weights slightly
                if len(support) > 1:
                    for _ in range(5):
                        perturbation = np.random.normal(0, 0.1, size=len(support))
                        new_weights = weights + perturbation
                        new_weights = np.clip(new_weights, 1e-6, None)
                        new_weights = new_weights / new_weights.sum()
                        P_trial = get_P_from_weights(support, new_weights)
                        D, M, score = compute_entropies(P_trial)
                        if score > current_score:
                            current_score = score
                            weights = new_weights
                            P = P_trial
                            improved = True

            if not improved:
                # Stuck in local optimum, try a bigger jump: randomize weights significantly
                if len(support) > 0 and np.random.random() < 0.1:
                    new_weights = np.random.dirichlet(np.ones(len(support)))
                    P_trial = get_P_from_weights(support, new_weights)
                    D, M, score = compute_entropies(P_trial)
                    if score > current_score:
                        current_score = score
                        weights = new_weights
                        P = P_trial
                    else:
                        # If randomization doesn't help, break to avoid wasting time
                        # unless we've done very few iterations
                        if iter_count > 100:
                            break

        return P, current_score

    # --- Generate Initial Candidates ---
    
    candidates = []
    
    # 1. Uniform
    P_uniform = np.ones(N) / N
    candidates.append(P_uniform)
    
    # 2. Diagonal (x=y) -> D=0, Bad.
    # 3. Anti-Diagonal (x+y=9) -> D is spread, M is concentrated?
    P_adiag = np.zeros(N)
    for i in range(n):
        P_adiag[i*n + (n-1-i)] = 1.0 / n
    candidates.append(P_adiag)
    
    # 4. Main Diagonal offset (x = y + 1)
    P_diag_off = np.zeros(N)
    for i in range(n-1):
        P_diag_off[i*n + (i+1)] = 1.0 / (n-1)
    candidates.append(P_diag_off)
    
    # 5. Sparse Random
    rng = np.random.default_rng(42)
    for _ in range(10):
        k = rng.integers(3, 15)
        idx = rng.choice(N, size=k, replace=False)
        P_sparse = np.zeros(N)
        P_sparse[idx] = 1.0 / k
        candidates.append(P_sparse)
        
    # 6. Structured: Checkerboard-like
    P_checker = np.zeros(N)
    for y in range(n):
        for x in range(n):
            if (x + y) % 2 == 0:
                P_checker[y, x] = 1.0
    P_checker /= P_checker.sum()
    candidates.append(P_checker)
    
    # 7. Rows/Columns
    P_row = np.zeros(N)
    P_row[0:n] = 1.0/n
    candidates.append(P_row)
    
    P_col = np.zeros(N)
    P_col[0::n] = 1.0/n
    candidates.append(P_col)

    best_score = -np.inf
    best_P = None
    
    # Evaluate and refine each candidate
    for P_init in candidates:
        if time.time() > deadline - 10:
            break
        
        P_refined, score = refine_distribution(P_init)
        
        if score > best_score:
            best_score = score
            best_P = P_refined.copy()
            
    # Fallback if all failed
    if best_P is None:
        best_P = np.ones(N) / N
        best_score = compute_entropies(best_P)[2]

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

"""Matrix multiplication 3x5x7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    trivial_rank = n * m * p
    
    # Build the trivial decomposition
    U = np.zeros((nm, trivial_rank), dtype=float)
    V = np.zeros((mp, trivial_rank), dtype=float)
    W = np.zeros((pn, trivial_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Precompute valid triples (a, b, c)
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
                
    # Convert to numpy arrays for faster residual calculation
    # We need to evaluate sum(U[a] * V[b] * W[c]) for all (a,b,c)
    # We can do this efficiently by looping over c or using einsum if we had full tensors,
    # but here we only have sparse non-zeros.
    
    # For zero entries, we need to ensure they are 0.
    # Total entries = nm * mp * pn = 15 * 35 * 21 = 11025.
    # This is small enough to check all in verification, but for optimization,
    # we should focus on the non-zero constraints first, or use a weighted sum.
    
    # To make optimization faster, we will define the residual based on the 105 non-zero entries
    # and a sample of zero entries, or all zero entries if fast enough.
    # 11025 entries is small. Let's use all entries for robustness.
    
    # Precompute indices for all entries
    all_indices_a = np.repeat(np.arange(nm), mp * pn)
    all_indices_b = np.tile(np.repeat(np.arange(mp), pn), nm)
    all_indices_c = np.tile(np.arange(pn), nm * mp)
    
    # Identify which are valid (1.0) and which are invalid (0.0)
    valid_set = set(triples)
    targets = np.zeros(nm * mp * pn)
    for idx, (a, b, c) in enumerate(zip(all_indices_a, all_indices_b, all_indices_c)):
        if (a, b, c) in valid_set:
            targets[idx] = 1.0
        else:
            targets[idx] = 0.0
            
    # Vectorized residual calculation
    # U[all_indices_a] is shape (N,)
    # V[all_indices_b] is shape (N,)
    # W[all_indices_c] is shape (N,)
    # Product is shape (N,)
    
    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        """Check if the decomposition correctly represents the multiplication tensor."""
        # Compute all tensor entries
        tensor_vals = U_mat[all_indices_a] * V_mat[all_indices_b] * W_mat[all_indices_c]
        # Check against targets
        if np.any(np.abs(tensor_vals - targets) > tol):
            return False
        return True

    def time_left():
        return deadline - time.time()

    def try_rank(r, n_seeds=10):
        """Try to find a rank-r decomposition using Levenberg-Marquardt."""
        if r >= trivial_rank:
            return None
        
        best_residual = float('inf')
        best_result = None
        
        # Flatten U, V, W into a single vector
        # Size: r * (nm + mp + pn)
        total_vars = r * (nm + mp + pn)
        
        for seed in range(n_seeds):
            if time_left() < 10:
                break
            
            np.random.seed(seed * 1000 + 42)
            
            # Random initialization
            # Scale to keep values around 1
            U_mat = np.random.randn(nm, r) * 0.5
            V_mat = np.random.randn(mp, r) * 0.5
            W_mat = np.random.randn(pn, r) * 0.5
            
            x0 = np.concatenate([U_mat.flatten(), V_mat.flatten(), W_mat.flatten()])
            
            def residuals(x):
                if time_left() < 5:
                    return np.zeros(1) # Signal stop? No, just return current.
                
                U_flat = x[:nm * r]
                V_flat = x[nm * r : nm * r + mp * r]
                W_flat = x[nm * r + mp * r :]
                
                U_cur = U_flat.reshape(nm, r)
                V_cur = V_flat.reshape(mp, r)
                W_cur = W_flat.reshape(pn, r)
                
                # Calculate tensor values
                vals = U_cur[all_indices_a] * V_cur[all_indices_b] * W_cur[all_indices_c]
                
                # Return residuals: vals - targets
                return vals - targets
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=5000)
                if result.success or result.cost < best_residual:
                    x_sol = result.x
                    U_sol = x_sol[:nm * r].reshape(nm, r)
                    V_sol = x_sol[nm * r : nm * r + mp * r].reshape(mp, r)
                    W_sol = x_sol[nm * r + mp * r :].reshape(pn, r)
                    
                    res_val = result.cost
                    if res_val < best_residual:
                        best_residual = res_val
                        best_result = (U_sol, V_sol, W_sol)
                        
                    if res_val < 1e-12:
                        # Attempt to snap to half-integers
                        U_clean = np.round(U_sol * 2) / 2
                        V_clean = np.round(V_sol * 2) / 2
                        W_clean = np.round(W_sol * 2) / 2
                        if verify(U_clean, V_clean, W_clean, tol=1e-6):
                            return (U_clean, V_clean, W_clean)
                        
                        # Attempt to snap to integers
                        U_int = np.round(U_sol)
                        V_int = np.round(V_sol)
                        W_int = np.round(W_sol)
                        if verify(U_int, V_int, W_int, tol=1e-6):
                            return (U_int, V_int, W_int)
                        
                        # If we have a very low residual but can't snap, 
                        # it might be due to scaling. Try to rescale?
                        # Or just accept it if we can't snap? 
                        # The problem requires integer/half-integer.
                        # If we can't snap, this rank is likely not achievable with these numbers 
                        # or we need a better local search.
                        # For now, continue searching for a better residual or different seed.
                        
            except Exception as e:
                continue
        
        # If we found a good residual but didn't snap, try harder to snap?
        if best_result is not None and best_residual < 1e-8:
            U_r, V_r, W_r = best_result
            # Try rounding with different tolerances or scales?
            # Sometimes the solution is (1/2, 1/2, 1/2) which rounds to 1/2.
            # If verify fails, it means the rounded values don't satisfy the equations.
            # This implies the exact solution is not half-integer, or our residual is misleading.
            # Given the problem statement, an exact decomposition exists.
            # We return None if we can't verify a snapped version.
            pass
            
        return None

    # Try ranks from lower bound up
    lower_bound = max(nm, mp, pn)
    
    U_best = U.copy()
    V_best = V.copy()
    W_best = W.copy()
    
    # We know rank 105 works.
    # Let's try to find a lower one.
    # Start from 35 (lower bound) up to 104.
    # Given time constraints, we can't try all.
    # Let's try a few promising ranks.
    # Known: R(3,5,7) <= 35? 
    # Actually, R(3,5,7) is likely around 35-40.
    
    ranks_to_try = list(range(lower_bound, trivial_rank))
    
    # Optimize order: try lower ranks first, but if they fail quickly, move on.
    # We will iterate through them.
    
    for r in ranks_to_try:
        if time_left() < 15:
            break
            
        # Allocate more time for lower ranks as they are harder? 
        # Or just fixed budget per rank.
        # 145s / 70 ranks ~ 2s per rank. Too short for LM.
        # We should limit the number of ranks we try deeply.
        
        # Strategy: Try ranks 35, 36, 37, 38, 39, 40, 42, 45, 50.
        # If we don't find one, fall back to 105.
        
        if r in [35, 36, 37, 38, 39, 40, 42, 45, 50, 60]:
             result = try_rank(r, n_seeds=5)
             if result is not None:
                 U_best, V_best, W_best = result
                 break
        else:
             # Skip for now to save time, or do a very quick check?
             # Let's just skip non-critical ranks if time is tight.
             if time_left() < 30:
                 continue
             result = try_rank(r, n_seeds=1)
             if result is not None:
                 U_best, V_best, W_best = result
                 break

    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

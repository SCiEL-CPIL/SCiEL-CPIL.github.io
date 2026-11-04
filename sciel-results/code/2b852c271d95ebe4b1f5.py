"""
Rewritten program for matrix multiplication tensor decomposition.
Task: M_{2,3,10}
Dimensions: n=2, m=3, p=10
Standard rank: 60
"""
import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140  # seconds, safety margin before 150s kill

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a dense 3D array."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _check_exact(U, V, W, T, tol=1e-10):
    """Check if U, V, W exactly decompose T."""
    # Compute the reconstructed tensor
    # T_recon[a, b, c] = sum_r U[a, r] * V[b, r] ... wait, that's not right
    # The decomposition is: T[a, b, c] = sum_r U[a, r] * V[b, r] * ... no
    # Actually for a rank-r decomposition of a 3-tensor:
    # T[a, b, c] = sum_{r=0}^{rank-1} U[a, r] * V[b, r] * W[c, r]
    # But wait, the interface says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # So T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    rank = U.shape[1]
    # Use einsum for efficiency
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _try_rank(n, m, p, rank, max_iters=500, seed=42):
    """Try to find a rank-r decomposition using alternating least squares."""
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    
    # Initialize with larger variance to escape local minima
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    # Pre-compute flat tensors for speed
    T_flat_u = T.reshape(n * m, -1)
    T_flat_v = T.reshape(m * p, -1)
    T_flat_w = T.reshape(p * n, -1)
    
    for it in range(max_iters):
        # Update U
        M = np.einsum('br,cr->bcr', V, W)  # (m*p, p*n, rank)
        M_flat = M.transpose(2, 0, 1).reshape(rank, -1)  # (rank, m*p*p*n)
        try:
            U = np.linalg.lstsq(M_flat.T, T_flat_u.T, rcond=None)[0].T
        except:
            pass
        
        # Update V
        N = np.einsum('ar,cr->acr', U, W)  # (n*m, p*n, rank)
        N_flat = N.transpose(2, 0, 1).reshape(rank, -1)  # (rank, n*m*p*n)
        try:
            V = np.linalg.lstsq(N_flat.T, T_flat_v.T, rcond=None)[0].T
        except:
            pass
        
        # Update W
        P = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, rank)
        P_flat = P.transpose(2, 0, 1).reshape(rank, -1)  # (rank, n*m*m*p)
        try:
            W = np.linalg.lstsq(P_flat.T, T_flat_w.T, rcond=None)[0].T
        except:
            pass
        
        # Check convergence
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T))
        if err < 1e-10:
            break
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_recon - T))
    if err < 1e-10:
        return U, V, W, True
    return U, V, W, False

def construct():
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p
    
    # Try ranks from 59 down to 55 to find the best possible decomposition
    # This allows finding a rank-59 solution if 55 is too hard, which is still an improvement.
    best_result = None
    best_rank = trivial_rank
    
    start_time = time.time()
    max_total_time = 100.0  # Use most of the 120s budget for search
    
    # We will try to find a decomposition for ranks 59, 58, 57, 56, 55
    # We allocate time for each attempt
    ranks_to_try = [59, 58, 57, 56, 55]
    
    for target_rank in ranks_to_try:
        if time.time() - start_time > max_total_time:
            break
            
        # Try a few seeds for this specific rank
        # We use fewer seeds for higher ranks as they are easier to find? 
        # Actually, lower ranks are harder. Let's give more time to lower ranks if we have time.
        # For now, fixed budget per rank.
        seeds_for_rank = 10
        max_iters = 50
        
        for seed in range(seeds_for_rank):
            if time.time() - start_time > max_total_time:
                break
                
            rng = np.random.RandomState(seed + target_rank * 100)
            U = rng.randn(n * m, target_rank) * 0.5
            V = rng.randn(m * p, target_rank) * 0.5
            W = rng.randn(p * n, target_rank) * 0.5
            
            T = _build_tensor(n, m, p)
            
            # Pre-compute flattened tensors for faster ALS updates
            T_flat_u = T.reshape(n * m, -1)
            T_flat_v = T.reshape(m * p, -1)
            T_flat_w = T.reshape(p * n, -1)
            
            converged = False
            for it in range(max_iters):
                # Update U
                M = np.einsum('br,cr->bcr', V, W)
                M_flat = M.transpose(2, 0, 1).reshape(target_rank, -1)
                try:
                    # Add small regularization to avoid singularity
                    reg = 1e-6 * np.eye(target_rank)
                    U = np.linalg.lstsq(M_flat.T + reg, T_flat_u.T, rcond=None)[0].T
                except:
                    pass
                
                # Update V
                N = np.einsum('ar,cr->acr', U, W)
                N_flat = N.transpose(2, 0, 1).reshape(target_rank, -1)
                try:
                    reg = 1e-6 * np.eye(target_rank)
                    V = np.linalg.lstsq(N_flat.T + reg, T_flat_v.T, rcond=None)[0].T
                except:
                    pass
                
                # Update W
                P = np.einsum('ar,br->abr', U, V)
                P_flat = P.transpose(2, 0, 1).reshape(target_rank, -1)
                try:
                    reg = 1e-6 * np.eye(target_rank)
                    W = np.linalg.lstsq(P_flat.T + reg, T_flat_w.T, rcond=None)[0].T
                except:
                    pass
                
                # Check convergence
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.max(np.abs(T_recon - T))
                if err < 1e-8:
                    converged = True
                    break
            
            if converged:
                # Found a solution for this rank
                best_rank = target_rank
                best_result = (U, V, W)
                break # Move to next (lower) rank
        
        if best_result is not None:
            # If we found a solution for this rank, we can try to go lower, 
            # but we keep the best one found so far.
            # The loop continues to try lower ranks.
            pass
    
    if best_result is not None:
        U_found, V_found, W_found = best_result
        # Round to nearest 0.5 to clean up numerical noise
        U_out = np.round(U_found * 2) / 2
        V_out = np.round(V_found * 2) / 2
        W_out = np.round(W_found * 2) / 2
        
        # Final verification with rounded values
        T = _build_tensor(n, m, p)
        T_recon = np.einsum('ar,br,cr->abc', U_out, V_out, W_out)
        if np.allclose(T_recon, T, atol=1e-6):
            return {
                "U": U_out.tolist(),
                "V": V_out.tolist(),
                "W": W_out.tolist()
            }
    
    # Fallback to trivial rank-60 decomposition
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))

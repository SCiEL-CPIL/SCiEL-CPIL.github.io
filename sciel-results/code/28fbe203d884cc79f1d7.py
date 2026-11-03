import time
import numpy as np

DIMENSIONS = [3, 3, 7]
n, m, p = DIMENSIONS

def build_tensor(n, m, p):
    """Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, n * p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def schoolbook_decomposition(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank

def verify_decomposition(U, V, W, T, tol=1e-8):
    """Verify that sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] ≈ T."""
    recon = np.einsum('ac,bc,cc->abc', U, V, W)
    err = np.linalg.norm(T - recon, 'fro')
    return err < tol, err

def alternating_least_squares(T, rank, max_iters=200, tol=1e-10, seed=42, deadline=None):
    """
    Alternating Least Squares to find a rank decomposition of tensor T.
    T has shape (I, J, K). We seek U (I x rank), V (J x rank), W (K x rank)
    such that sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] ≈ T.
    """
    I, J, K = T.shape
    np.random.seed(seed)
    
    U = np.random.randn(I, rank)
    V = np.random.randn(J, rank)
    W = np.random.randn(K, rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Update U columns
        for c in range(rank):
            # Compute residual: T - sum_{c'!=c} U_{c'} ⊗ V_{c'} ⊗ W_{c'}
            # For efficiency, compute the full reconstruction and subtract this term
            recon = np.einsum('ac,bc,cc->abc', U, V, W)
            recon_c = np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
            T_rest = T - recon + recon_c
            
            # U[:,c] = (T_rest * V_c * W_c) / (||V_c||^2 * ||W_c||^2)
            numer = np.einsum('abc,b,c->a', T_rest, V[:, c], W[:, c])
            denom = np.dot(V[:, c], V[:, c]) * np.dot(W[:, c], W[:, c])
            if denom > 1e-15:
                U[:, c] = numer / denom
        
        # Update V columns
        for c in range(rank):
            recon = np.einsum('ac,bc,cc->abc', U, V, W)
            recon_c = np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
            T_rest = T - recon + recon_c
            
            numer = np.einsum('abc,a,c->b', T_rest, U[:, c], W[:, c])
            denom = np.dot(U[:, c], U[:, c]) * np.dot(W[:, c], W[:, c])
            if denom > 1e-15:
                V[:, c] = numer / denom
        
        # Update W columns
        for c in range(rank):
            recon = np.einsum('ac,bc,cc->abc', U, V, W)
            recon_c = np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
            T_rest = T - recon + recon_c
            
            numer = np.einsum('abc,a,b->c', T_rest, U[:, c], V[:, c])
            denom = np.dot(U[:, c], U[:, c]) * np.dot(V[:, c], V[:, c])
            if denom > 1e-15:
                W[:, c] = numer / denom
        
        # Compute error
        recon = np.einsum('ac,bc,cc->abc', U, V, W)
        err = np.linalg.norm(T - recon, 'fro')
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < tol:
            break
    
    return best_U, best_V, best_W, best_err

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    T = build_tensor(n, m, p)
    I, J, K = T.shape  # 9, 21, 21
    
    # Start with schoolbook
    U_sb, V_sb, W_sb, rank_sb = schoolbook_decomposition(n, m, p)
    valid, err = verify_decomposition(U_sb, V_sb, W_sb, T)
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    best_err = err if valid else float('inf')
    
    # Try to find lower rank decompositions
    # The tensor rank of 3x3x7 is at most 63 (schoolbook).
    # We can potentially do better using Strassen-style methods.
    # Known: 3x3x3 has rank 21, 3x3x4 has rank at most 36 (schoolbook) but possibly lower.
    # For 3x3x7, we can try various ranks.
    
    # Strategy: Try ranks from lower to higher, using ALS
    # Start with ranks that are plausible improvements
    target_ranks = [54, 51, 48, 45, 42, 39, 36, 33, 30, 27, 24, 21]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            continue
        
        # Try multiple random seeds for robustness
        found = False
        for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]:
            if time.time() > deadline - 5:
                break
            
            try:
                U_t, V_t, W_t, err = alternating_least_squares(
                    T, target_rank, max_iters=100, tol=1e-10, seed=seed,
                    deadline=deadline
                )
                
                if err < 1e-8:
                    # Verify
                    valid, err = verify_decomposition(U_t, V_t, W_t, T, tol=1e-8)
                    if valid:
                        best_rank = target_rank
                        best_result = {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
                        best_err = err
                        found = True
                        break
            except:
                continue
        
        if found:
            # Found a valid decomposition at this rank, try to go lower
            continue
        else:
            # Didn't find at this rank, but keep trying lower ranks
            # (ALS might need more iterations or different seeds)
            continue
    
    # Final verification
    U_final = np.array(best_result["U"])
    V_final = np.array(best_result["V"])
    W_final = np.array(best_result["W"])
    valid, final_err = verify_decomposition(U_final, V_final, W_final, T, tol=1e-8)
    
    if not valid:
        # Fallback to schoolbook
        U_sb, V_sb, W_sb, rank_sb = schoolbook_decomposition(n, m, p)
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return best_result

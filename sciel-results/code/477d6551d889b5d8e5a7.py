"""
Rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication.

Strategy:
1. Use a block decomposition (Strassen + Schoolbook) as a high-rank baseline.
2. Perform a stochastic search for lower ranks using Randomized ALS.
3. Incorporate "sparsity snapping" in the ALS loop to encourage integer/half-integer solutions.
4. Search ranks from low to high, returning the first valid exact decomposition found.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomp(n, m, p, U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    # Efficient verification using einsum
    T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
    T_target = _build_target_tensor(n, m, p)
    return np.allclose(T_comp, T_target, atol=tol)

def _snap_to_half_integers(matrix, threshold=0.05):
    """
    Snap matrix entries close to 0, 0.5, 1, -0.5, -1 to those values.
    This helps in finding exact discrete decompositions.
    """
    M = matrix.copy()
    # Snap to 0
    M[np.abs(M) < threshold] = 0.0
    # Snap to 0.5
    close_to_05 = np.abs(np.abs(M) - 0.5) < threshold
    M[close_to_05] = np.sign(M[close_to_05]) * 0.5
    # Snap to 1
    close_to_1 = np.abs(np.abs(M) - 1.0) < threshold
    M[close_to_1] = np.sign(M[close_to_1]) * 1.0
    # Snap to -0.5 and -1 is handled by sign
    return M

def _als_step_U(U, V, W, T_target, n, m, p, lam=1e-6):
    """Update U given V, W, and target tensor T."""
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    
    # Construct design matrix M where M[b*pn+c, k] = V[b,k] * W[c,k]
    # We can build this efficiently using broadcasting
    V_exp = V[:, np.newaxis, :] # (mp, 1, r)
    W_exp = W[np.newaxis, :, :] # (1, pn, r)
    M_3d = V_exp * W_exp        # (mp, pn, r)
    M = M_3d.reshape(mp * pn, r)
    
    U_new = np.zeros_like(U)
    # Precompute M^T M + lam*I
    MtM = M.T @ M + lam * np.eye(r)
    # Precompute M^T
    MT = M.T
    
    for a in range(nm):
        t_a = T_target[a, :, :].reshape(-1)
        Mtt = MT @ t_a
        try:
            U_new[a, :] = np.linalg.solve(MtM, Mtt)
        except np.linalg.LinAlgError:
            U_new[a, :] = np.linalg.lstsq(M, t_a, rcond=None)[0]
            
    return U_new

def _als_step_V(U, V, W, T_target, n, m, p, lam=1e-6):
    """Update V given U, W, and target tensor T."""
    nm, mp, pn = n * m, m * p, p * n
    r = V.shape[1]
    
    U_exp = U[:, np.newaxis, :] # (nm, 1, r)
    W_exp = W[np.newaxis, :, :] # (1, pn, r)
    M_3d = U_exp * W_exp        # (nm, pn, r)
    M = M_3d.reshape(nm * pn, r)
    
    V_new = np.zeros_like(V)
    MtM = M.T @ M + lam * np.eye(r)
    MT = M.T
    
    for b in range(mp):
        t_b = T_target[:, b, :].reshape(-1)
        Mtt = MT @ t_b
        try:
            V_new[b, :] = np.linalg.solve(MtM, Mtt)
        except np.linalg.LinAlgError:
            V_new[b, :] = np.linalg.lstsq(M, t_b, rcond=None)[0]
            
    return V_new

def _als_step_W(U, V, W, T_target, n, m, p, lam=1e-6):
    """Update W given U, V, and target tensor T."""
    nm, mp, pn = n * m, m * p, p * n
    r = W.shape[1]
    
    U_exp = U[:, np.newaxis, :] # (nm, 1, r)
    V_exp = V[np.newaxis, :, :] # (1, mp, r)
    M_3d = U_exp * V_exp        # (nm, mp, r)
    M = M_3d.reshape(nm * mp, r)
    
    W_new = np.zeros_like(W)
    MtM = M.T @ M + lam * np.eye(r)
    MT = M.T
    
    for c in range(pn):
        t_c = T_target[:, :, c].reshape(-1)
        Mtt = MT @ t_c
        try:
            W_new[c, :] = np.linalg.solve(MtM, Mtt)
        except np.linalg.LinAlgError:
            W_new[c, :] = np.linalg.lstsq(M, t_c, rcond=None)[0]
            
    return W_new

def _random_als_search(n, m, p, target_rank, T_target, max_iters=50, seed=0, deadline=None):
    """
    Run a single randomized ALS search for a decomposition of rank target_rank.
    Returns (U, V, W) if a valid decomposition is found, else None.
    """
    np.random.seed(seed)
    nm, mp, pn = n * m, m * p, p * n
    
    # Initialize with random small values
    # Scaling helps with convergence
    scale = 0.5
    U = np.random.randn(nm, target_rank) * scale
    V = np.random.randn(mp, target_rank) * scale
    W = np.random.randn(pn, target_rank) * scale
    
    # Optionally, initialize with a structured random pattern (e.g. sparse)
    # But dense random is a good start for ALS to find the manifold
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            return None
            
        U = _als_step_U(U, V, W, T_target, n, m, p, lam=1e-8)
        V = _als_step_V(U, V, W, T_target, n, m, p, lam=1e-8)
        W = _als_step_W(U, V, W, T_target, n, m, p, lam=1e-8)
        
        # Periodically snap to half-integers to guide towards discrete solutions
        if it % 10 == 9:
            U = _snap_to_half_integers(U, threshold=0.1)
            V = _snap_to_half_integers(V, threshold=0.1)
            W = _snap_to_half_integers(W, threshold=0.1)
            
        # Check for exact solution
        if _verify_decomp(n, m, p, U, V, W, tol=1e-9):
            return U, V, W
            
    return None

def _block_decomposition_356():
    """
    Construct a valid block decomposition for 3x5x6.
    This serves as a fallback and a high-rank initial guess.
    """
    n, m, p = 3, 5, 6
    
    # Simple schoolbook is rank 90.
    # Let's just use schoolbook to be safe and fast to construct, 
    # then let the search find lower ranks.
    # Or we can use the Strassen block structure from the previous version.
    # For now, schoolbook is a guaranteed valid start.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    T_target = _build_target_tensor(n, m, p)
    nm, mp, pn = n * m, m * p, p * n
    
    # Fallback: Schoolbook decomposition (Rank 90)
    U_school, V_school, W_school = _block_decomposition_356()
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = nm * mp
    
    # Search for lower ranks
    # Lower bound estimate: For matrix multiplication, rank is at least max(n*m, m*p, p*n) 
    # but usually higher. For 3x5x6, known bounds suggest rank < 90.
    # Let's try ranks from 60 to 89.
    
    # We prioritize finding *any* valid decomposition with rank < 90.
    # We iterate ranks from low to high.
    
    # To speed up, we can try a few random restarts for each rank.
    # If we find a valid one, we update best and try lower.
    
    # Start from a reasonable lower bound.
    # Rank 1 is impossible. Rank 90 is schoolbook.
    # Let's try 70, 75, 80, 85, 89 etc.
    
    # Strategy: Try a sequence of ranks.
    # If we find a solution at rank r, we can try r-1, r-2...
    
    # To maximize chance of finding a solution within time:
    # 1. Try rank 89 (likely easy to find with ALS from schoolbook)
    # 2. Try rank 80
    # 3. Try rank 70
    # 4. Try rank 60
    
    # Actually, random initialization ALS is powerful.
    # Let's iterate through a list of candidate ranks.
    
    candidate_ranks = [89, 85, 80, 75, 70, 65, 60]
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        # Try a few random restarts for this rank
        found = False
        for seed in range(5):
            if time.time() > deadline - 5:
                break
                
            result = _random_als_search(n, m, p, r, T_target, max_iters=30, seed=seed, deadline=deadline)
            if result is not None:
                U_r, V_r, W_r = result
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = r
                found = True
                break # Found valid decomposition for this rank, try lower rank
                
        if found:
            # If we found a solution, we can try to go lower immediately
            # The loop structure handles this by continuing to next lower rank
            pass

    # If we found a solution at rank r, we should also try r-1, r-2... 
    # But the candidate_ranks list is sparse.
    # Let's do a fine-grained search around the best found rank if we have time.
    
    if best_rank < nm * mp and time.time() < deadline - 10:
        # Search downwards from best_rank - 1
        for r in range(best_rank - 1, max(50, best_rank - 10), -1):
            if time.time() > deadline - 5:
                break
            
            found = False
            for seed in range(3):
                if time.time() > deadline - 2:
                    break
                result = _random_als_search(n, m, p, r, T_target, max_iters=20, seed=seed, deadline=deadline)
                if result is not None:
                    U_r, V_r, W_r = result
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = r
                    found = True
                    break
            if not found:
                # If we fail at r, it might be possible at r+1, but we already checked r+1.
                # If we fail at r, we might want to try r-1? No, lower rank is harder.
                # If we fail at r, we assume it's not possible or we didn't find it.
                # We continue to r-1.
                pass

    # Final verification and cleanup
    # Ensure we return lists as per interface
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END

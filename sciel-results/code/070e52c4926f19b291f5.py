"""Program for rank-minimal exact tensor decomposition of 3x5x6 matrix multiplication."""
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

def _verify_decomp(n, m, p, U, V, W, atol=1e-9):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    T_comp = np.einsum('ak,bk,ck->abc', U, V, W, optimize=True)
    T_target = _build_target_tensor(n, m, p)
    return np.allclose(T_comp, T_target, atol=atol)

def _schoolbook(n_sub, m_sub, p_sub):
    """Return schoolbook decomposition for n_sub x m_sub x p_sub."""
    rank = n_sub * m_sub * p_sub
    U = np.zeros((n_sub * m_sub, rank), dtype=np.float64)
    V = np.zeros((m_sub * p_sub, rank), dtype=np.float64)
    W = np.zeros((p_sub * n_sub, rank), dtype=np.float64)
    col = 0
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                U[i * m_sub + j, col] = 1.0
                V[j * p_sub + k, col] = 1.0
                W[k * n_sub + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_222():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7. Returns (U, V, W) where U is 4x7, V is 4x7, W is 4x7.
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    W convention: W[k*n+i] where n=2, so W[k*2+i] for C[i,k]
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[2], A[1,1]=U[3]
    # B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[2], B[1,1]=V[3]
    # C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _embed_sub_decomp(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                      i_off, j_off, k_off, n, m, p):
    """
    Embed a sub-decomposition into the full 3x5x6 problem.
    """
    rank = U_sub.shape[1]
    U_full = np.zeros((n * m, rank), dtype=np.float64)
    V_full = np.zeros((m * p, rank), dtype=np.float64)
    W_full = np.zeros((p * n, rank), dtype=np.float64)
    
    for idx in range(n_sub * m_sub):
        i_s = idx // m_sub
        j_s = idx % m_sub
        i_f = i_off + i_s
        j_f = j_off + j_s
        U_full[i_f * m + j_f, :] = U_sub[idx, :]
    
    for idx in range(m_sub * p_sub):
        j_s = idx // p_sub
        k_s = idx % p_sub
        j_f = j_off + j_s
        k_f = k_off + k_s
        V_full[j_f * p + k_f, :] = V_sub[idx, :]
    
    for idx in range(p_sub * n_sub):
        k_s = idx // n_sub
        i_s = idx % n_sub
        k_f = k_off + k_s
        i_f = i_off + i_s
        W_full[k_f * n + i_f, :] = W_sub[idx, :]
    
    return U_full, V_full, W_full

def _block_decomposition(n, m, p, n_splits, m_splits, p_splits):
    """
    Construct a block decomposition for n x m x p with given splits.
    Uses Strassen (rank 7) for 2x2x2 blocks, schoolbook for others.
    """
    U_total = np.zeros((n * m, 0), dtype=np.float64)
    V_total = np.zeros((m * p, 0), dtype=np.float64)
    W_total = np.zeros((p * n, 0), dtype=np.float64)
    
    i_off = 0
    for i_s, n_sub in enumerate(n_splits):
        j_off = 0
        for j_s, m_sub in enumerate(m_splits):
            k_off = 0
            for k_s, p_sub in enumerate(p_splits):
                if n_sub == 2 and m_sub == 2 and p_sub == 2:
                    U_sub, V_sub, W_sub = _strassen_222()
                else:
                    U_sub, V_sub, W_sub = _schoolbook(n_sub, m_sub, p_sub)
                
                U_full, V_full, W_full = _embed_sub_decomp(
                    U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                    i_off, j_off, k_off, n, m, p
                )
                
                if U_total.shape[1] == 0:
                    U_total = U_full
                    V_total = V_full
                    W_total = W_full
                else:
                    U_total = np.hstack([U_total, U_full])
                    V_total = np.hstack([V_total, V_full])
                    W_total = np.hstack([W_total, W_full])
                
                k_off += p_sub
            j_off += m_sub
        i_off += n_sub
    
    return U_total, V_total, W_total

def _block_decomposition_356():
    """
    Construct a block decomposition for 3x5x6 with rank 84.
    Split: n=[2,1], m=[2,2,1], p=[2,2,2]
    """
    n, m, p = 3, 5, 6
    return _block_decomposition(n, m, p, [2, 1], [2, 2, 1], [2, 2, 2])

def _try_alternative_splits(n, m, p, deadline):
    """
    Try different block splits to find a lower-rank decomposition.
    Returns the best (U, V, W) found or None.
    """
    best = None
    best_rank = 90  # schoolbook
    
    # Try various splits
    splits_to_try = [
        # (n_splits, m_splits, p_splits)
        ([2, 1], [2, 2, 1], [2, 2, 2]),  # current: 84
        ([2, 1], [1, 2, 2], [2, 2, 2]),  # permuted m
        ([2, 1], [2, 2, 1], [1, 2, 3]),  # different p split
        ([1, 2], [2, 2, 1], [2, 2, 2]),  # permuted n
        ([2, 1], [2, 1, 2], [2, 2, 2]),  # different m split
        ([2, 1], [2, 2, 1], [3, 3]),     # coarser p split
        ([2, 1], [3, 2], [2, 2, 2]),     # different m split
        ([2, 1], [2, 2, 1], [2, 3, 1]),  # different p split
        ([2, 1], [2, 1, 2], [2, 3, 1]),  # mixed
        ([1, 2], [2, 2, 1], [2, 2, 2]),  # n permuted
        ([2, 1], [1, 1, 3], [2, 2, 2]),  # fine m
        ([2, 1], [2, 2, 1], [1, 1, 4]),  # fine p
        ([1, 2], [1, 2, 2], [2, 2, 2]),  # permuted n, m
        ([2, 1], [2, 2, 1], [2, 2, 1, 1]), # finer p
        ([2, 1], [2, 1, 1, 1], [2, 2, 2]), # finer m
    ]
    
    for n_splits, m_splits, p_splits in splits_to_try:
        if time.time() > deadline - 10:
            break
        if sum(n_splits) != n or sum(m_splits) != m or sum(p_splits) != p:
            continue
        
        U_total, V_total, W_total = _block_decomposition(n, m, p, n_splits, m_splits, p_splits)
        rank = U_total.shape[1]
        
        if rank < best_rank:
            if _verify_decomp(n, m, p, U_total, V_total, W_total):
                best_rank = rank
                best = (U_total.copy(), V_total.copy(), W_total.copy())
    
    return best

def _rank_1_projection_correction(n, m, p, U, V, W, deadline):
    """
    Try to reduce rank by identifying and cancelling linearly dependent terms.
    This is a heuristic approach: check if any column of U, V, W can be removed
    while maintaining the tensor decomposition (approximately).
    
    The idea: if we have a decomposition of rank r, we can try to express one
    rank-1 term as a linear combination of others, reducing the effective rank.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    
    # Try removing each column and see if we can compensate
    # For each column k, try to find alpha such that
    # sum_k' U[:,k'] V[:,k'] W[:,k'] (k' != k) + alpha * U[:,k] V[:,k] W[:,k]
    # still matches the target
    # This is complex, so we use a simpler heuristic:
    # Check if any two columns are nearly linearly dependent in a way that
    # allows cancellation.
    
    # Alternative: use SVD on the flattened tensor to find low-rank structure
    # But this is expensive for large tensors.
    
    # For now, return the input unchanged as a placeholder.
    # A more sophisticated implementation would use iterative refinement.
    return U, V, W, r

def _als_refine(n, m, p, U, V, W, max_iter=50, tol=1e-10, deadline=None):
    """
    Attempt to refine the decomposition using ALS (Alternating Least Squares)
    to find a lower-rank solution. This is a heuristic and may not always succeed.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    
    T_target = _build_target_tensor(n, m, p)
    
    # Build the tensor in flattened form for ALS
    # T[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
    # We can use this to do ALS updates
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Update U: for each a, solve for U[a,:]
        # T[a,:,:] = sum_k U[a,k] * (V[:,k] outer W[:,k])
        # This is a linear system in U[a,:]
        
        # For efficiency, precompute V outer W for each k
        # But this is memory intensive. Instead, use the fact that
        # T[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
        # Fix V, W, solve for U
        # For each a: T[a,:,:] = sum_k U[a,k] * (V[:,k] outer W[:,k])
        # Let M_k = V[:,k] outer W[:,k] (shape m*p x p*n)
        # Then T[a,:,:] = sum_k U[a,k] M_k
        # Flatten: T[a,:,:].flatten() = sum_k U[a,k] M_k.flatten()
        # This is a linear system: A U[a,:] = T[a,:,:].flatten()
        # where A is (m*p*p*n) x r with columns M_k.flatten()
        
        # This is too expensive. Use a simpler approach:
        # Gradient-based refinement is also complex.
        
        # For now, just return the input.
        break
    
    return U, V, W, r

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # 140s budget, 10s safety margin
    
    # Primary: block decomposition with rank 84
    U, V, W = _block_decomposition_356()
    
    # Verify primary
    primary_valid = _verify_decomp(n, m, p, U, V, W)
    primary_rank = U.shape[1] if primary_valid else 999
    
    # Try alternative splits
    if time.time() < deadline - 20:
        better = _try_alternative_splits(n, m, p, deadline)
        if better is not None:
            U_b, V_b, W_b = better
            if U_b.shape[1] < primary_rank:
                U, V, W = U_b, V_b, W_b
                primary_rank = U.shape[1]
    
    # Try rank-1 projection correction (heuristic)
    if time.time() < deadline - 10:
        U_c, V_c, W_c, r_c = _rank_1_projection_correction(n, m, p, U, V, W, deadline)
        if r_c < primary_rank and _verify_decomp(n, m, p, U_c, V_c, W_c):
            U, V, W = U_c, V_c, W_c
            primary_rank = r_c
    
    # Final verification
    if _verify_decomp(n, m, p, U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook
    U_s, V_s, W_s = _schoolbook(n, m, p)
    return {
        "U": U_s.tolist(),
        "V": V_s.tolist(),
        "W": W_s.tolist()
    }

# EVOLVE-BLOCK-END

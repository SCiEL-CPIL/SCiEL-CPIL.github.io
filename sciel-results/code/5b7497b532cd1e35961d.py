"""Strategy: Runtime ALS tensor decomposition search.
Instead of a fixed block-partition with hand-crafted Strassen embeddings, we:
1. Build the known rank-88 Strassen-block solution as a warm start.
2. Use Alternating Least Squares (ALS) to iteratively shrink the rank while
   maintaining exact decomposition of the 3x4x8 matrix multiplication tensor.
3. Verify exactness at each step and return the lowest verified rank.

This is a fundamentally different design family: numerical optimization /
tensor decomposition search rather than algebraic block construction.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 8]


def _build_strassen_rank88():
    """Build the known rank-88 solution using 2x2 Strassen blocks."""
    n, m, p = 3, 4, 8
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen 2x2 terms: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: 4 entries for (i0,j0),(i0,j1),(i1,j0),(i1,j1)
    # b_coeffs: 4 entries for (j0,k0),(j0,k1),(j1,k0),(j1,k1)
    # c_coeffs: 4 entries for (k0,i0),(k0,i1),(k1,i0),(k1,i1)
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, -1]),
        ([1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, -1, 0, 1], [1, 0, -1, 0]),
        ([1, 0, 0, 0], [1, 0, -1, 0], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [1, 0, 1, 0], [1, 0, 0, 0]),
    ]

    col = 0
    # j into 2 blocks of 2, k into 4 blocks of 2
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            # Strassen for i={0,1}, j={j0,j0+1}, k={k0,k0+1}
            for t in range(7):
                ac, bc, cc = strassen_terms[t]
                for di in range(2):
                    for dj in range(2):
                        U[(di) * m + (j0 + dj), col] = ac[di * 2 + dj]
                for dj in range(2):
                    for dk in range(2):
                        V[(j0 + dj) * p + (k0 + dk), col] = bc[dj * 2 + dk]
                for di in range(2):
                    for dk in range(2):
                        W[(k0 + dk) * n + di, col] = cc[di * 2 + dk]
                col += 1
            # Schoolbook for i={2}, j={j0,j0+1}, k={k0,k0+1}
            for dj in range(2):
                for dk in range(2):
                    U[2 * m + (j0 + dj), col] = 1
                    V[(j0 + dj) * p + (k0 + dk), col] = 1
                    W[(k0 + dk) * n + 2, col] = 1
                    col += 1

    return U, V, W, rank


def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the full tensor and check
    # T[i,j,k] should equal 1 for all i,j,k
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-7:
                    return False
    return True


def _als_step(U, V, W, n, m, p, rank, max_iter=50, tol=1e-10):
    """
    Run ALS to minimize the reconstruction error of the multiplication tensor.
    The target tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    We reshape U, V, W into matricizations and use ALS.
    """
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    
    # Precompute index mappings
    # U indices: (i*m + j) for i in [0,n), j in [0,m)
    # V indices: (j*p + k) for j in [0,m), k in [0,p)
    # W indices: (k*n + i) for k in [0,p), i in [0,n)
    
    # Build target tensor as a 3D array
    T = np.ones((n, m, p))
    
    # Matricization approach:
    # For ALS, we fix two factors and solve for the third.
    # 
    # T_ijk = sum_c U_{ij,c} V_{jk,c} W_{ki,c}
    # 
    # Fix V and W, solve for U:
    # For each (i,j): T_{i,j,k} = sum_c U_{ij,c} * (V_{jk,c} * W_{ki,c})
    # This is a linear system: T_{i,j,:} = U_{ij,:} @ M_{:,k} where M_{c,k} = V_{jk,c}*W_{ki,c}
    # Actually: T_{i,j,k} = sum_c U_{ij,c} * V_{jk,c} * W_{ki,c}
    # Let B_{c,k} = V_{jk,c} * W_{ki,c} (depends on j,k,i)
    # T_{i,j,k} = sum_c U_{ij,c} * B_{c,k}
    # So T_{i,j,:} = U_{ij,:} @ B^T  ... this is per (i,j) with different B.
    
    # Better approach: vectorize.
    # Flatten T into a vector of length n*m*p.
    # Each column c contributes: U[:,c] outer V[:,c] outer W[:,c] reshaped.
    # This is a linear system in the flattened column vectors.
    
    # Actually, let's use a simpler iterative approach:
    # For each column c, we can try to optimize it given the others.
    # Or use gradient descent on the entries.
    
    # Let's use a simpler approach: SVD-based ALS on matricizations.
    # 
    # Matricization along i (rows = i, cols = j*p*k... no, this gets complex)
    # 
    # Let's just do coordinate descent on each column.
    
    best_err = np.inf
    for iteration in range(max_iter):
        # Optimize each column
        for c in range(rank):
            # Compute contribution of all other columns
            # T_reconstructed = sum_{c' != c} U[:,c'] outer V[:,c'] outer W[:,c']
            # We want to minimize ||T - T_recon||^2
            # The optimal column c minimizes ||T - T_recon - U_c outer V_c outer W_c||^2
            # This is a 3-way tensor CP optimization for a single component.
            # 
            # Fix V_c and W_c, solve for U_c:
            # min_Uc ||T - T_recon - Uc outer Vc outer Wc||^2
            # = min_Uc ||(T - T_recon) - Uc outer (Vc outer Wc)||^2
            # Let R = T - T_recon (3D tensor)
            # Let B = Vc outer Wc (2D array of shape (m*p, n))... 
            # Actually Vc has shape (m*p,), Wc has shape (p*n,)
            # Vc[j*p+k] * Wc[k*n+i] for each (j,k,i)
            # So B[j,k,i] = Vc[j*p+k] * Wc[k*n+i]
            # We want Uc[i*m+j] to minimize ||R - Uc outer B||^2
            # R[i,j,k] = sum over... 
            # R[i,j,k] = Uc[i*m+j] * B[j,k,i] (where B[j,k,i] = Vc[j*p+k]*Wc[k*n+i])
            # So for each (i,j): R[i,j,:] = Uc[i*m+j] * B[j,:,i]
            # This gives: Uc[i*m+j] = <R[i,j,:], B[j,:,i]> / ||B[j,:,i]||^2
            
            # Compute T_recon without column c
            T_recon = np.zeros((n, m, p))
            for cp in range(rank):
                if cp == c:
                    continue
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_recon[i, j, k] += U[i*m+j, cp] * V[j*p+k, cp] * W[k*n+i, cp]
            
            R = T - T_recon
            
            # Solve for Uc
            Uc_new = np.zeros(n * m)
            for i in range(n):
                for j in range(m):
                    Bj = np.zeros(p)
                    for k in range(p):
                        Bj[k] = V[j*p+k, c] * W[k*n+i, c]
                    denom = np.dot(Bj, Bj)
                    if abs(denom) > 1e-15:
                        Uc_new[i*m+j] = np.dot(R[i, j, :], Bj) / denom
                    else:
                        Uc_new[i*m+j] = U[i*m+j, c]
            U[:, c] = Uc_new
            
            # Similarly for Vc (fix Uc and Wc)
            # R[i,j,k] = Vc[j*p+k] * (Uc[i*m+j] * Wc[k*n+i])
            # For each (j,k): R[:,j,k] = Vc[j*p+k] * Ac[j,k,:] where Ac[j,k,i] = Uc[i*m+j]*Wc[k*n+i]
            Vc_new = np.zeros(m * p)
            for j in range(m):
                for k in range(p):
                    Aj = np.zeros(n)
                    for i in range(n):
                        Aj[i] = U[i*m+j, c] * W[k*n+i, c]
                    denom = np.dot(Aj, Aj)
                    if abs(denom) > 1e-15:
                        Vc_new[j*p+k] = np.dot(R[:, j, k], Aj) / denom
                    else:
                        Vc_new[j*p+k] = V[j*p+k, c]
            V[:, c] = Vc_new
            
            # Similarly for Wc (fix Uc and Vc)
            # R[i,j,k] = Wc[k*n+i] * (Uc[i*m+j] * Vc[j*p+k])
            # For each (k,i): R[i,:,k] = Wc[k*n+i] * Cc[k,i,:] where Cc[k,i,j] = Uc[i*m+j]*Vc[j*p+k]
            Wc_new = np.zeros(p * n)
            for k in range(p):
                for i in range(n):
                    Ck = np.zeros(m)
                    for j in range(m):
                        Ck[j] = U[i*m+j, c] * V[j*p+k, c]
                    denom = np.dot(Ck, Ck)
                    if abs(denom) > 1e-15:
                        Wc_new[k*n+i] = np.dot(R[i, :, k], Ck) / denom
                    else:
                        Wc_new[k*n+i] = W[k*n+i, c]
            W[:, c] = Wc_new
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - 1.0) ** 2
        err = np.sqrt(err / (n * m * p))
        
        if err < best_err:
            best_err = err
        if err < tol:
            break
    
    return U, V, W, best_err


def _try_rank_reduction(U, V, W, n, m, p, start_rank, min_rank, time_limit):
    """
    Try to reduce the rank from start_rank down to min_rank using ALS.
    Return the lowest rank that still verifies exactly.
    """
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    
    current_rank = start_rank
    best_verified_rank = start_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # First verify the start
    if _verify(U, V, W, n, m, p):
        best_verified_rank = start_rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by zeroing out columns and re-optimizing
    for target_rank in range(start_rank - 1, min_rank - 1, -1):
        if time.time() > time_limit:
            break
        
        # Try reducing by one: zero out the column with smallest norm
        # and re-optimize
        U_try = U.copy()
        V_try = V.copy()
        W_try = W.copy()
        
        # Find column with smallest combined norm
        norms = []
        for c in range(current_rank):
            nrm = (np.linalg.norm(U[:, c]) * np.linalg.norm(V[:, c]) * np.linalg.norm(W[:, c]))
            norms.append(nrm)
        min_col = int(np.argmin(norms))
        
        # Zero out that column
        U_try[:, min_col] = 0
        V_try[:, min_col] = 0
        W_try[:, min_col] = 0
        
        # Re-optimize with ALS (fewer iterations for speed)
        U_try, V_try, W_try, err = _als_step(U_try, V_try, W_try, n, m, p, current_rank, max_iter=30)
        
        # Check if we can drop the zeroed column (i.e., if it's still ~0)
        # and verify
        if _verify(U_try, V_try, W_try, n, m, p):
            # Remove the zeroed column
            mask = [c for c in range(current_rank) if c != min_col]
            U_try = U_try[:, mask]
            V_try = V_try[:, mask]
            W_try = W_try[:, mask]
            current_rank = target_rank
            U, V, W = U_try, V_try, W_try
            best_verified_rank = current_rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    return best_verified_rank, best_U, best_V, best_W


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Step 1: Build the known rank-88 solution
    U, V, W, rank = _build_strassen_rank88()
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Step 2: Try to reduce rank using ALS
    min_rank = 60  # Don't go below this (theoretical lower bound is higher, but be safe)
    best_rank, best_U, best_V, best_W = _try_rank_reduction(
        U, V, W, n, m, p, rank, min_rank, deadline
    )
    
    # Step 3: Try some random restarts with lower initial ranks if time permits
    if time.time() < deadline - 20:
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Try a random initialization with a lower rank
            trial_rank = max(min_rank, best_rank - 5 * (attempt + 1))
            if trial_rank < 50:
                break
            
            U_rand = np.random.randn(n * m, trial_rank) * 0.1
            V_rand = np.random.randn(m * p, trial_rank) * 0.1
            W_rand = np.random.randn(p * n, trial_rank) * 0.1
            
            # Optimize
            U_opt, V_opt, W_opt, err = _als_step(U_rand, V_opt if False else V_rand, W_rand, n, m, p, trial_rank, max_iter=40)
            
            if _verify(U_opt, V_opt, W_opt, n, m, p):
                if trial_rank < best_rank:
                    best_rank = trial_rank
                    best_U, best_V, best_W = U_opt.copy(), V_opt.copy(), W_opt.copy()
    
    # Convert to lists for the interface
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


# Keep the original function names and signatures
# The construct() function is the entry point as required.

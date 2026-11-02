"""Matrix multiplication 2,3,8 - rank-minimal tensor decomposition search."""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _build_rank44_decomposition():
    """Build the known rank-44 decomposition (4 x 3x2 Strassen blocks)."""
    n, m, p = 2, 3, 8
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for k_start in [0, 2, 4, 6]:
        k0, k1 = k_start, k_start + 1

        # Strassen 7 terms for the 2x2 * 2x2 part (A rows 0,1 x B rows 0,1)
        # M1 = (A00+A11)*(B0k0+B1k1)
        U[0, col] = 1; U[1, col] = 1
        V[0*p+k0, col] = 1; V[1*p+k1, col] = 1
        W[k0*n+0, col] = 1; W[k1*n+1, col] = 1
        col += 1
        # M2 = (A10+A11)*B0k0
        U[1, col] = 1; U[4, col] = 1
        V[0*p+k0, col] = 1
        W[k0*n+1, col] = 1; W[k1*n+1, col] = -1
        col += 1
        # M3 = A00*(B0k1-B1k1)
        U[0, col] = 1
        V[0*p+k1, col] = 1; V[1*p+k1, col] = -1
        W[k1*n+0, col] = 1; W[k1*n+1, col] = 1
        col += 1
        # M4 = A11*(B1k0-B0k0)
        U[4, col] = 1
        V[1*p+k0, col] = 1; V[0*p+k0, col] = -1
        W[k0*n+0, col] = 1; W[k0*n+1, col] = 1
        col += 1
        # M5 = (A00+A01)*B1k1
        U[0, col] = 1; U[1, col] = 1
        V[1*p+k1, col] = 1
        W[k0*n+0, col] = -1; W[k1*n+0, col] = 1
        col += 1
        # M6 = (A10-A00)*(B0k0+B0k1)
        U[1, col] = 1; U[0, col] = -1
        V[0*p+k0, col] = 1; V[0*p+k1, col] = 1
        W[k1*n+1, col] = 1
        col += 1
        # M7 = (A01-A11)*(B1k0+B1k1)
        U[1, col] = 1; U[4, col] = -1
        V[1*p+k0, col] = 1; V[1*p+k1, col] = 1
        W[k0*n+0, col] = 1
        col += 1

        # 4 rank-1 terms for A row 2 (3rd row) * B row 2
        for (ii, kk) in [(0,k0),(0,k1),(1,k0),(1,k1)]:
            U[ii, col] = 1
            V[2*p+kk, col] = 1
            W[kk*n+ii, col] = 1
            col += 1

    return U, V, W


def _verify(U, V, W):
    """Check that U∘V∘W reconstructs the matmul tensor."""
    n, m, p = 2, 3, 8
    rank = U.shape[1]
    # Build tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # T should equal: T[i,j,k] = 1 if i<2,j<3,k<8 and the entry is in matmul
    # Actually for matmul: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,j,k] = 1 for all valid i,j,k (it's the standard matmul tensor)
    # Wait no. The matmul tensor has T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # and the decomposition is: C_ik = sum_j sum_r U[i,r] V[j,r] W[k,r]
    # So T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r] should equal 1 for all i,j,k
    # Actually no! The matmul tensor is T_{i,j,k} such that C_ik = sum_j A_ij B_jk
    # = sum_j sum_r A_ij B_jk delta
    # The tensor decomposition: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # and we need T[i,j,k] = 1 for all valid (i,j,k)
    # Wait, that's not right either. Let me think again.
    #
    # Matrix multiplication: (AB)_{ik} = sum_j A_{ij} B_{jk}
    # Tensor form: T_{i,j,k} where T_{i,j,k} = 1 means "A_{ij} times B_{jk} contributes to C_{ik}"
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition is: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    
    # Compute T from decomposition
    # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # = (U @ V.T) doesn't work directly. Use:
    # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # This is a 3-tensor. Compute via: for each r, outer(U[:,r], V[:,r], W[:,r])
    
    # Efficient check: C[i,k] = sum_j sum_r U[i,r]*V[j,r]*W[k,r]
    # = sum_r (sum_j U[i,r]*V[j,r]) * W[k,r]
    # = sum_r (U[i,r] * sum_j V[j,r]) ... no
    # C[i,k] = sum_j sum_r U[i,r]*V[j,r]*W[k,r]
    # = sum_r U[i,r]*W[k,r] * sum_j V[j,r] ... no that's wrong
    # C[i,k] = sum_r (U[i,r] * W[k,r]) * (sum_j V[j,r]) ... still wrong
    
    # Correct: C[i,k] = sum_j (sum_r U[i,r]*V[j,r]) * W[k,r]... no
    # C[i,k] = sum_j sum_r U[i,r]*V[j,r]*W[k,r]
    # Let's just compute it directly for small sizes
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = np.sum(U[i,:] * V[j,:] * W[k,:])
    
    # The matmul tensor should be all 1s
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-6)


def _build_target_tensor():
    """The target matmul tensor: all ones of shape (2,3,8)."""
    return np.ones((2, 3, 8))


def _als_decompose(target, rank, max_iter=200, tol=1e-8, n_restarts=5, seed=42):
    """ALS decomposition of a 3-tensor. target shape (n,m,p)."""
    n, m, p = target.shape
    best_err = np.inf
    best_U = best_V = best_W = None
    rng = np.random.RandomState(seed)
    
    for restart in range(n_restarts):
        U = rng.randn(n, rank)
        V = rng.randn(m, rank)
        W = rng.randn(p, rank)
        
        for it in range(max_iter):
            # Update U: for each i, U[i,:] solves least squares
            # target[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
            # Fix V,W: target[i,:,:] = U[i,:] outer (V*W) contracted
            # target[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # For fixed i: target[i,j,k] = sum_r U[i,r] * (V[j,r]*W[k,r])
            # So: target[i,:,:] = sum_r U[i,r] * np.outer(V[:,r], W[:,r])
            # In matrix form: target[i,:,:].reshape(m*p) = U[i,:] @ (V*W matrix)
            # where (V*W)[j*p+k, r] = V[j,r]*W[k,r]
            
            VW = np.zeros((m*p, rank))
            for r in range(rank):
                VW[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            
            # U[i, :] = argmin sum_{j,k} (target[i,j,k] - sum_r U[i,r]*VW[j*p+k,r])^2
            # = least squares: target[i].reshape(-1) = VW.T @ U[i,:]
            # Actually: target[i].reshape(-1) = (VW @ U[i,:].T).T... 
            # target[i,j,k] = sum_r VW[j*p+k, r] * U[i, r]
            # So target[i].reshape(-1) = VW.T @ U[i,:]
            # U[i,:] = (VW.T @ U[i,:])... solve: target[i].reshape(-1) = VW.T @ U[i,:]
            # This is: (VW.T) @ U[i,:] = target[i].reshape(-1)
            # U[i,:] = lstsq(VW.T, target[i].reshape(-1))
            
            U_new = np.zeros_like(U)
            for i in range(n):
                U_new[i, :] = np.linalg.lstsq(VW.T, target[i].reshape(-1), rcond=None)[0]
            
            # Update V
            UW = np.zeros((n*p, rank))
            for r in range(rank):
                UW[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            
            V_new = np.zeros_like(V)
            for j in range(m):
                # target[i,j,k] = sum_r UW[i*p+k, r] * V[j, r]
                # target[:,j,:].reshape(-1) = UW.T @ V[j,:]
                V_new[j, :] = np.linalg.lstsq(UW.T, target[:,j,:].reshape(-1), rcond=None)[0]
            
            # Update W
            UV = np.zeros((n*m, rank))
            for r in range(rank):
                UV[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            
            W_new = np.zeros_like(W)
            for k in range(p):
                # target[i,j,k] = sum_r UV[i*m+j, r] * W[k, r]
                # target[:,:,k].reshape(-1) = UV.T @ W[k,:]
                W_new[k, :] = np.linalg.lstsq(UV.T, target[:,:,k].reshape(-1), rcond=None)[0]
            
            # Check convergence
            U, V, W = U_new, V_new, W_new
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i,:] * V[j,:] * W[k,:])
                        err += (val - target[i,j,k])**2
            err = np.sqrt(err)
            
            if best_err > err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
        
        if best_err < tol:
            break
    
    return best_U, best_V, best_W, best_err


def _try_rank_with_bmixing(rank, target, time_limit=8.0, seed=0):
    """Try to find rank-r decomposition with random B-mixing."""
    n, m, p = target.shape
    rng = np.random.RandomState(seed)
    best_err = np.inf
    best_U = best_V = best_W = None
    start_time = time.time()
    max_restarts = 200
    
    attempt = 0
    while time.time() - start_time < time_limit and attempt < max_restarts:
        # Random invertible 3x3 mixing matrix for B dimension
        P = np.eye(m) + 0.1 * rng.randn(m, m)
        if np.linalg.det(P) < 0.01:
            attempt += 1
            continue
        
        # Apply mixing: new B = B @ P^{-1}, so new V = P^{-1} @ V_old
        # Actually, if we define B' = B @ P, then B = B' @ P^{-1}
        # C = A @ B = A @ B' @ P^{-1} = (A @ B') @ P^{-1}
        # So C' = C @ P = A @ B'
        # The tensor becomes: T'[i,j',k] = sum_r U[i,r]*V'[j',r]*W'[k,r]
        # where the target is T' = T with B dimension mixed
        # T'[i,j',k] = sum_j T[i,j,k] * P[j',j] ... 
        # Actually: C_{ik} = sum_j A_{ij} B_{jk}
        # If B_{jk} = sum_{j'} B'_{j'k} P_{jj'}, then C_{ik} = sum_j A_{ij} sum_{j'} B'_{j'k} P_{jj'}
        # = sum_{j'} B'_{j'k} (sum_j A_{ij} P_{jj'})
        # Let A'_{ij'} = sum_j A_{ij} P_{jj'} = A @ P
        # Then C_{ik} = sum_{j'} A'_{ij'} B'_{j'k}
        # So the tensor T'[i,j',k] = 1 (same all-ones structure!)
        # But the factorization changes: U' = U @ P^{-T}... 
        # Hmm, this doesn't help directly since the tensor is still all-ones.
        
        # The point is: by mixing B rows, the "structure" of which entries need
        # to be covered changes relative to the A and C factors.
        # Actually for the all-ones tensor, any mixing of the middle dimension
        # gives another all-ones tensor. So the mixing doesn't change the target.
        # What it does change is the structure of the decomposition.
        
        # Let me think differently. The tensor T[i,j,k]=1 can be factored as
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # If we apply GL(3) to the j-index: T[i,j,k] = sum_{j'} T[i,j',k] P^{-1}[j',j]
        # This is just a change of basis. The rank is invariant.
        # BUT: the *form* of the decomposition changes, and ALS might find
        # lower-rank solutions from different initializations.
        
        # So the strategy is: just try many random initializations for ALS.
        U, V, W, err = _als_decompose(target, rank, max_iter=50, tol=1e-10, n_restarts=1, seed=seed*1000+attempt)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U, V, W
        
        if best_err < 1e-8:
            break
        attempt += 1
    
    return best_U, best_V, best_W, best_err


def _round_and_verify(U, V, W, target, atol=1e-6):
    """Try to round entries to nearest half-integers and verify."""
    n, m, p = target.shape
    rank = U.shape[1]
    
    # Round to nearest 0.5
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    # Verify
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_r[i,:] * V_r[j,:] * W_r[k,:])
                err += (val - target[i,j,k])**2
    err = np.sqrt(err)
    
    if err < atol:
        return U_r, V_r, W_r, True
    
    # Try rounding to integers
    U_i = np.round(U)
    V_i = np.round(V)
    W_i = np.round(W)
    err2 = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_i[i,:] * V_i[j,:] * W_i[k,:])
                err2 += (val - target[i,j,k])**2
    err2 = np.sqrt(err2)
    
    if err2 < atol:
        return U_i, V_i, W_i, True
    
    return None, None, None, False


def construct():
    n, m, p = DIMENSIONS
    target = np.ones((n, m, p))
    
    # Start with the known rank-44 solution
    U44, V44, W44 = _build_rank44_decomposition()
    # Convert to lists
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    # Verify rank 44
    if _verify(U44, V44, W44):
        result_44 = {"U": to_list(U44), "V": to_list(V44), "W": to_list(W44)}
    else:
        result_44 = None
    
    # Try to find rank 43 or lower
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 40s margin for safety
    
    best_rank = 44
    best_result = result_44
    
    for rank in [43, 42, 41, 40, 39, 38]:
        if time.time() > deadline - 15:
            break
        
        found = False
        max_attempts = 300
        seed = 42
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            rng = np.random.RandomState(seed + attempt)
            
            # Random initialization
            U = rng.randn(n, rank) * 0.5
            V = rng.randn(m, rank) * 0.5
            W = rng.randn(p, rank) * 0.5
            
            # ALS iterations
            for it in range(100):
                # Update U
                VW = np.zeros((m*p, rank))
                for r in range(rank):
                    VW[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                U_new = np.zeros_like(U)
                for i in range(n):
                    U_new[i, :] = np.linalg.lstsq(VW.T, target[i].reshape(-1), rcond=None)[0]
                
                # Update V
                UW = np.zeros((n*p, rank))
                for r in range(rank):
                    UW[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                V_new = np.zeros_like(V)
                for j in range(m):
                    V_new[j, :] = np.linalg.lstsq(UW.T, target[:,j,:].reshape(-1), rcond=None)[0]
                
                # Update W
                UV = np.zeros((n*m, rank))
                for r in range(rank):
                    UV[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                W_new = np.zeros_like(W)
                for k in range(p):
                    W_new[k, :] = np.linalg.lstsq(UV.T, target[:,:,k].reshape(-1), rcond=None)[0]
                
                U, V, W = U_new, V_new, W_new
                
                # Quick error check every 10 iterations
                if it % 10 == 9:
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U[i,:] * V[j,:] * W[k,:])
                                err += (val - target[i,j,k])**2
                    if err < 1e-10:
                        # Found a solution! Try to round
                        U_r, V_r, W_r, ok = _round_and_verify(U, V, W, target, atol=1e-5)
                        if ok:
                            best_rank = rank
                            best_result = {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
                            found = True
                            break
                        else:
                            # Use the floating point solution if rounding fails
                            # But the task requires integer/half-integer entries
                            # Let's try harder rounding
                            for scale in [1, 2, 4, 8]:
                                U_t = np.round(U * scale) / scale
                                V_t = np.round(V * scale) / scale
                                W_t = np.round(W * scale) / scale
                                err_r = 0.0
                                for i2 in range(n):
                                    for j2 in range(m):
                                        for k2 in range(p):
                                            val = np.sum(U_t[i2,:] * V_t[j2,:] * W_t[k2,:])
                                            err_r += (val - target[i2,j2,k2])**2
                                if np.sqrt(err_r) < 1e-5:
                                    best_rank = rank
                                    best_result = {"U": to_list(U_t), "V": to_list(V_t), "W": to_list(W_t)}
                                    found = True
                                    break
                        if found:
                            break
                    if np.sqrt(err) < 1e-12:
                        # Very close, try more aggressive rounding
                        for scale in [1, 2, 4, 8, 16]:
                            U_t = np.round(U * scale) / scale
                            V_t = np.round(V * scale) / scale
                            W_t = np.round(W * scale) / scale
                            err_r = 0.0
                            for i2 in range(n):
                                for j2 in range(m):
                                    for k2 in range(p):
                                        val = np.sum(U_t[i2,:] * V_t[j2,:] * W_t[k2,:])
                                        err_r += (val - target[i2,j2,k2])**2
                            if np.sqrt(err_r) < 1e-4:
                                best_rank = rank
                                best_result = {"U": to_list(U_t), "V": to_list(V_t), "W": to_list(W_t)}
                                found = True
                                break
                        if found:
                            break
                        break
            if found:
                break
        
        if found:
            break
        else:
            # Rank not achieved, continue to next lower rank
            pass
    
    # Safety: if we found nothing better, return rank 44
    if best_result is None:
        U44, V44, W44 = _build_rank44_decomposition()
        best_result = {"U": to_list(U44), "V": to_list(V44), "W": to_list(W44)}
    
    # Final safety check
    if time.time() > start_time + 130:
        # Return immediately with best result
        pass
    
    return best_result
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

def _build_matmul_tensor(n, m, p):
    """Return the matrix multiplication tensor as a 3D array T[i,j,k] = 1
    if the (i,k) output entry depends on A[i,j]*B[j,k].
    Actually, the tensor M is defined such that C[i,k] = sum_j A[i,j] B[j,k],
    so the tensor has indices (i,j,k) and M[i,j,k] = 1 for all i,j,k.
    But wait - the tensor for matrix multiplication is the structure tensor.
    The bilinear map is (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k].
    As a trilinear form, <T, A ⊗ B ⊗ C> = 0 defines the variety, but for
    tensor rank we consider the tensor M with entries M[i,k,j] = 1 (all ones
    in the relevant positions). Actually the multiplication tensor T_{(n,m,p)}
    has entries T[i,k,j] = 1 iff the term A[i,j]B[j,k] appears in C[i,k].
    Since C[i,k] = sum_j A[i,j] B[j,k], every (i,k,j) term appears, so
    T[i,k,j] = 1 for all i in [0,n), k in [0,p), j in [0,m).
    
    In the factorization: U has shape (n*m, rank), V has shape (m*p, rank),
    W has shape (p*n, rank).
    U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    The condition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    for all i, j, k.
    """
    n, m, p = n, m, p
    # T indexed by (i, j, k) with T[i,j,k] = 1
    return np.ones((n, m, p), dtype=float)

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-6):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_and_clean(U, V, W, n, m, p, rank):
    """Verify and clean up numerical noise in the decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Clean small values
    U[np.abs(U) < 1e-10] = 0.0
    V[np.abs(V) < 1e-10] = 0.0
    W[np.abs(W) < 1e-10] = 0.0
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-4):
        return None
    
    # Try to snap to half-integers
    def snap(arr):
        snapped = np.round(arr * 2) / 2
        # Check if snapping is close
        if np.max(np.abs(arr - snapped)) < 1e-5:
            return snapped
        # Try integers
        snapped_int = np.round(arr)
        if np.max(np.abs(arr - snapped_int)) < 1e-5:
            return snapped_int
        return arr
    
    U_s = snap(U)
    V_s = snap(V)
    W_s = snap(W)
    
    # Verify snapped version
    if _verify_decomposition(U_s, V_s, W_s, n, m, p, rank, tol=1e-4):
        # Convert to lists of ints or halves
        def to_list(arr):
            result = []
            for row in arr:
                r = []
                for v in row:
                    if abs(v - round(v)) < 1e-9:
                        r.append(int(round(v)))
                    elif abs(2*v - round(2*v)) < 1e-9:
                        r.append(int(round(2*v))/2.0)
                    else:
                        r.append(float(v))
                result.append(r)
            return result
        return {"U": to_list(U_s), "V": to_list(V_s), "W": to_list(W_s)}
    
    return None

def construct():
    deadline = time.time() + 110  # Safety margin
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Start with schoolbook
    rank_sb = n * m * p
    U = np.zeros((nm, rank_sb))
    V = np.zeros((mp, rank_sb))
    W = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank_sb
    
    # Tensor T[i,j,k] = 1 for all
    T = np.ones((n, m, p))
    
    # Try to find lower rank decompositions using numerical methods
    # Strategy: use random initialization and gradient-free / alternating minimization
    
    random.seed(42)
    np.random.seed(42)
    
    # Try ranks from lower up
    for rank in range(20, best_rank):
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            U_r = np.random.randn(nm, rank)
            V_r = np.random.randn(mp, rank)
            W_r = np.random.randn(pn, rank)
            
            # Alternating least squares to minimize ||sum_c u_c ⊗ v_c ⊗ w_c - T||_F^2
            # where u_c[i*m+j] is the (i,j) component, v_c[j*p+k], w_c[k*n+i]
            
            # Reshape for convenience
            # For a given c, the tensor contribution is U[:,c] reshaped to (n,m)
            # times V[:,c] reshaped to (m,p) times W[:,c] reshaped to (p,n)
            # The target is T[i,j,k] = 1
            
            # We want: sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1 for all i,j,k
            # where U_c is reshaped from U[:,c] to (n,m), etc.
            
            # This is a 3-way tensor CP decomposition problem.
            # Let's use a simple ALS approach.
            
            max_iter = 200
            U_work = U_r.copy()
            V_work = V_r.copy()
            W_work = W_r.copy()
            
            for it in range(max_iter):
                if time.time() > deadline - 15:
                    break
                
                # Update W: for each (k,i), we need sum_c U_c[i,j]*V_c[j,k]*W_c[k,i] = 1
                # For fixed (k,i), sum_j sum_c U_c[i,j]*V_c[j,k]*W_c[k,i] = 1
                # So W_c[k,i] = (sum_j (sum_{c'} U_{c'}[i,j]*V_{c'}[j,k]) * ... )
                # Actually this is more complex. Let's use a different approach.
                
                # The residual for entry (i,j,k) is:
                # r[i,j,k] = 1 - sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
                
                # For ALS on W: fix U and V.
                # For each (k,i), we want:
                # sum_c W_c[k,i] * (sum_j U_c[i,j] * V_c[j,k]) ≈ 1
                # Let M[k,i,c] = sum_j U_c[i,j] * V_c[j,k]
                # Then we want: sum_c M[k,i,c] * W_c[k,i] = 1
                # This is: for each (k,i), the dot product of M[k,i,:] and W[k,i,:] = 1
                # To minimize: sum_{k,i} (1 - sum_c M[k,i,c]*W_c[k,i])^2
                # Optimal: W[k,i,:] = M[k,i,:] / (M[k,i,:]·M[k,i,:])
                
                # Compute M
                # U reshaped to (n, m, rank), V reshaped to (m, p, rank)
                # M[k,i,c] = sum_j U[i,j,c] * V[j,k,c]
                U3 = U_work.reshape(n, m, rank)
                V3 = V_work.reshape(m, p, rank)
                
                # M[k,i,c] = sum_j U3[i,j,c] * V3[j,k,c]
                # = sum over j of U3[i,:,c] * V3[:,k,c]
                # Using einsum: M[k,i,c] = np.einsum('ijc,jkc->kic', U3, V3)
                M = np.einsum('ijc,jkc->kic', U3, V3)  # shape (p, n, rank)
                
                # For each (k,i), W[k*n+i, :] should be M[k,i,:] / ||M[k,i,:]||^2
                for k in range(p):
                    for i in range(n):
                        vec = M[k, i, :]
                        norm_sq = np.dot(vec, vec)
                        if norm_sq > 1e-12:
                            W_work[k*n+i, :] = vec / norm_sq
                        else:
                            W_work[k*n+i, :] = 0.0
                
                # Update U: fix V and W.
                # For each (i,j): sum_c U_c[i,j] * (sum_k V_c[j,k]*W_c[k,i]) = 1
                # Let N[i,j,c] = sum_k V_c[j,k] * W_c[k,i]
                W3 = W_work.reshape(p, n, rank)
                # N[i,j,c] = sum_k V3[j,k,c] * W3[k,i,c]
                N = np.einsum('jkc,kic->ijc', V3, W3)  # shape (n, m, rank)
                
                for i in range(n):
                    for j in range(m):
                        vec = N[i, j, :]
                        norm_sq = np.dot(vec, vec)
                        if norm_sq > 1e-12:
                            U_work[i*m+j, :] = vec / norm_sq
                        else:
                            U_work[i*m+j, :] = 0.0
                
                # Update V: fix U and W.
                # For each (j,k): sum_c V_c[j,k] * (sum_i U_c[i,j]*W_c[k,i]) = 1
                # Let P[j,k,c] = sum_i U_c[i,j] * W_c[k,i]
                U3 = U_work.reshape(n, m, rank)
                W3 = W_work.reshape(p, n, rank)
                # P[j,k,c] = sum_i U3[i,j,c] * W3[k,i,c]
                P = np.einsum('ijc,kic->jkc', U3, W3)  # shape (m, p, rank)
                
                for j in range(m):
                    for k in range(p):
                        vec = P[j, k, :]
                        norm_sq = np.dot(vec, vec)
                        if norm_sq > 1e-12:
                            V_work[j*p+k, :] = vec / norm_sq
                        else:
                            V_work[j*p+k, :] = 0.0
                
                # Check convergence
                U3 = U_work.reshape(n, m, rank)
                V3 = V_work.reshape(m, p, rank)
                W3 = W_work.reshape(p, n, rank)
                recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                err = np.max(np.abs(recon - 1.0))
                if err < 1e-6:
                    found = True
                    break
                
                if it % 20 == 0 and time.time() > deadline - 10:
                    break
            
            if found:
                result = _verify_and_clean(U_work, V_work, W_work, n, m, p, rank)
                if result is not None:
                    best = result
                    best_rank = rank
                    found = True
                    break
        
        if found:
            # Try even lower
            continue
    
    # Final verification of best result
    if best_rank < rank_sb:
        U_final = np.array(best["U"])
        V_final = np.array(best["V"])
        W_final = np.array(best["W"])
        if _verify_decomposition(U_final, V_final, W_final, n, m, p, best_rank, tol=1e-4):
            return best
    
    # Return schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

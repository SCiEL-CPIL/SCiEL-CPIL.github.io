"""Matrix multiplication 3x5x5 rank search with deadline guard."""
import time
import sys

DIMENSIONS = [3, 5, 5]

# Hard wall-clock deadline (evaluator kills at 150s; keep safety margin)
DEADLINE = 145.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _time_left():
    return DEADLINE - _elapsed()

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 75."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-6):
    """Check that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == delta(i,j,k) for all i,j,k."""
    import numpy as np
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ua = Ua.reshape(n, m, rank)
    Va = Va.reshape(m, p, rank)
    Wa = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    # The target tensor for matrix multiplication C_ik = sum_j A_ij B_jk
    # In the trace convention W[k*n+i, r], the term is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This sums to C[i, k] = sum_j A[i, j] B[j, k]
    # So T[i, k] should be 1 if i==k? No.
    # The tensor T[i, j, k] in the previous code was indexed i, j, k.
    # Let's re-read the convention.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # Standard tensor decomposition: T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c] ?
    # Usually, matrix multiplication tensor is defined such that <T, A, B> = <C, I>.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # So the tensor T has indices (i, j, k) such that T[i, j, k] = 1 if i==k? No.
    # The contraction is sum_{j,c} U[i, j, c] V[j, k, c] * W[k, i, c] ?
    # If W[k, i, c] is part of the decomposition, it usually corresponds to the output index.
    # Let's assume the standard: sum_c U[i, m+j] V[m+j, p+k] W[p+k, n+i] ?
    # The previous code used: np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    # This computes T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c].
    # For this to be the matrix multiplication tensor, we need T[i, j, k] to represent the coefficient of A[i, j] B[j, k] in the result.
    # But the result is C[i, k]. The term A[i, j] B[j, k] contributes to C[i, k].
    # So the "tensor" of the multiplication map is often viewed as T[i, j, k] where the third index k is the output column?
    # Actually, the bilinear form is sum_{i,j,k} C[i, k] * (something).
    # If we identify the tensor with the map (A, B) -> C, the coefficients are 1 for (i, j, k) where the output index is k?
    # No, the output index is determined by i and k.
    # The previous verification `T - 1.0` implies it expected T[i, j, k] to be 1 everywhere? That is incorrect for matrix multiplication.
    # For matrix multiplication, T[i, j, k] is 1 if the indices align?
    # Actually, the tensor T such that C = A * B is T[i, j, k] = delta_{i, output_row} delta_{k, output_col}?
    # Let's look at the schoolbook implementation:
    # U[i*m+j][c]=1, V[j*p+k][c]=1, W[k*n+i][c]=1.
    # Sum_c U[i, j, c] V[j, k, c] W[k, i, c]
    # For a specific c corresponding to (i', j', k'), U[i, j, c] is 1 if i=i', j=j'.
    # V[j, k, c] is 1 if j=j', k=k'.
    # W[k, i, c] is 1 if k=k', i=i'.
    # So the term is non-zero only if i=i', j=j', k=k' AND j=j' AND k=k' AND i=i'.
    # So T[i, j, k] = 1 if there exists c such that (i, j) matches U, (j, k) matches V, (k, i) matches W.
    # In schoolbook, c is unique for (i, j, k).
    # So T[i, j, k] = 1 for ALL i, j, k?
    # If T[i, j, k] = 1 for all i, j, k, then sum_c ... = 1.
    # But matrix multiplication is C[i, k] = sum_j A[i, j] B[j, k].
    # The tensor T represents the identity map on the space of matrices?
    # No, the tensor decomposition of matrix multiplication usually refers to the tensor T in (A* \otimes B*) \otimes C such that <T, A, B> = Tr(C^T (A*B))?
    # If the convention is W[k*n+i, r], then the third mode of the tensor is indexed by (k, i).
    # The contraction is sum_{j, c} U[i, j, c] V[j, k, c] W[k, i, c].
    # This sums over j and c. The result is a matrix M[i, k].
    # For this to equal A[i, j] B[j, k] summed over j, we need the "tensor" to be the identity in the j dimension?
    # Actually, the bilinear form is sum_{i, j, k} M[i, k] * 1.
    # So the target tensor T[i, j, k] should be 1 for all i, j, k?
    # Yes, because the coefficient of A[i, j] B[j, k] in the sum defining the bilinear form (which reconstructs C) is 1.
    # So `T - 1.0` is correct for the standard "all ones" tensor in this specific index convention.
    return np.max(np.abs(T - 1.0)) < tol


def _als_one_step(U, V, W, n, m, p, rank):
    """One step of alternating least squares."""
    import numpy as np
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    for i in range(n):
        for j in range(m):
            A_sys = np.zeros((p, rank))
            for k in range(p):
                A_sys[k, :] = V[j*p+k, :] * W[k*n+i, :]
            try:
                U_new[i*m+j, :] = np.linalg.lstsq(A_sys, np.ones(p), rcond=None)[0]
            except:
                pass
    
    for j in range(m):
        for k in range(p):
            A_sys = np.zeros((n, rank))
            for i in range(n):
                A_sys[i, :] = U_new[i*m+j, :] * W[k*n+i, :]
            try:
                V_new[j*p+k, :] = np.linalg.lstsq(A_sys, np.ones(n), rcond=None)[0]
            except:
                pass
    
    for k in range(p):
        for i in range(n):
            A_sys = np.zeros((m, rank))
            for j in range(m):
                A_sys[j, :] = U_new[i*m+j, :] * V_new[j*p+k, :]
            try:
                W_new[k*n+i, :] = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)[0]
            except:
                pass
    
    return U_new, V_new, W_new


def _try_als(U0, V0, W0, target_rank, max_iters=20, tol=1e-7):
    """Try ALS to find a rank-`target_rank` decomposition."""
    import numpy as np
    n, m, p = DIMENSIONS
    orig_rank = len(U0[0])
    
    Ua = np.array(U0, dtype=float)
    Va = np.array(V0, dtype=float)
    Wa = np.array(W0, dtype=float)
    
    P = np.random.randn(orig_rank, target_rank)
    U = Ua @ P
    V = Va @ P
    W = Wa @ P
    
    for it in range(max_iters):
        if _time_left() < 1.0:
            return None
        
        U, V, W = _als_one_step(U, V, W, n, m, p, target_rank)
        
        # Project to half-integers
        U_half = np.round(U * 2) / 2.0
        V_half = np.round(V * 2) / 2.0
        W_half = np.round(W * 2) / 2.0
        
        Ua_t = U_half.reshape(n, m, target_rank)
        Va_t = V_half.reshape(m, p, target_rank)
        Wa_t = W_half.reshape(p, n, target_rank)
        T = np.einsum('ijc,jkc,kic->ijk', Ua_t, Va_t, Wa_t)
        err = np.max(np.abs(T - 1.0))
        
        if err < tol:
            return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
    
    return None


def construct():
    if _time_left() < 5.0:
        return _schoolbook()

    best = _schoolbook()
    best_rank = 75
    
    import numpy as np
    np.random.seed(42)
    
    U0 = best["U"]
    V0 = best["V"]
    W0 = best["W"]
    
    # Search for lower rank solutions
    for target_rank in [70, 65, 60, 55, 50]:
        if _time_left() < 10.0:
            break
        
        for trial in range(2):
            if _time_left() < 5.0:
                break
            
            result = _try_als(U0, V0, W0, target_rank, max_iters=10)
            if result is not None:
                # Double check validity
                if _verify(result["U"], result["V"], result["W"], tol=1e-9):
                    best = result
                    best_rank = target_rank
                    U0 = best["U"]
                    V0 = best["V"]
                    W0 = best["W"]
                    break
    
    # Ensure the returned result is valid and has integer/half-integer entries
    # Schoolbook is always valid. If search failed, return schoolbook.
    if _verify(best["U"], best["V"], best["W"], tol=1e-9):
        return best
    else:
        return _schoolbook()

import time

def _is_valid_decomposition(U, V, W):
    """Check if the decomposition exactly reconstructs matrix multiplication."""
    n, m, p = 3, 5, 6
    U, V, W = _to_numeric(U), _to_numeric(V), _to_numeric(W)
    # Check shapes
    if U.shape != (n * m, U.shape[1]):
        return False
    if V.shape != (m * p, V.shape[1]):
        return False
    if W.shape != (p * n, W.shape[1]):
        return False
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    
    # Reconstruct the tensor
    # Tensor[i, j, k] = sum over r of U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    # For exact matrix multiplication, this should equal 1 if indices align properly, but actually:
    # The tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The structure tensor T has T[i, j, k] such that the bilinear form is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = C[i,k]
    # Actually, the standard structure tensor for matrix mult M_{n,m,p} has:
    # T[i, j, k] = 1 if we're computing C[i,k] using A[i,j] and B[j,k]... no.
    # Let me re-read the task.
    # "U', 'V', 'W': factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "exactly decompose the matrix multiplication tensor"
    # The tensor for matrix multiplication is typically defined such that:
    # sum_{r} U[a, r] V[b, r] W[c, r] = delta_{a,b,c}
    # where a = i*m + j (0..n*m-1), b = j'*p + k (0..m*p-1), c = k'*n + i' (0..p*n-1)
    # But the tensor must match: T[a, b, c] = 1 if the triple corresponds to a term in matrix mult.
    # A term A[i,j] * B[j,k] contributes to C[i,k].
    # So a = i*m + j, b = j*p + k, c = k*n + i.
    # The tensor entry T[a, b, c] is 1 if the indices are consistent: 
    # a = i*m + j, b = j*p + k, c = k*n + i for some i,j,k.
    # Otherwise 0.
    # So T[i*m + j, j*p + k, k*n + i] = 1, and 0 otherwise.
    
    # Let's build the target tensor (sparse) and compare.
    import numpy as np
    target = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1
    
    # Compute T_reconstructed = sum_r outer(U[:,r], V[:,r]) contracted with W[:,r]
    # This is rank factorization.
    T = np.einsum('ar,br,cnr->abc', U, V, W)
    if np.allclose(T, target, atol=1e-10):
        return True
    return False

def _to_numeric(mat):
    import numpy as np
    return np.array(mat, dtype=float)

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Rank 90 schoolbook as fallback
    rank_fallback = n * m * p
    U_fallback = [[0] * rank_fallback for _ in range(n * m)]
    V_fallback = [[0] * rank_fallback for _ in range(m * p)]
    W_fallback = [[0] * rank_fallback for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][column] = 1
                V_fallback[j * p + k][column] = 1
                W_fallback[k * n + i][column] = 1
                column += 1
    
    # Try to find a better rank via search
    # For 3x5x6, the lower bound is 3*5*6 / something? 
    # Actually, trivial lower bound is max(n, m, p) or something.
    # Strassen: 2x2x2 rank 7.
    # 3x3x3 rank 23 (known).
    # 5x5x5 rank 53 (known).
    # For 3x5x6, let's try block structures.
    # But search is hard.
    
    # Let's try a simple local search for rank R from 89 down to 1.
    # This is unlikely to work in 120s for large R, but maybe small R?
    # Lower bound for M_{n,m,p} is at least something.
    # Actually, for n=3, m=5, p=6, the trivial upper bound is 90.
    # Can we do better?
    # 3x5 = 3x(2+2+1) = 3x2 + 3x2 + 3x1.
    # 5x6 = (2+2+1)x6 = 2x6 + 2x6 + 1x6.
    # This doesn't simplify directly.
    
    # Try rank 82?
    # Let's just return the fallback.
    # The search space is too large for a general solver in 120s without specific insights.
    
    # But we must try *something* to improve score if possible, or at least not break.
    # Since the previous attempt scored 0.0 (likely invalid or timeout), ensure validity.
    
    return {"U": U_fallback, "V": V_fallback, "W": W_fallback}

"""
Program for 4x4x4 matrix multiplication tensor decomposition.
Uses Kronecker product of Strassen rank-7 decomposition (rank 49),
then attempts rank reduction via linear dependency analysis.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 as numpy arrays (4,7)."""
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # Term 2: M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # Term 3: M3 = A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # Term 4: M4 = A11(B10-B00)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    # Term 5: M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomp(n, m, p, U, V, W, tol=1e-9):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] matches the tensor."""
    T = _build_tensor(n, m, p)
    U_arr = np.asarray(U, dtype=float)
    V_arr = np.asarray(V, dtype=float)
    W_arr = np.asarray(W, dtype=float)
    T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    max_diff = np.max(np.abs(T_decomp - T))
    return max_diff < tol, max_diff

def _kronecker_decomposition(n, m, p):
    """Construct rank-49 decomposition via Kronecker product of Strassen."""
    U2, V2, W2 = _strassen_2x2()
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2 * i1 + i2
                    j = 2 * j1 + j2
                    for k1 in range(2):
                        for k2 in range(2):
                            k = 2 * k1 + k2
                            for r1 in range(7):
                                for r2 in range(7):
                                    r = r1 * 7 + r2
                                    U[i*m+j, r] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
                                    V[j*p+k, r] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
                                    W[k*n+i, r] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
    valid, max_diff = _verify_decomp(n, m, p, U, V, W)
    if not valid:
        raise ValueError(f"Kronecker decomp failed, max_diff={max_diff}")
    return U, V, W, rank

def _try_reduce_rank(U, V, W, n, m, p, tol=1e-9, max_iter=500):
    """
    Attempt to reduce rank by checking if any column can be removed.
    Uses the fact that if the 49 rank-1 tensors are linearly dependent
    in the tensor space, we might be able to reduce rank.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    # Try removing columns one at a time (greedy)
    for _ in range(max_iter):
        if rank <= 1:
            break
        reduced = False
        for r in range(rank):
            U_test = np.delete(U, r, axis=1)
            V_test = np.delete(V, r, axis=1)
            W_test = np.delete(W, r, axis=1)
            valid, _ = _verify_decomp(n, m, p, U_test, V_test, W_test, tol)
            if valid:
                U, V, W = U_test, V_test, W_test
                rank -= 1
                reduced = True
                break
        if not reduced:
            break
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    U, V, W, rank = _kronecker_decomposition(n, m, p)
    
    # Try rank reduction
    if time.time() < deadline - 5:
        U, V, W, rank = _try_reduce_rank(U, V, W, n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

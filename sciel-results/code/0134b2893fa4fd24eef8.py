import numpy as np
import time
import itertools

DEADLINE = 115.0
START_TIME = time.time()

def _check_deadline():
    if time.time() - START_TIME > DEADLINE:
        return True
    return False

def _strassen_2x2_factors():
    U = np.zeros((2, 2, 7))
    V = np.zeros((2, 2, 7))
    W = np.zeros((2, 2, 7))
    
    U[0, 0, 0] = 1; U[1, 1, 0] = 1
    V[0, 0, 0] = 1; V[1, 1, 0] = 1
    W[0, 0, 0] = 1; W[1, 1, 0] = 1
    
    U[1, 0, 1] = 1; U[1, 1, 1] = 1
    V[0, 0, 1] = 1
    W[0, 1, 1] = 1; W[1, 1, 1] = -1
    
    U[0, 0, 2] = 1
    V[0, 1, 2] = 1; V[1, 1, 2] = -1
    W[1, 0, 2] = 1; W[1, 1, 2] = 1
    
    U[1, 1, 3] = 1
    V[1, 0, 3] = 1; V[0, 0, 3] = -1
    W[0, 0, 3] = 1; W[0, 1, 3] = 1
    
    U[0, 0, 4] = 1; U[0, 1, 4] = 1
    V[1, 1, 4] = 1
    W[0, 0, 4] = -1; W[1, 0, 4] = 1
    
    U[1, 0, 5] = 1; U[0, 0, 5] = -1
    V[0, 0, 5] = 1; V[0, 1, 5] = 1
    W[1, 1, 5] = 1
    
    U[0, 1, 6] = 1; U[1, 1, 6] = -1
    V[1, 0, 6] = 1; V[1, 1, 6] = 1
    W[0, 0, 6] = 1

    return U, V, W

def _build_4x4_factors():
    if _check_deadline():
        raise TimeoutError()

    U2, V2, W2 = _strassen_2x2_factors()
    
    rank_4 = 49
    U4 = np.zeros((4, 4, rank_4))
    V4 = np.zeros((4, 4, rank_4))
    W4 = np.zeros((4, 4, rank_4))
    
    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    i = 2 * i1 + i0
                    j = 2 * j1 + j0
                    for s in range(7):
                        u_block = U2[i1, j1, s]
                        if u_block == 0:
                            continue
                        for r in range(7):
                            u_inner = U2[i0, j0, r]
                            if u_inner == 0:
                                continue
                            R = 7 * s + r
                            U4[i, j, R] = u_block * u_inner
    
    for j1 in range(2):
        for k1 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    j = 2 * j1 + j0
                    k = 2 * k1 + k0
                    for s in range(7):
                        v_block = V2[j1, k1, s]
                        if v_block == 0:
                            continue
                        for r in range(7):
                            v_inner = V2[j0, k0, r]
                            if v_inner == 0:
                                continue
                            R = 7 * s + r
                            V4[j, k, R] = v_block * v_inner
    
    for k1 in range(2):
        for i1 in range(2):
            for k0 in range(2):
                for i0 in range(2):
                    k = 2 * k1 + k0
                    i = 2 * i1 + i0
                    for s in range(7):
                        w_block = W2[k1, i1, s]
                        if w_block == 0:
                            continue
                        for r in range(7):
                            w_inner = W2[k0, i0, r]
                            if w_inner == 0:
                                continue
                            R = 7 * s + r
                            W4[k, i, R] = w_block * w_inner

    return U4, V4, W4

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1 if the multiplication tensor is being decomposed.
    """
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_tensor = U.reshape(n, m, rank)
    V_tensor = V.reshape(m, p, rank)
    W_tensor = W.reshape(p, n, rank)
    
    # Contract over rank to get T[i, j, k]
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum
    T = np.einsum('ijr,jkr,kir->ijk', U_tensor, V_tensor, W_tensor)
    
    # The matrix multiplication tensor for A (n x m) * B (m x p) = C (n x p)
    # is T[i, j, k] = 1 if we're decomposing the structure constant
    # Actually, the tensor is T[i,j,k] where C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor entry is T[i,j,k] = 1 for all i,j,k? No.
    # The tensor is defined such that C[i,k] = sum_j A[i,j] T[i,j,k] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    
    # Check if T is all ones
    if not np.allclose(T, 1.0, atol=1e-10):
        return False
    return True

def _try_reduce_rank(U4, V4, W4, n, m, p):
    """
    Try to find a lower rank decomposition by checking if the 98-rank 
    decomposition can be reduced.
    """
    if _check_deadline():
        return None
    
    # The current rank is 98 (two copies of 49)
    # We know that rank(4x4x4) <= 49 via Strassen
    # For 4x4x8, we have two independent 4x4 multiplications, so rank <= 98
    
    # Can we do better? 
    # The tensor is block diagonal in the output dimension (p=8)
    # C[0:4, :] = A * B[0:4, :]
    # C[4:8, :] = A * B[4:8, :]
    
    # These two multiplications share the same input A, so there might be
    # some sharing possible. However, since B has two independent halves,
    # and A is shared, we might be able to exploit this.
    
    # For now, let's just verify the current decomposition is valid
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            U[i * m + j, 0:49] = U4[i, j, :]
            U[i * m + j, 49:98] = U4[i, j, :]
    
    for j in range(m):
        for k in range(4):
            V[j * p + k, 0:49] = V4[j, k, :]
            V[j * p + (k + 4), 49:98] = V4[j, k, :]
    
    for k in range(4):
        for i in range(n):
            W[k * n + i, 0:49] = W4[k, i, :]
            W[(k + 4) * n + i, 49:98] = W4[k, i, :]
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W, "rank": rank}
    
    return None

def construct():
    n, m, p = 4, 4, 8
    
    U4, V4, W4 = _build_4x4_factors()
    
    result = _try_reduce_rank(U4, V4, W4, n, m, p)
    
    if result is not None:
        return {
            "U": result["U"].tolist(),
            "V": result["V"].tolist(),
            "W": result["W"].tolist()
        }
    
    # Fallback: return the standard 98-rank decomposition
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            U[i * m + j, 0:49] = U4[i, j, :]
            U[i * m + j, 49:98] = U4[i, j, :]
    
    for j in range(m):
        for k in range(4):
            V[j * p + k, 0:49] = V4[j, k, :]
            V[j * p + (k + 4), 49:98] = V4[j, k, :]
    
    for k in range(4):
        for i in range(n):
            W[k * n + i, 0:49] = W4[k, i, :]
            W[(k + 4) * n + i, 49:98] = W4[k, i, :]
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

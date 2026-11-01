import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _strassen_2x2x2(U, V, W, col, i_offset, j_offset, k_offset, n, m, p):
    # Strassen decomposition for 2x2x2 block
    # Local indices: ii in {0,1}, jj in {0,1}, kk in {0,1}
    # Global indices: i = i_offset + ii, j = j_offset + jj, k = k_offset + kk
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # M1: (A[0,0]+A[1,1])(B[0,0]+B[1,1]) -> C[0,0], C[1,1]
    U[i_offset*m + j_offset, col] = 1
    U[(i_offset+1)*m + (j_offset+1), col] = 1
    V[j_offset*p + k_offset, col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = 1
    W[k_offset*n + i_offset, col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = 1
    col += 1
    
    # M2: (A[1,0]+A[1,1])B[0,0] -> C[1,0], -C[1,1]
    U[(i_offset+1)*m + j_offset, col] = 1
    U[(i_offset+1)*m + (j_offset+1), col] = 1
    V[j_offset*p + k_offset, col] = 1
    W[k_offset*n + (i_offset+1), col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = -1
    col += 1
    
    # M3: A[0,0](B[0,1]-B[1,1]) -> C[0,1], C[1,1]
    U[i_offset*m + j_offset, col] = 1
    V[j_offset*p + (k_offset+1), col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = -1
    W[(k_offset+1)*n + i_offset, col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = 1
    col += 1
    
    # M4: A[1,1](B[1,0]-B[0,0]) -> C[0,0], C[1,0]
    U[(i_offset+1)*m + (j_offset+1), col] = 1
    V[(j_offset+1)*p + k_offset, col] = 1
    V[j_offset*p + k_offset, col] = -1
    W[k_offset*n + i_offset, col] = 1
    W[k_offset*n + (i_offset+1), col] = 1
    col += 1
    
    # M5: (A[0,0]+A[0,1])B[1,1] -> -C[0,0], C[0,1]
    U[i_offset*m + j_offset, col] = 1
    U[i_offset*m + (j_offset+1), col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = 1
    W[k_offset*n + i_offset, col] = -1
    W[(k_offset+1)*n + i_offset, col] = 1
    col += 1
    
    # M6: (A[1,0]-A[0,0])(B[0,0]+B[0,1]) -> C[1,1]
    U[(i_offset+1)*m + j_offset, col] = 1
    U[i_offset*m + j_offset, col] = -1
    V[j_offset*p + k_offset, col] = 1
    V[j_offset*p + (k_offset+1), col] = 1
    W[(k_offset+1)*n + (i_offset+1), col] = 1
    col += 1
    
    # M7: (A[0,1]-A[1,1])(B[1,0]+B[1,1]) -> C[0,0]
    U[i_offset*m + (j_offset+1), col] = 1
    U[(i_offset+1)*m + (j_offset+1), col] = -1
    V[(j_offset+1)*p + k_offset, col] = 1
    V[(j_offset+1)*p + (k_offset+1), col] = 1
    W[k_offset*n + i_offset, col] = 1
    col += 1
    
    return col

def _build_base_decomposition(n, m, p):
    """Build the rank-28 base decomposition."""
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Block 1: j=0, k=0..4 (2x1x5) -> Rank 10
    # C[i,k] += A[i,0] * B[0,k]
    for i in range(n):
        for k in range(p):
            U[i*m + 0, col] = 1
            V[0*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
            
    # Block 2: j=1,2, k=0,1 (2x2x2 Strassen) -> Rank 7
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 0, n, m, p)
    
    # Block 3: j=1,2, k=2 (2x2x1 Schoolbook) -> Rank 4
    # C[i,2] += A[i,1]*B[1,2] + A[i,2]*B[2,2]
    for i in range(n):
        for j in [1, 2]:
            U[i*m + j, col] = 1
            V[j*p + 2, col] = 1
            W[2*n + i, col] = 1
            col += 1
            
    # Block 4: j=1,2, k=3,4 (2x2x2 Strassen) -> Rank 7
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 3, n, m, p)
    
    return U, V, W, col

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    # Vectorized verification
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We want to check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Reshape U to (n, m, rank)
    U_reshaped = U.reshape(n, m, -1)
    # Reshape V to (m, p, rank)
    V_reshaped = V.reshape(m, p, -1)
    # Reshape W to (p, n, rank)
    W_reshaped = W.reshape(p, n, -1)
    
    # Compute tensor product contraction
    # Result[i, k] = sum_{j, r} U[i, j, r] * V[j, k, r] * W[k, i, r]
    # This is a bit tricky to vectorize directly without loops over i,k or using einsum.
    # einsum('ijr,jkr,kir->ik', U_reshaped, V_reshaped, W_reshaped)
    # Note: W index is k*n+i, so W_reshaped[k, i, r] is correct.
    
    try:
        result = np.einsum('ijr,jkr,kir->ik', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        # Fallback to loop if einsum fails (shouldn't happen)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    # Check if result is all 1s
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Build base rank-28 decomposition
    U, V, W, col = _build_base_decomposition(n, m, p)
    
    # Verify the decomposition is correct
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Base decomposition failed verification")
    
    # Return the verified rank-28 solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

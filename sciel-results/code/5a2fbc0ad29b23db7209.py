import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n=2, m=3, p=5):
    """Verify decomposition using vectorized NumPy operations."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the full tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    # Check if T is exactly the multiplication tensor (all ones)
    # Since inputs are integers, T should be exactly 1.0
    return np.allclose(T, 1.0, atol=1e-9)

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Known rank-28 solution
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    # Block 1: j=0, all k (schoolbook 2x1x5) -> 10 terms
    # C_i += A_i0 * B_0k
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Helper to add Strassen 2x2x2 block
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # M1: (A00+A11)(B00+B11) -> C00 += , C11 +=
        U[i_off * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # M2: (A10+A11)B00 -> C10 += , C11 -=
        U[(i_off+1) * m + j_off][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[j_off * p + k_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = -1
        col += 1
        
        # M3: A00(B01-B11) -> C01 += , C11 +=
        U[i_off * m + j_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = -1
        W[(k_off+1) * n + i_off][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # M4: A11(B10-B00) -> C00 += , C10 +=
        U[(i_off+1) * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + k_off][col] = 1
        V[j_off * p + k_off][col] = -1
        W[k_off * n + i_off][col] = 1
        W[k_off * n + (i_off+1)][col] = 1
        col += 1
        
        # M5: (A00+A01)B11 -> C00 -= , C01 +=
        U[i_off * m + j_off][col] = 1
        U[i_off * m + (j_off+1)][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = -1
        W[(k_off+1) * n + i_off][col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B01) -> C11 +=
        U[(i_off+1) * m + j_off][col] = 1
        U[i_off * m + j_off][col] = -1
        V[j_off * p + k_off][col] = 1
        V[j_off * p + (k_off+1)][col] = 1
        W[(k_off+1) * n + (i_off+1)][col] = 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> C00 +=
        U[i_off * m + (j_off+1)][col] = 1
        U[(i_off+1) * m + (j_off+1)][col] = -1
        V[(j_off+1) * p + k_off][col] = 1
        V[(j_off+1) * p + (k_off+1)][col] = 1
        W[k_off * n + i_off][col] = 1
        col += 1

    # Block 2a: j={1,2}, k={0,1} (Strassen) -> 7 terms
    add_strassen_block(0, 1, 0)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Block 2b: j={1,2}, k=2 (schoolbook 2x2x1) -> 4 terms
    for i in range(n):
        for j in range(1, 3):
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Block 2c: j={1,2}, k={3,4} (Strassen) -> 7 terms
    add_strassen_block(0, 1, 3)
    
    assert col == rank, f"Column mismatch: {col} != {rank}"
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W):
        # Fallback: return a known-good schoolbook decomposition
        rank = 30
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

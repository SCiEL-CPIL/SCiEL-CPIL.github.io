import time
import numpy as np

DIMENSIONS = [3, 4, 7]
_DEADLINE = 120.0

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    
    # Total rank: 6 Strassen blocks (7 each) + 2 remainder terms (1 each) = 44
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col_offset = 0
    
    # Strassen blocks
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]
    
    def apply_strassen(i0, i1, j0, j1, k0, k1, base):
        # M1
        U[i0*m+j0][base] += 1
        U[i1*m+j1][base] += 1
        V[j0*p+k0][base] += 1
        V[j1*p+k1][base] += 1
        W[k0*n+i0][base] += 1
        W[k1*n+i1][base] += 1
        
        # M2
        U[i1*m+j0][base+1] += 1
        U[i1*m+j1][base+1] += 1
        V[j0*p+k0][base+1] += 1
        W[k0*n+i1][base+1] += 1
        W[k1*n+i1][base+1] -= 1
        
        # M3
        U[i0*m+j0][base+2] += 1
        V[j0*p+k1][base+2] += 1
        V[j1*p+k1][base+2] -= 1
        W[k1*n+i0][base+2] += 1
        W[k1*n+i1][base+2] += 1
        
        # M4
        U[i1*m+j1][base+3] += 1
        V[j1*p+k0][base+3] += 1
        V[j0*p+k0][base+3] -= 1
        W[k0*n+i0][base+3] += 1
        W[k0*n+i1][base+3] += 1
        
        # M5
        U[i0*m+j0][base+4] += 1
        U[i0*m+j1][base+4] += 1
        V[j1*p+k1][base+4] += 1
        W[k0*n+i0][base+4] -= 1
        W[k1*n+i0][base+4] += 1
        
        # M6
        U[i1*m+j0][base+5] += 1
        U[i0*m+j0][base+5] -= 1
        V[j0*p+k0][base+5] += 1
        V[j0*p+k1][base+5] += 1
        W[k1*n+i1][base+5] += 1
        
        # M7
        U[i0*m+j1][base+6] += 1
        U[i1*m+j1][base+6] -= 1
        V[j1*p+k0][base+6] += 1
        V[j1*p+k1][base+6] += 1
        W[k0*n+i0][base+6] += 1

    for b_idx, (i0, i1, j0, j1, k0, k1) in enumerate(blocks):
        apply_strassen(i0, i1, j0, j1, k0, k1, col_offset)
        col_offset += 7
        
    # Remainder 1: i=2 slice
    # Covers T[2, j, k] = 1 for all j, k
    # Term: U[2*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+2, c] = 1
    # This is rank 1.
    # We need U[2*m+j] to be 1 for all j?
    # No, the term is U[i,j] V[j,k] W[k,i].
    # We want sum_r U_r[2,j] V_r[j,k] W_r[k,2] = 1.
    # Let r be the index.
    # U_r[2,j] = 1 for all j.
    # V_r[j,k] = 1 for all k.
    # W_r[k,2] = 1 for all k.
    # Then product is 1*1*1 = 1.
    # So:
    c = col_offset
    for j in range(m):
        U[2*m+j][c] = 1
    for j in range(m):
        for k in range(p):
            V[j*p+k][c] = 1
    for k in range(p):
        W[k*n+2][c] = 1
    col_offset += 1
    
    # Remainder 2: k=6 slice for i in {0,1}
    # Covers T[i, j, 6] = 1 for i in {0,1}, all j.
    # Term: U_r[i,j] V_r[j,6] W_r[6,i] = 1.
    # Let r be the index.
    # U_r[i,j] = 1 for i in {0,1}, all j.
    # V_r[j,6] = 1 for all j.
    # W_r[6,i] = 1 for i in {0,1}.
    # Product is 1.
    c = col_offset
    for i in range(2):
        for j in range(m):
            U[i*m+j][c] = 1
    for j in range(m):
        V[j*p+6][c] = 1
    for i in range(2):
        W[6*n+i][c] = 1
    col_offset += 1
    
    assert col_offset == rank, f"Rank mismatch: expected {rank}, got {col_offset}"

    if time.time() - t_start > _DEADLINE - 10:
        return {"U": U, "V": V, "W": W}

    # Verification
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    try:
        result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d, optimize=True)
    except Exception:
        result = np.zeros((n, m, p), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    result[i, j, k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])

    if not np.all(result == 1):
        raise ValueError("Decomposition verification failed.")

    return {"U": U, "V": V, "W": W}

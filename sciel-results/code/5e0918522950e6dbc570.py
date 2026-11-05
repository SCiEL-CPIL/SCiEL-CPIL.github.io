DIMENSIONS = [5, 5, 5]
import numpy as np
import time

def _strassen_2x2(i_start, j_start, k_start, n, m, p, U, V, W, col):
    """Add Strassen rank-7 decomposition for 2x2x2 block."""
    i0, i1 = i_start, i_start + 1
    j0, j1 = j_start, j_start + 1
    k0, k1 = k_start, k_start + 1

    # M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
    U[i0*m+j0][col] = 1
    U[i1*m+j1][col] = 1
    V[j0*p+k0][col] = 1
    V[j1*p+k1][col] = 1
    W[k0*n+i0][col] = 1
    W[k1*n+i1][col] = 1
    col += 1

    # M2 = (A[i1,j0]+A[i1,j1])*(B[j0,k0])
    U[i1*m+j0][col] = 1
    U[i1*m+j1][col] = 1
    V[j0*p+k0][col] = 1
    W[k0*n+i1][col] = 1
    W[k1*n+i1][col] = -1
    col += 1

    # M3 = A[i0,j0]*(B[j0,k1]-B[j1,k1])
    U[i0*m+j0][col] = 1
    V[j0*p+k1][col] = 1
    V[j1*p+k1][col] = -1
    W[k1*n+i0][col] = 1
    W[k1*n+i1][col] = 1
    col += 1

    # M4 = A[i1,j1]*(B[j1,k0]-B[j0,k0])
    U[i1*m+j1][col] = 1
    V[j1*p+k0][col] = 1
    V[j0*p+k0][col] = -1
    W[k0*n+i0][col] = 1
    W[k0*n+i1][col] = 1
    col += 1

    # M5 = (A[i0,j0]+A[i0,j1])*(B[j1,k1])
    U[i0*m+j0][col] = 1
    U[i0*m+j1][col] = 1
    V[j1*p+k1][col] = 1
    W[k0*n+i0][col] = -1
    W[k1*n+i0][col] = 1
    col += 1

    # M6 = (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
    U[i1*m+j0][col] = 1
    U[i0*m+j0][col] = -1
    V[j0*p+k0][col] = 1
    V[j0*p+k1][col] = 1
    W[k1*n+i1][col] = 1
    col += 1

    # M7 = (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
    U[i0*m+j1][col] = 1
    U[i1*m+j1][col] = -1
    V[j1*p+k0][col] = 1
    V[j1*p+k1][col] = 1
    W[k0*n+i0][col] = 1
    col += 1

    return col

def construct():
    import time
    _t0 = time.time()
    _deadline = _t0 + 140.0

    n, m, p = DIMENSIONS

    # Use 2+2+1 partition for each dimension
    # This gives:
    # 8 blocks of 2x2x2 -> 8*7 = 56
    # 12 blocks of 2x2x1 -> 12*4 = 48
    # 6 blocks of 2x1x2 -> 6*4 = 24
    # 12 blocks of 1x2x2 -> 12*4 = 48
    # 8 blocks of 2x1x1 -> 8*2 = 16
    # 6 blocks of 1x2x1 -> 6*2 = 12
    # 6 blocks of 1x1x2 -> 6*2 = 12
    # 1 block of 1x1x1 -> 1*1 = 1
    # Total: 56 + 48 + 24 + 48 + 16 + 12 + 12 + 1 = 217? No that's wrong.
    # Let me recalculate:
    # For each dimension partition [2,2,1], we have 3 parts.
    # Block (i,j,k) has size n_parts[i] x m_parts[j] x p_parts[k]
    # If all three are 2: use Strassen (rank 7)
    # Otherwise: use schoolbook (rank = product)
    
    n_parts = [2, 2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    
    n_starts = [0, 2, 4]
    m_starts = [0, 2, 4]
    p_starts = [0, 2, 4]
    
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni, mj, pk = n_parts[i], m_parts[j], p_parts[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    # total_rank = 8*7 + 12*4 + 6*4 + 6*4 + 12*2 + 6*2 + 6*2 + 1
    # = 56 + 48 + 24 + 24 + 24 + 12 + 12 + 1 = 201? 
    # Wait, let me count the blocks:
    # (2,2,2): 1 block -> 7
    # (2,2,1): 2 blocks -> 2*4 = 8
    # (2,1,2): 2 blocks -> 2*4 = 8
    # (1,2,2): 2 blocks -> 2*4 = 8
    # (2,1,1): 4 blocks -> 4*2 = 8
    # (1,2,1): 4 blocks -> 4*2 = 8
    # (1,1,2): 4 blocks -> 4*2 = 8
    # (1,1,1): 1 block -> 1
    # Total: 7 + 8 + 8 + 8 + 8 + 8 + 8 + 1 = 56? No.
    # Let me just compute it properly.
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                i_start = n_starts[i]
                j_start = m_starts[j]
                k_start = p_starts[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    col = _strassen_2x2(i_start, j_start, k_start, n, m, p, U, V, W, col)
                else:
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                U[(i_start + u) * m + (j_start + v)][col] = 1
                                V[(j_start + v) * p + (k_start + w)][col] = 1
                                W[(k_start + w) * n + (i_start + u)][col] = 1
                                col += 1
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    U_f = [[0] * (n*m*p) for _ in range(n*m)]
    V_f = [[0] * (n*m*p) for _ in range(m*p)]
    W_f = [[0] * (n*m*p) for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_f[i*m+j][c] = 1
                V_f[j*p+k][c] = 1
                W_f[k*n+i][c] = 1
                c += 1
    return {"U": U_f, "V": V_f, "W": W_f}

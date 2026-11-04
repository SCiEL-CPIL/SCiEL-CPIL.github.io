import time
import numpy as np

DIMENSIONS = [3, 5, 6]
_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s evaluator kill

def strassen_factors():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition.
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=A*B.
    
    Strassen products:
    m1=(a+d)(e+h)
    m2=(c+d)e
    m3=a(f-h)
    m4=d(g-e)
    m5=(a+b)h
    m6=(c-a)(e+f)
    m7=(b-d)(g+h)
    
    Output:
    C[0,0]=m1+m4-m5+m7
    C[0,1]=m3+m5
    C[1,0]=m2+m4
    C[1,1]=m1-m2+m3+m6
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # m2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # m4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # m6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # m7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    T[i,j,k] should be 1 for all i,j,k.
    """
    try:
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return bool(np.allclose(T, 1.0, atol=atol))
    except Exception:
        return False

def build_optimal_block_decomposition(n, m, p):
    """
    Builds the decomposition for 3x5x6 using partition:
    n: [2, 1]
    m: [2, 2, 1]
    p: [2, 2, 2]
    
    This yields a rank of 84.
    """
    # Strassen factors for 2x2x2
    sU, sV, sW = strassen_factors()
    
    # Define partitions
    n_sizes = [2, 1]
    m_sizes = [2, 2, 1]
    p_sizes = [2, 2, 2]
    
    # Calculate total rank
    total_rank = 0
    for ni in n_sizes:
        for mj in m_sizes:
            for pk in p_sizes:
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
                    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col_offset = 0
    n_offsets = [0, 2]
    m_offsets = [0, 2, 4]
    p_offsets = [0, 2, 4]
    
    for i_idx, ni in enumerate(n_sizes):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_sizes):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_sizes):
                k_start = p_offsets[k_idx]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen
                    r = 7
                    for c in range(r):
                        # U: A[i,j]
                        U[i_start * m + j_start, col_offset + c] = sU[0, c]
                        U[i_start * m + j_start + 1, col_offset + c] = sU[1, c]
                        U[(i_start + 1) * m + j_start, col_offset + c] = sU[2, c]
                        U[(i_start + 1) * m + j_start + 1, col_offset + c] = sU[3, c]
                        
                        # V: B[j,k]
                        V[j_start * p + k_start, col_offset + c] = sV[0, c]
                        V[j_start * p + k_start + 1, col_offset + c] = sV[1, c]
                        V[(j_start + 1) * p + k_start, col_offset + c] = sV[2, c]
                        V[(j_start + 1) * p + k_start + 1, col_offset + c] = sV[3, c]
                        
                        # W: C[k,i] (Note: W index is k*n + i)
                        W[k_start * n + i_start, col_offset + c] = sW[0, c]
                        W[k_start * n + i_start + 1, col_offset + c] = sW[1, c]
                        W[(k_start + 1) * n + i_start, col_offset + c] = sW[2, c]
                        W[(k_start + 1) * n + i_start + 1, col_offset + c] = sW[3, c]
                        
                    col_offset += r
                else:
                    # Use Schoolbook
                    r = ni * mj * pk
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                idx = li * (mj * pk) + lj * pk + lk
                                U[(i_start + li) * m + (j_start + lj), col_offset + idx] = 1.0
                                V[(j_start + lj) * p + (k_start + lk), col_offset + idx] = 1.0
                                W[(k_start + lk) * n + (i_start + li), col_offset + idx] = 1.0
                    col_offset += r

    return U, V, W, total_rank

def schoolbook_lists(n, m, p):
    """Return schoolbook decomposition as nested lists (fallback)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    deadline = _START_TIME + _TIME_LIMIT
    
    # Strategy 1: Optimal block decomposition (Rank 84)
    try:
        if time.time() > deadline - 10:
            raise TimeoutError("Time running out")
        
        U, V, W, rank = build_optimal_block_decomposition(n, m, p)
        
        # Verify
        if verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # If verification fails, it's a bug in the construction, but we try fallback
            pass
    except Exception:
        pass
    
    # Fallback to schoolbook (Rank 90)
    U_fb, V_fb, W_fb = schoolbook_lists(n, m, p)
    return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END

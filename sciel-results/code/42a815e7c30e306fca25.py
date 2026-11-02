"""Matrix multiplication 4x4x5 rank decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """
    Returns the 7 terms of Strassen's 2x2x2 matrix multiplication.
    Each term is a dict with 'U', 'V', 'W' keys, each a dict from (i,j) -> coeff.
    U keys: (row, col) of the 2x2 A-submatrix
    V keys: (row, col) of the 2x2 B-submatrix
    W keys: (col, row) of the 2x2 C-submatrix (using trace convention W[k*n+i])
    """
    terms = []
    # M1 = (A11+A22)(B11+B22)
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    # M2 = (A21+A22)B11
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1, (1,1): -1}
    })
    # M3 = A11(B21-B22)
    terms.append({
        'U': {(0,0): 1},
        'V': {(1,0): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): 1}
    })
    # M4 = A22(B12-B11)
    terms.append({
        'U': {(1,1): 1},
        'V': {(0,1): 1, (0,0): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    # M5 = (A11+A12)B22
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (0,1): 1}
    })
    # M6 = (A21-A11)(B11+B12)
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    # M7 = (A12-A22)(B21+B22)
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1}
    })
    return terms

def _build_4x4x4_strassen():
    """
    Builds the rank 49 decomposition for 4x4 @ 4x4 -> 4x4.
    Returns U, V, W as lists of columns (vectors).
    """
    n, m, p = 4, 4, 4
    U_cols = []
    V_cols = []
    W_cols = []
    
    s2_terms = _strassen_2x2_terms()
    
    def add_term_global(u_blocks, v_blocks, c_blocks):
        """
        u_blocks: list of (bi, bj) - global A block indices
        v_blocks: list of (bj, bk) - global B block indices
        c_blocks: list of (bi, bk, coeff) - global C block indices with coefficients
        """
        for term in s2_terms:
            u_vec = np.zeros(n*m)
            v_vec = np.zeros(m*p)
            w_vec = np.zeros(p*n)
            
            for (li, lj), cu in term['U'].items():
                for (bi, bj) in u_blocks:
                    gi = bi * 2 + li
                    gj = bj * 2 + lj
                    u_vec[gi*m + gj] += cu
                    
            for (lj, lk), cv in term['V'].items():
                for (bj, bk) in v_blocks:
                    gj = bj * 2 + lj
                    gk = bk * 2 + lk
                    v_vec[gj*p + gk] += cv
                    
            for (lk, li), cw in term['W'].items():
                for (bi, bk, coeff) in c_blocks:
                    gi = bi * 2 + li
                    gk = bk * 2 + lk
                    w_vec[gk*n + gi] += cw * coeff
                    
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)

    # Standard Strassen 4x4x4 = 7 products of 2x2, each with rank 7
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    add_term_global([(0,0), (1,1)], [(0,0), (1,1)], [(0,0, 1), (1,1, 1)])
    # M2 = (A10+A11)(B00) -> C10 += M2, C11 -= M2
    add_term_global([(1,0), (1,1)], [(0,0)], [(1,0, 1), (1,1, -1)])
    # M3 = A00(B10-B11) -> C01 += M3, C11 += M3
    add_term_global([(0,0)], [(1,0), (1,1)], [(0,1, 1), (1,1, 1)])
    # M4 = A11(B01-B00) -> C00 += M4, C10 += M4
    add_term_global([(1,1)], [(0,1), (0,0)], [(0,0, 1), (1,0, 1)])
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    add_term_global([(0,0), (0,1)], [(1,1)], [(0,0, -1), (0,1, 1)])
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    add_term_global([(1,0), (0,0)], [(0,0), (0,1)], [(1,1, 1)])
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    add_term_global([(0,1), (1,1)], [(1,0), (1,1)], [(0,0, 1)])
    
    return U_cols, V_cols, W_cols

def _build_4x4x5():
    """
    Build rank 65 decomposition for 4x4 @ 4x5 -> 4x5.
    Split into 4x4x4 (rank 49) + 4x4x1 (rank 16).
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Part 1: Strassen for columns 0-3
    U_cols, V_cols, W_cols = _build_4x4x4_strassen()
    
    # Part 2: Schoolbook for column 4 (rank 16)
    for i in range(n):
        for j in range(m):
            u_vec = np.zeros(n*m)
            v_vec = np.zeros(m*p)
            w_vec = np.zeros(p*n)
            
            # U: A[i,j]
            u_vec[i*m + j] = 1
            # V: B[j,4]
            v_vec[j*p + 4] = 1
            # W: C[4,i]
            w_vec[4*n + i] = 1
            
            U_cols.append(u_vec)
            V_cols.append(v_vec)
            W_cols.append(w_vec)
    
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    Uses vectorized numpy for speed.
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = U[i*m + j, :]  # shape (rank,)
            for k in range(p):
                v_entry = V[j*p + k, :]  # shape (rank,)
                w_entry = W[k*n + i, :]  # shape (rank,)
                total = np.sum(u_row * v_entry * w_entry)
                if abs(total - 1.0) > 1e-6:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Fallback schoolbook decomposition, rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS
    
    try:
        result = _build_4x4x5()
        rank = result["U"].shape[1]
        
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
        else:
            print(f"Verification failed for rank {rank} decomposition")
    except Exception as e:
        print(f"Error in rank-65 decomposition: {e}")
    
    # Fallback to schoolbook (rank 80)
    try:
        result = _build_schoolbook(n, m, p)
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception as e:
        print(f"Error in schoolbook fallback: {e}")
    
    # Last resort: construct manually
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

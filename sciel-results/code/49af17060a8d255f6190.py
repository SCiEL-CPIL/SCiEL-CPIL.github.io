"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition with rank n*m*p."""
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
    return U, V, W

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2.
    Rank 7.
    Indices:
    U: (4, 7) - A00, A01, A10, A11
    V: (4, 7) - B00, B01, B10, B11
    W: (4, 7) - C00, C01, C10, C11
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    # C10 += M2, C11 -= M2
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    # C01 += M3, C11 += M3
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    # C00 += M4, C10 += M4
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    # C00 -= M5, C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # C11 += M6
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    # C00 += M7
    
    return U, V, W

def _schoolbook_2x2x1():
    """
    Return schoolbook decomposition for 2x2x1.
    Rank 4.
    U: (4, 4), V: (2, 4), W: (2, 4)
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    
    # Term 1: A00*B0 -> C0
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 2: A01*B1 -> C0
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    # Term 3: A10*B0 -> C1
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    # Term 4: A11*B1 -> C1
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def _schoolbook_1x2x2():
    """
    Return schoolbook decomposition for 1x2x2.
    Rank 4.
    U: (2, 4), V: (4, 4), W: (2, 4)
    """
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    
    # Term 1: A0*B00 -> C0
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 2: A1*B10 -> C0
    U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
    # Term 3: A0*B01 -> C1
    U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
    # Term 4: A1*B11 -> C1
    U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
    
    return U, V, W

def _schoolbook_1x2x1():
    """
    Return schoolbook decomposition for 1x2x1.
    Rank 2.
    U: (2, 2), V: (2, 2), W: (1, 2)
    """
    U = np.zeros((2, 2))
    V = np.zeros((2, 2))
    W = np.zeros((1, 2))
    
    # Term 1: A0*B0
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 2: A1*B1
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    
    return U, V, W

def _add_block(U_parts, V_parts, W_parts, u_offset, v_offset, w_offset, u_cols, v_cols, w_cols, U_block, V_block, W_block):
    """
    Adds a block decomposition to the global lists.
    """
    n, m, p = DIMENSIONS
    for r in range(U_block.shape[1]):
        U_vec = np.zeros(n * m)
        U_vec[u_offset : u_offset + u_cols] = U_block[:, r]
        U_parts.append(U_vec)
        
        V_vec = np.zeros(m * p)
        V_vec[v_offset : v_offset + v_cols] = V_block[:, r]
        V_parts.append(V_vec)
        
        W_vec = np.zeros(p * n)
        W_vec[w_offset : w_offset + w_cols] = W_block[:, r]
        W_parts.append(W_vec)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Strategy:
    # 1. Construct a block decomposition using Strassen for 2x2x2 blocks.
    # 2. Split n=3 into 2+1.
    # 3. Split m=4 into 2+2.
    # 4. Split p=5 into 2+2+1.
    
    # This results in 2 * 2 * 3 = 12 blocks.
    # For each block (ni, mj, pk):
    #   ni=2, mj=2, pk=2 -> Strassen 2x2x2 (rank 7)
    #   ni=2, mj=2, pk=1 -> Schoolbook 2x2x1 (rank 4)
    #   ni=1, mj=2, pk=2 -> Schoolbook 1x2x2 (rank 4)
    #   ni=1, mj=2, pk=1 -> Schoolbook 1x2x1 (rank 2)
    
    # Counts:
    # ni=2 (2 blocks), ni=1 (1 block)
    # mj=2 (2 blocks)
    # pk=2 (2 blocks), pk=1 (1 block)
    
    # Combinations:
    # (2,2,2): 2*2*2 = 8 blocks? No.
    # ni is either 2 or 1.
    # mj is always 2 (since m=4 split into 2+2).
    # pk is 2, 2, or 1.
    
    # Blocks:
    # 1. ni=2, mj=2, pk=2 (first p-block) -> Rank 7
    # 2. ni=2, mj=2, pk=2 (second p-block) -> Rank 7
    # 3. ni=2, mj=2, pk=1 -> Rank 4
    # 4. ni=1, mj=2, pk=2 (first p-block) -> Rank 4
    # 5. ni=1, mj=2, pk=2 (second p-block) -> Rank 4
    # 6. ni=1, mj=2, pk=1 -> Rank 2
    
    # Wait, m is split into two blocks of size 2. So for each ni and pk, there are 2 mj blocks.
    
    # Let's enumerate:
    # ni=2:
    #   mj=0 (indices 0,1), pk=0 (indices 0,1) -> 2x2x2 (7)
    #   mj=1 (indices 2,3), pk=0 (indices 0,1) -> 2x2x2 (7)
    #   mj=0 (indices 0,1), pk=1 (indices 2,3) -> 2x2x2 (7)
    #   mj=1 (indices 2,3), pk=1 (indices 2,3) -> 2x2x2 (7)
    #   mj=0 (indices 0,1), pk=2 (index 4) -> 2x2x1 (4)
    #   mj=1 (indices 2,3), pk=2 (index 4) -> 2x2x1 (4)
    # ni=1:
    #   mj=0 (indices 0,1), pk=0 (indices 0,1) -> 1x2x2 (4)
    #   mj=1 (indices 2,3), pk=0 (indices 0,1) -> 1x2x2 (4)
    #   mj=0 (indices 0,1), pk=1 (indices 2,3) -> 1x2x2 (4)
    #   mj=1 (indices 2,3), pk=1 (indices 2,3) -> 1x2x2 (4)
    #   mj=0 (indices 0,1), pk=2 (index 4) -> 1x2x1 (2)
    #   mj=1 (indices 2,3), pk=2 (index 4) -> 1x2x1 (2)
    
    # Total Rank = 4*7 + 2*4 + 4*4 + 2*2 = 28 + 8 + 16 + 4 = 56.
    
    # This matches the inspiration's estimate of 56.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Define offsets
    # n=3: ni=2 (rows 0,1), ni=1 (row 2)
    # m=4: mj=2 (cols 0,1), mj=2 (cols 2,3)
    # p=5: pk=2 (cols 0,1), pk=2 (cols 2,3), pk=1 (col 4)
    
    # Helper to get global indices
    def get_u_indices(ni, mj):
        # ni: 0 or 1 (block index)
        # mj: 0 or 1 (block index)
        # Actual row start: ni * 2 (if ni=0, start 0; if ni=1, start 2)
        # Actual col start: mj * 2
        row_start = ni * 2
        col_start = mj * 2
        return row_start, col_start
        
    def get_v_indices(mj, pk):
        # mj: 0 or 1
        # pk: 0, 1, or 2
        # Actual row start: mj * 2
        # Actual col start: depends on pk
        row_start = mj * 2
        if pk == 0:
            col_start = 0
        elif pk == 1:
            col_start = 2
        else:
            col_start = 4
        return row_start, col_start
        
    def get_w_indices(pk, ni):
        # pk: 0, 1, or 2
        # ni: 0 or 1
        # Actual row start: depends on pk
        # Actual col start: ni * 2
        if pk == 0:
            row_start = 0
        elif pk == 1:
            row_start = 2
        else:
            row_start = 4
        col_start = ni * 2
        return row_start, col_start

    # Loop over blocks
    for ni in range(2): # 0, 1
        for mj in range(2): # 0, 1
            for pk in range(3): # 0, 1, 2
                
                u_row_start, u_col_start = get_u_indices(ni, mj)
                v_row_start, v_col_start = get_v_indices(mj, pk)
                w_row_start, w_col_start = get_w_indices(pk, ni)
                
                ni_size = 2 if ni == 0 else 1
                mj_size = 2
                pk_size = 2 if pk < 2 else 1
                
                # Select decomposition
                if ni_size == 2 and mj_size == 2 and pk_size == 2:
                    U_b, V_b, W_b = _strassen_2x2x2()
                elif ni_size == 2 and mj_size == 2 and pk_size == 1:
                    U_b, V_b, W_b = _schoolbook_2x2x1()
                elif ni_size == 1 and mj_size == 2 and pk_size == 2:
                    U_b, V_b, W_b = _schoolbook_1x2x2()
                elif ni_size == 1 and mj_size == 2 and pk_size == 1:
                    U_b, V_b, W_b = _schoolbook_1x2x1()
                else:
                    # Fallback to schoolbook if dimensions don't match expected
                    U_b, V_b, W_b = _schoolbook(ni_size, mj_size, pk_size)
                
                _add_block(U_parts, V_parts, W_parts, 
                           u_row_start, v_row_start, w_row_start,
                           ni_size, mj_size, pk_size,
                           U_b, V_b, W_b)
                
                if time.time() > deadline - 5:
                    break
            
            if time.time() > deadline - 5:
                break
        
        if time.time() > deadline - 5:
            break

    # Convert to matrices
    if U_parts:
        U = np.array(U_parts).T
        V = np.array(V_parts).T
        W = np.array(W_parts).T
    else:
        U, V, W = _schoolbook(n, m, p)
    
    # Verify
    if _verify(U, V, W, n, m, p, tol=1e-7):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook if verification fails
        U, V, W = _schoolbook(n, m, p)
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

# EVOLVE-BLOCK-END

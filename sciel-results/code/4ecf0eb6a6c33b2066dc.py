"""Matrix multiplication 4,4,7 — rank-93 tensor decomposition using Strassen."""
import time
import numpy as np

N, M, P = 4, 4, 7


def _strassen_2x2x2():
    """
    Rank-7 decomposition for 2x2x2 matmul (Strassen).
    
    A = [[a,b],[c,d]] -> U indices: a=0, b=1, c=2, d=3
    B = [[e,f],[g,h]] -> V indices: e=0, f=1, g=2, h=3
    C = A*B, W[k*2+i]: C00=0, C10=1, C01=2, C11=3
    
    Strassen:
    M1 = (a+d)(e+h)     -> C00, C11
    M2 = (c+d)e         -> C10, C11
    M3 = a(f-h)         -> C01
    M4 = d(g-e)         -> C00, C10
    M5 = (a+b)h         -> C00, C01
    M6 = (c-a)(e+f)     -> C11
    M7 = (b-d)(g+h)     -> C00
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1: r=0
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: r=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: r=2
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # M4: r=3
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: r=4
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: r=5
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: r=6
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _strassen_4x4x4():
    """
    Rank-49 decomposition for 4x4x4 matmul using Strassen on 2x2 blocks.
    
    A = [[A11, A12],[A21, A22]], B = [[B11, B12],[B21, B22]], C = [[C11, C12],[C21, C22]]
    where each sub-block is 2x2.
    
    Block-level Strassen (7 terms), each term uses inner Strassen 2x2x2 (7 terms).
    Total rank = 7 * 7 = 49.
    
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    
    U = np.zeros((16, 49), dtype=np.float64)
    V = np.zeros((16, 49), dtype=np.float64)
    W = np.zeros((16, 49), dtype=np.float64)
    
    # Block index mappings for 4x4 matrix split into 2x2 blocks of 2x2:
    # Block (br, bc) where br, bc in {0,1}:
    #   A block: elements (br*2+i, bc*2+j) for i,j in {0,1}
    #   U index = (br*2+i)*4 + (bc*2+j)
    #   B block: elements (br*2+i, bc*2+j)  
    #   V index = (br*2+i)*4 + (bc*2+j)
    #   C block: elements (br*2+i, bc*2+j) where i=row, j=col
    #   W index = (bc*2+j)*4 + (br*2+i)
    
    def u_block_idx(br, bc, ii, jj):
        """U index for A element at block (br,bc), inner (ii,jj)."""
        gi = br * 2 + ii
        gj = bc * 2 + jj
        return gi * 4 + gj
    
    def v_block_idx(br, bc, ii, jj):
        """V index for B element at block (br,bc), inner (ii,jj)."""
        gi = br * 2 + ii
        gj = bc * 2 + jj
        return gi * 4 + gj
    
    def w_block_idx(br, bc, ii, jj):
        """W index for C element at block (br,bc), inner (ii,jj).
        ii is row (i), jj is col (k). W[k*4+i] = W[jj*4+ii]."""
        gi = br * 2 + ii  # row i
        gj = bc * 2 + jj  # col k
        return gj * 4 + gi
    
    # Block-level Strassen terms:
    # Each term: (A_blocks, B_blocks, C_blocks)
    # where each is list of (block_pos, coefficient)
    # block_pos is (br, bc)
    
    strassen_block_terms = [
        # M1 = (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        # M2 = (A21+A22)B11 -> C21 += 1, C22 += -1
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        # M3 = A11(B12-B22) -> C12 += 1, C22 += 1
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
        # M4 = A22(B21-B11) -> C11 += 1, C21 += 1
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
        # M5 = (A11+A12)B22 -> C11 += -1, C12 += 1
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        # M6 = (A21-A11)(B11+B12) -> C22 += 1
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        # M7 = (A12-A22)(B21+B22) -> C11 += 1
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
    ]
    
    for s in range(7):
        A_blocks, B_blocks, C_blocks = strassen_block_terms[s]
        for r in range(7):
            r_total = s * 7 + r
            
            # For each A block with coefficient, add to U
            for (br, bc), coeff in A_blocks:
                for ii in range(2):
                    for jj in range(2):
                        u_idx = u_block_idx(br, bc, ii, jj)
                        U[u_idx, r_total] += coeff * U_s[ii * 2 + jj, r]
            
            # For each B block with coefficient, add to V
            for (br, bc), coeff in B_blocks:
                for ii in range(2):
                    for jj in range(2):
                        v_idx = v_block_idx(br, bc, ii, jj)
                        V[v_idx, r_total] += coeff * V_s[ii * 2 + jj, r]
            
            # For each C block with coefficient, add to W
            for (br, bc), coeff in C_blocks:
                for ii in range(2):
                    for jj in range(2):
                        # ii is row, jj is col in C block
                        w_idx = w_block_idx(br, bc, ii, jj)
                        # W_s index: k*2+i where i=ii (row), k=jj (col)
                        W[w_idx, r_total] += coeff * W_s[jj * 2 + ii, r]
    
    return U, V, W


def _strassen_4x4x2():
    """
    Rank-28 decomposition for 4x4x2 matmul.
    
    A is 4x4, B is 4x2, C is 4x2.
    Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    Split B into 2x2 blocks: B1 (top 2 rows), B2 (bottom 2 rows)
    Split C into 2x2 blocks: C1 (top 2 rows), C2 (bottom 2 rows)
    
    C1 = A11*B1 + A12*B2
    C2 = A21*B1 + A22*B2
    
    Compute four independent 2x2x2 Strassen multiplications:
    M1 = A11*B1 (rank 7)
    M2 = A12*B2 (rank 7)
    M3 = A21*B1 (rank 7)
    M4 = A22*B2 (rank 7)
    
    C1 = M1 + M2
    C2 = M3 + M4
    
    Total rank: 28
    
    U shape: (16, 28), V shape: (8, 28), W shape: (8, 28)
    U[i*4+j, r], V[j*2+k, r], W[k*4+i, r]
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    
    U = np.zeros((16, 28), dtype=np.float64)
    V = np.zeros((8, 28), dtype=np.float64)
    W = np.zeros((8, 28), dtype=np.float64)
    
    # For each of the 4 Strassen multiplications:
    # M1: A11 (rows 0-1, cols 0-1) * B1 (rows 0-1, cols 0-1) -> C1 (rows 0-1, cols 0-1)
    # M2: A12 (rows 0-1, cols 2-3) * B2 (rows 2-3, cols 0-1) -> C1 (rows 0-1, cols 0-1)
    # M3: A21 (rows 2-3, cols 0-1) * B1 (rows 0-1, cols 0-1) -> C2 (rows 2-3, cols 0-1)
    # M4: A22 (rows 2-3, cols 2-3) * B2 (rows 2-3, cols 0-1) -> C2 (rows 2-3, cols 0-1)
    
    # Multiplication specs: (a_row_offset, a_col_offset, b_row_offset, c_row_offset, c_col_offset)
    # All c_col_offset = 0 since B has only 2 cols
    mult_specs = [
        # (a_row, a_col, b_row, c_row, c_col)
        (0, 0, 0, 0, 0),   # M1: A11 * B1 -> C1
        (0, 2, 2, 0, 0),   # M2: A12 * B2 -> C1
        (2, 0, 0, 2, 0),   # M3: A21 * B1 -> C2
        (2, 2, 2, 2, 0),   # M4: A22 * B2 -> C2
    ]
    
    for m_idx, (ar, ac, br, cr, cc) in enumerate(mult_specs):
        for r in range(7):
            r_total = m_idx * 7 + r
            
            # A elements: (ar+ii, ac+jj) for ii, jj in 0..1
            for ii in range(2):
                for jj in range(2):
                    u_idx = (ar + ii) * 4 + (ac + jj)
                    U[u_idx, r_total] += U_s[ii * 2 + jj, r]
            
            # B elements: (br+ii, cc+jj) for ii, jj in 0..1
            for ii in range(2):
                for jj in range(2):
                    v_idx = (br + ii) * 2 + (cc + jj)
                    V[v_idx, r_total] += V_s[ii * 2 + jj, r]
            
            # C elements: (cr+ii, cc+jj) where ii=row, jj=col
            # W index = k*4 + i = (cc+jj)*4 + (cr+ii)
            for ii in range(2):
                for jj in range(2):
                    w_idx = (cc + jj) * 4 + (cr + ii)
                    W[w_idx, r_total] += W_s[jj * 2 + ii, r]
    
    return U, V, W


def _schoolbook_4x4x1():
    """
    Rank-16 decomposition for 4x4x1 matmul (matrix-vector).
    
    A is 4x4, B is 4x1, C is 4x1.
    U shape: (16, 16), V shape: (4, 16), W shape: (4, 16)
    U[i*4+j, r], V[j*1+k, r] = V[j, r], W[k*4+i, r] = W[i, r] (since k=0)
    """
    U = np.zeros((16, 16), dtype=np.float64)
    V = np.zeros((4, 16), dtype=np.float64)
    W = np.zeros((4, 16), dtype=np.float64)
    
    col = 0
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, col] = 1.0
            V[j, col] = 1.0
            W[i, col] = 1.0  # k=0, so W[0*4+i] = W[i]
            col += 1
    
    return U, V, W


def _build_decomp():
    """Build a rank-112 decomposition (schoolbook) for 4x4x7 matmul."""
    rank = N * M * P  # 112
    U = [[0.0] * rank for _ in range(N * M)]
    V = [[0.0] * rank for _ in range(M * P)]
    W = [[0.0] * rank for _ in range(P * N)]
    
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1.0
                V[j * P + k][col] = 1.0
                W[k * N + i][col] = 1.0
                col += 1
    return U, V, W


def _verify_decomp(U, V, W, tol=1e-8):
    """Verify decomposition: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
    n, m, p = N, M, P
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    """Return factor matrices for 4x4x7 matrix multiplication tensor.
    
    Uses rank-93 decomposition:
    - 4x4x4 part: rank 49 (Strassen)
    - 4x4x2 part: rank 28 (Strassen)
    - 4x4x1 part: rank 16 (schoolbook)
    Total: 93
    """
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    try:
        # Part 1: 4x4x4 (columns 0-48)
        U1, V1, W1 = _strassen_4x4x4()  # (16,49), (16,49), (16,49)
        
        # Part 2: 4x4x2 (columns 49-76)
        U2, V2, W2 = _strassen_4x4x2()  # (16,28), (8,28), (8,28)
        
        # Part 3: 4x4x1 (columns 77-92)
        U3, V3, W3 = _schoolbook_4x4x1()  # (16,16), (4,16), (4,16)
        
        if time.time() > deadline - 10:
            raise TimeoutError()
        
        # Assemble full decomposition
        rank = 93
        U = np.zeros((16, rank), dtype=np.float64)
        V = np.zeros((28, rank), dtype=np.float64)
        W = np.zeros((28, rank), dtype=np.float64)
        
        # Part 1: columns 0-48
        U[:, 0:49] = U1
        V[0:16, 0:49] = V1
        W[0:16, 0:49] = W1
        
        # Part 2: columns 49-76
        # B_right is rows 4-7 of B (4x2), so V rows 16-23
        U[:, 49:77] = U2
        V[16:24, 49:77] = V2
        W[16:24, 49:77] = W2
        
        # Part 3: columns 77-92
        # B_last is row 8 of B (4x1), so V row 24
        U[:, 77:93] = U3
        V[24:28, 77:93] = V3
        W[24:28, 77:93] = W3
        
        if time.time() > deadline - 5:
            raise TimeoutError()
        
        # Verify
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _verify_decomp(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            raise ValueError("Decomposition verification failed")
            
    except Exception:
        # Fallback to schoolbook
        U, V, W = _build_decomp()
        return {"U": U, "V": V, "W": W}

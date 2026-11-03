"""Matrix multiplication 4,4,7 — rank-minimal tensor decomposition."""
import time
import numpy as np

N, M, P = 4, 4, 7


def _strassen_2x2x2():
    """Rank-7 decomposition for 2x2x2 matmul (Strassen).
    
    A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    U indices (i*2+j): a=0, b=1, c=2, d=3
    V indices (j*2+k): e=0, f=1, g=2, h=3
    W indices (k*2+i): C00=0, C01=2, C10=1, C11=3
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a+d)(e+h) -> C00 += 1, C11 += 1
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (c+d)e -> C10 += 1, C11 += -1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # M3 = a(f-h) -> C01 += 1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1
    # M4 = d(g-e) -> C00 += 1, C10 += 1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    # M5 = (a+b)h -> C00 += -1, C01 += 1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6 = (c-a)(e+f) -> C11 += 1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (b-d)(g+h) -> C00 += 1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    
    return U, V, W


def _strassen_4x4x4():
    """Rank-49 decomposition for 4x4x4 matmul via Strassen on 2x2 blocks.
    
    A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]], C = [[C11,C12],[C21,C22]]
    Each block is 2x2. Strassen at block level (7 terms), each computed via Strassen 2x2x2 (rank 7).
    Total rank = 7*7 = 49.
    
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    
    U = np.zeros((16, 49), dtype=np.float64)
    V = np.zeros((16, 49), dtype=np.float64)
    W = np.zeros((16, 49), dtype=np.float64)
    
    # Block-level Strassen terms:
    # M1 = (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
    # M2 = (A21+A22)B11    -> C21 += 1, C22 += -1
    # M3 = A11(B12-B22)     -> C12 += 1, C22 += 1
    # M4 = A22(B21-B11)     -> C11 += 1, C21 += 1
    # M5 = (A11+A12)B22     -> C11 += -1, C12 += 1
    # M6 = (A21-A11)(B11+B12) -> C22 += 1
    # M7 = (A12-A22)(B21+B22) -> C11 += 1
    
    # For each term s, define:
    #   A_contrib: list of (block_row, block_col, coeff)
    #   B_contrib: list of (block_row, block_col, coeff)
    #   C_contrib: list of (block_row, block_col, coeff)
    
    strassen_block_terms = [
        # s=0: M1 = (A11+A22)(B11+B22)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # s=1: M2 = (A21+A22)B11
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # s=2: M3 = A11(B12-B22)
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # s=3: M4 = A22(B21-B11)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # s=4: M5 = (A11+A12)B22
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # s=5: M6 = (A21-A11)(B11+B12)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # s=6: M7 = (A12-A22)(B21+B22)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    for s in range(7):
        A_contrib, B_contrib, C_contrib = strassen_block_terms[s]
        for r in range(7):
            r_total = s * 7 + r
            
            # For each A block contribution, map inner 2x2 indices to global
            for (br, bc, coeff) in A_contrib:
                for ii in range(2):
                    for jj in range(2):
                        inner_idx = ii * 2 + jj
                        gi = br * 2 + ii
                        gj = bc * 2 + jj
                        u_idx = gi * 4 + gj
                        U[u_idx, r_total] += coeff * U_s[inner_idx, r]
            
            # For each B block contribution
            for (br, bc, coeff) in B_contrib:
                for ii in range(2):
                    for jj in range(2):
                        inner_idx = ii * 2 + jj
                        gi = br * 2 + ii
                        gj = bc * 2 + jj
                        v_idx = gi * 4 + gj
                        V[v_idx, r_total] += coeff * V_s[inner_idx, r]
            
            # For each C block contribution
            # C[i,k] -> W[k*4+i], where i is row, k is col
            # Block at (br, bc): rows br*2..br*2+1, cols bc*2..bc*2+1
            # Inner element (ii, jj) in block -> global row = br*2+ii, global col = bc*2+jj
            # W index = (bc*2+jj)*4 + (br*2+ii)
            for (br, bc, coeff) in C_contrib:
                for ii in range(2):
                    for jj in range(2):
                        # Inner W index: W_s[k_inner*2 + i_inner] where i_inner=ii (row), k_inner=jj (col)
                        inner_idx = jj * 2 + ii
                        gi = br * 2 + ii
                        gj = bc * 2 + jj
                        w_idx = gj * 4 + gi
                        W[w_idx, r_total] += coeff * W_s[inner_idx, r]
    
    return U, V, W


def _schoolbook_4x4x3():
    """Schoolbook rank-48 decomposition for 4x4x3 matmul.
    
    A: 4x4, B: 4x3, C: 4x3
    U shape: (16, 48), V shape: (12, 48), W shape: (12, 48)
    U[i*4+j, r], V[j*3+k, r], W[k*4+i, r]
    """
    U = np.zeros((16, 48), dtype=np.float64)
    V = np.zeros((12, 48), dtype=np.float64)
    W = np.zeros((12, 48), dtype=np.float64)
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i * 4 + j, col] = 1.0
                V[j * 3 + k, col] = 1.0
                W[k * 4 + i, col] = 1.0
                col += 1
    
    return U, V, W


def _construct_4x4x7_strassen():
    """Rank-97 decomposition for 4x4x7 matmul.
    
    Split B (4x7) into B_left (4x4) and B_right (4x3).
    C = A * B_left + A * B_right
    
    A * B_left: 4x4x4, rank 49 (Strassen)
    A * B_right: 4x4x3, rank 48 (schoolbook)
    Total rank: 97
    """
    U1, V1, W1 = _strassen_4x4x4()  # (16,49), (16,49), (16,49)
    U2, V2, W2 = _schoolbook_4x4x3()  # (16,48), (12,48), (12,48)
    
    rank = 97
    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((28, rank), dtype=np.float64)
    W = np.zeros((28, rank), dtype=np.float64)
    
    # First 49 columns: Strassen 4x4x4 for B_left
    U[:, :49] = U1
    V[:16, :49] = V1
    W[:16, :49] = W1
    
    # Next 48 columns: schoolbook 4x4x3 for B_right
    U[:, 49:] = U2
    V[16:, 49:] = V2
    W[16:, 49:] = W2
    
    return U.tolist(), V.tolist(), W.tolist()


def _verify_decomp(U, V, W, tol=1e-8):
    """Verify decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)"""
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


def _build_decomp():
    """Schoolbook rank-112 decomposition for 4x4x7 matmul."""
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


def construct():
    """Return factor matrices for 4x4x7 matrix multiplication tensor."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try Strassen-based decomposition (rank 97)
    try:
        U, V, W = _construct_4x4x7_strassen()
        if _verify_decomp(U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 112)
    U, V, W = _build_decomp()
    return {"U": U, "V": V, "W": W}

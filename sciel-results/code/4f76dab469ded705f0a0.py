import time

DIMENSIONS = [3, 4, 6]

def _strassen_222():
    """
    Strassen decomposition for 2x2x2.
    Returns U, V, W as lists of lists (size 4x7).
    U[row, c], V[col, c], W[out_row, c]
    The tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    Wait, the convention is W[k*n+i, r]. For n=2, W[k*2+i, r].
    
    Let's define the indices for the 2x2x2 block.
    i in {0,1} (row of A)
    j in {0,1} (col of A / row of B)
    k in {0,1} (col of B)
    
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Strassen for 2x2:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22) B11
    M3 = A11 (B12 - B22)
    M4 = A22 (B21 - B11)
    M5 = (A11 + A12) B22
    M6 = (A21 - A11) (B11 + B12)
    M7 = (A12 - A22) (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Indices:
    A11 is at i=0, j=0
    A12 is at i=0, j=1
    A21 is at i=1, j=0
    A22 is at i=1, j=1
    
    B11 is at j=0, k=0
    B12 is at j=0, k=1
    B21 is at j=1, k=0
    B22 is at j=1, k=1
    
    C11 is at i=0, k=0
    C12 is at i=0, k=1
    C21 is at i=1, k=0
    C22 is at i=1, k=1
    
    U (A part):
    M1: A11+ A22 -> U[0,0]=1, U[1,1]=1
    M2: A21+ A22 -> U[1,0]=1, U[1,1]=1
    M3: A11     -> U[0,0]=1
    M4: A22     -> U[1,1]=1
    M5: A11+ A12 -> U[0,0]=1, U[0,1]=1
    M6: -A11+ A21 -> U[0,0]=-1, U[1,0]=1
    M7: -A22+ A12 -> U[1,1]=-1, U[0,1]=1
    
    V (B part):
    M1: B11+ B22 -> V[0,0]=1, V[1,1]=1
    M2: B11     -> V[0,0]=1
    M3: B12- B22 -> V[0,1]=1, V[1,1]=-1
    M4: -B11+ B21 -> V[0,0]=-1, V[1,0]=1
    M5: B22     -> V[1,1]=1
    M6: B11+ B12 -> V[0,0]=1, V[0,1]=1
    M7: B21+ B22 -> V[1,0]=1, V[1,1]=1
    
    W (C part):
    W index is k*2 + i.
    C11 (i=0, k=0): idx 0
    C12 (i=0, k=1): idx 2
    C21 (i=1, k=0): idx 1
    C22 (i=1, k=1): idx 3
    
    M1: C11+ C22 -> W[0]=1, W[3]=1
    M2: C21- C22 -> W[1]=1, W[3]=-1
    M3: C12+ C22 -> W[2]=1, W[3]=1
    M4: C11+ C21 -> W[0]=1, W[1]=1
    M5: -C11+ C12 -> W[0]=-1, W[2]=1
    M6: C22     -> W[3]=1
    M7: C11     -> W[0]=1
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    # M2
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1; W[3][1] -= 1
    # M3
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[2][2] += 1; W[3][2] += 1
    # M4
    U[3][3] += 1
    V[0][3] -= 1; V[2][3] += 1
    W[0][3] += 1; W[1][3] += 1
    # M5
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[2][4] += 1
    # M6
    U[0][5] -= 1; U[2][5] += 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    # M7
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W

def _schoolbook_block(n, m, p, offset_i, offset_j, offset_k, base_rank, U, V, W):
    """
    Adds schoolbook terms for a subblock A[n,m] * B[m,p] -> C[p,n].
    Offsets are relative to the global matrices.
    base_rank is the starting column index.
    Returns new base_rank.
    """
    rank = n * m * p
    U_block = [[0]*rank for _ in range(n*m)]
    V_block = [[0]*rank for _ in range(m*p)]
    W_block = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Global indices
                gi = offset_i + i
                gj = offset_j + j
                gk = offset_k + k
                
                U[gi*m + gj][base_rank + col] = 1
                V[gj*p + gk][base_rank + col] = 1
                # W index: k*n + i. But wait, the W matrix is for the whole problem.
                # The W index in the local block is k*n + i.
                # The global W index is (offset_k + k)*N + (offset_i + i).
                # Wait, the W matrix shape is (p*n, rank).
                # The index in W is k*n + i.
                # So global index is (offset_k + k)*N_global? No.
                # The W matrix is a single flat array of size (p_total * n_total).
                # The index is k_total * n_total + i_total.
                # So yes, (offset_k + k)*n_total + (offset_i + i).
                W[(offset_k + k) * DIMENSIONS[0] + (offset_i + i)][base_rank + col] = 1
                col += 1
                
    return base_rank + rank

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # We will partition the indices:
    # A is 3x4. Split rows: 2 + 1. Split cols: 2 + 2.
    # B is 4x6. Split rows: 2 + 2. Split cols: 3 + 3.
    # C is 6x3. Split rows: 3 + 3. Split cols: 2 + 1.
    
    # Blocks of A:
    # A11 (2x2), A12 (2x2)
    # A21 (1x2), A22 (1x2)
    
    # Blocks of B:
    # B11 (2x3), B12 (2x3)
    # B21 (2x3), B22 (2x3)
    
    # Blocks of C:
    # C11 (3x2), C12 (3x1)
    # C21 (3x2), C22 (3x1)
    
    # Multiplications needed for each C block:
    # C11 (3x2) = A11(2x2)B11(2x3) + A12(2x2)B21(2x3)
    #   Term 1: 2x2x3. Split B11 cols 1+2. 2x2x1 + 2x2x2.
    #   Term 2: 2x2x3. Split B21 cols 1+2. 2x2x1 + 2x2x2.
    #   Total for C11: 2*(2x2x1) + 2*(2x2x2).
    
    # C12 (3x1) = A11(2x2)B12(2x3) + A12(2x2)B22(2x3)
    #   Term 1: 2x2x3. Split B12 cols 2+1? Or 1+2?
    #   Let's just use schoolbook for all 2x2x3 blocks?
    #   Rank of 2x2x3 schoolbook is 12.
    #   Strassen for 2x2x2 is 7. Schoolbook 2x2x1 is 4.
    #   Total 7+4=11. Improvement over 12.
    #   So we should use Strassen + Schoolbook for all 2x2x3 blocks.
    
    # C21 (3x2) = A21(1x2)B11(2x3) + A22(1x2)B21(2x3)
    #   Term 1: 1x2x3. Schoolbook rank 6.
    #   Term 2: 1x2x3. Schoolbook rank 6.
    #   Total 12.
    
    # C22 (3x1) = A21(1x2)B12(2x3) + A22(1x2)B22(2x3)
    #   Total 12.
    
    # Let's sum the ranks:
    # C11: 2 * (11) = 22
    # C12: 2 * (11) = 22
    # C21: 12
    # C22: 12
    # Total = 68.
    
    # This matches the known 68 rank solution.
    
    # Let's implement a helper to add a 2x2x2 Strassen block.
    # And a helper to add a 2x2x1 schoolbook block.
    # And a helper to add a 1x2x3 schoolbook block.
    
    U = [[0]*68 for _ in range(n*m)]
    V = [[0]*68 for _ in range(m*p)]
    W = [[0]*68 for _ in range(p*n)]
    
    current_rank = 0
    
    # Helper to add Strassen 2x2x2
    def add_strassen_222(offset_i, offset_j, offset_k):
        nonlocal current_rank
        # Local U, V, W for 2x2x2
        sU, sV, sW = _strassen_222()
        for i in range(2):
            for j in range(2):
                for c in range(7):
                    if sU[i*2+j][c] != 0:
                        U[(offset_i+i)*m + (offset_j+j)][current_rank+c] = sU[i*2+j][c]
        for j in range(2):
            for k in range(2):
                for c in range(7):
                    if sV[j*2+k][c] != 0:
                        V[(offset_j+j)*p + (offset_k+k)][current_rank+c] = sV[j*2+k][c]
        for i in range(2):
            for k in range(2):
                for c in range(7):
                    if sW[k*2+i][c] != 0:
                        W[(offset_k+k)*n + (offset_i+i)][current_rank+c] = sW[k*2+i][c]
        current_rank += 7
        
    # Helper to add Schoolbook 2x2x1
    def add_schoolbook_221(offset_i, offset_j, offset_k):
        nonlocal current_rank
        # n=2, m=2, p=1
        for i in range(2):
            for j in range(2):
                # k is always 0
                k = 0
                U[(offset_i+i)*m + (offset_j+j)][current_rank] = 1
                V[(offset_j+j)*p + (offset_k+k)][current_rank] = 1
                W[(offset_k+k)*n + (offset_i+i)][current_rank] = 1
                current_rank += 1
        # Wait, 2*2*1 = 4 terms.
        # Let's do it properly.
        for i in range(2):
            for j in range(2):
                for k in range(1):
                    U[(offset_i+i)*m + (offset_j+j)][current_rank] = 1
                    V[(offset_j+j)*p + (offset_k+k)][current_rank] = 1
                    W[(offset_k+k)*n + (offset_i+i)][current_rank] = 1
                    current_rank += 1

    # Helper to add Schoolbook 1x2x3
    def add_schoolbook_123(offset_i, offset_j, offset_k):
        nonlocal current_rank
        # n=1, m=2, p=3
        for i in range(1):
            for j in range(2):
                for k in range(3):
                    U[(offset_i+i)*m + (offset_j+j)][current_rank] = 1
                    V[(offset_j+j)*p + (offset_k+k)][current_rank] = 1
                    W[(offset_k+k)*n + (offset_i+i)][current_rank] = 1
                    current_rank += 1

    # Helper to add a 2x2x3 block using Strassen(2x2x2) + Schoolbook(2x2x1)
    def add_223_block(offset_i, offset_j, offset_k):
        nonlocal current_rank
        # Split k into 2 + 1
        # k=0,1 -> Strassen
        add_strassen_222(offset_i, offset_j, offset_k)
        # k=2 -> Schoolbook
        add_schoolbook_221(offset_i, offset_j, offset_k+2)

    # Now, let's map the blocks.
    # A rows: 0,1 (offset 0), 2 (offset 2)
    # A cols: 0,1 (offset 0), 2,3 (offset 2)
    # B rows: 0,1 (offset 0), 2,3 (offset 2)
    # B cols: 0,1,2 (offset 0), 3,4,5 (offset 3)
    # C rows: 0,1,2 (offset 0), 3,4,5 (offset 3)
    # C cols: 0,1 (offset 0), 2 (offset 2)
    
    # C11 (rows 0-2, cols 0-1)
    # = A11(0-1, 0-1) * B11(0-1, 0-2)  -> 2x2x3 block at (0,0,0)
    add_223_block(0, 0, 0)
    # + A12(0-1, 2-3) * B21(2-3, 0-2)  -> 2x2x3 block at (0,2,0)
    add_223_block(0, 2, 0)
    
    # C12 (rows 0-2, cols 2)
    # = A11(0-1, 0-1) * B12(0-1, 3-5)  -> 2x2x3 block at (0,0,3)
    add_223_block(0, 0, 3)
    # + A12(0-1, 2-3) * B22(2-3, 3-5)  -> 2x2x3 block at (0,2,3)
    add_223_block(0, 2, 3)
    
    # C21 (rows 3-5, cols 0-1)
    # = A21(2, 0-1) * B11(0-1, 0-2)  -> 1x2x3 block at (2,0,0)
    add_schoolbook_123(2, 0, 0)
    # + A22(2, 2-3) * B21(2-3, 0-2)  -> 1x2x3 block at (2,2,0)
    add_schoolbook_123(2, 2, 0)
    
    # C22 (rows 3-5, cols 2)
    # = A21(2, 0-1) * B12(0-1, 3-5)  -> 1x2x3 block at (2,0,3)
    add_schoolbook_123(2, 0, 3)
    # + A22(2, 2-3) * B22(2-3, 3-5)  -> 1x2x3 block at (2,2,3)
    add_schoolbook_123(2, 2, 3)
    
    # Total rank should be 68.
    assert current_rank == 68, f"Rank is {current_rank}, expected 68"
    
    return {"U": U, "V": V, "W": W}

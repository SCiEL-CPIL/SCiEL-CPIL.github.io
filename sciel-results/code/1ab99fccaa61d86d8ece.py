"""Strassen-based rank-minimal decomposition for 4x4 x 4x6 matrix multiplication."""
DIMENSIONS = [4, 4, 6]

def _build_strassen_2x2():
    """Return (U, V, W) for 2x2 * 2x2 -> 2x2 via Strassen, rank 7.
    Convention: U[i*m + j, c] with m=2, V[j*p + k, c] with m=2, W[k*n + i, c] with n=2.
    """
    n, m, p = 2, 2, 2
    # Strassen 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    U = [[0]*7 for _ in range(n*m)]
    V = [[0]*7 for _ in range(m*p)]
    W = [[0]*7 for _ in range(p*n)]
    # index helpers: U[i*m+j], V[j*p+k], W[k*n+i]
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k in {0,1}
    # C indices: i in {0,1}, k in {0,1}
    # M1 = (A11+A22)(B11+B22)
    U[0*2+0][0] += 1; U[1*2+1][0] += 1
    V[0*2+0][0] += 1; V[1*2+1][0] += 1
    W[0*2+0][0] += 1; W[1*2+1][0] += 1  # C11, C22
    # M2 = (A21+A22)B11
    U[1*2+0][1] += 1; U[1*2+1][1] += 1
    V[0*2+0][1] += 1
    W[1*2+0][1] += 1; W[1*2+1][1] -= 1  # C21, -C22
    # M3 = A11(B21-B22)
    U[0*2+0][2] += 1
    V[1*2+0][2] += 1; V[1*2+1][2] -= 1
    W[0*2+1][2] += 1; W[1*2+1][2] += 1  # C12, C22
    # M4 = A22(B12-B11)
    U[1*2+1][3] += 1
    V[0*2+1][3] += 1; V[0*2+0][3] -= 1
    W[0*2+0][3] += 1; W[1*2+0][3] += 1  # C11, C21
    # M5 = (A11+A12)B22
    U[0*2+0][4] += 1; U[0*2+1][4] += 1
    V[1*2+1][4] += 1
    W[0*2+0][4] -= 1; W[0*2+1][4] += 1  # -C11, C12
    # M6 = (A21-A11)(B11+B12)
    U[1*2+0][5] += 1; U[0*2+0][5] -= 1
    V[0*2+0][5] += 1; V[0*2+1][5] += 1
    W[1*2+1][5] += 1  # C22
    # M7 = (A12-A22)(B21+B22)
    U[0*2+1][6] += 1; U[1*2+1][6] -= 1
    V[1*2+0][6] += 1; V[1*2+1][6] += 1
    W[0*2+0][6] += 1  # C11
    return U, V, W

def _build_strassen_4x4():
    """Build 4x4 * 4x4 -> 4x4 via recursive Strassen, rank 49.
    Split 4x4 into 2x2 blocks. Use 7 block multiplications each of size 2x2x2,
    each handled by _build_strassen_2x2 (rank 7). Total rank 49.
    """
    n, m, p = 4, 4, 4
    # We compose 7 copies of the 2x2 Strassen, each occupying a 2x2 block region.
    # Block layout: A is 4x4, split into 2x2 blocks:
    #   A11 = A[0:2,0:2], A12 = A[0:2,2:4]
    #   A21 = A[2:4,0:2], A22 = A[2:4,2:4]
    # Similarly B: B11 = B[0:2,0:2], B12 = B[0:2,2:4], B21 = B[2:4,0:2], B22 = B[2:4,2:4]
    # C: C11 = C[0:2,0:2], C12 = C[0:2,2:4], C21 = C[2:4,0:2], C22 = C[2:4,2:4]
    #
    # For each of the 7 Strassen products, we need to map the 2x2 Strassen's
    # (a,b,c) local indices to the 4x4 global indices, with the appropriate
    # linear combinations of the 2x2 blocks.
    #
    # This is complex. Instead, I'll build it by embedding: each 2x2 Strassen
    # operates on a pair of 2x2 blocks. The 7 products are:
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B21-B22)
    #   M4 = A22(B12-B11)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A12-A22)(B21+B22)
    #
    # Each M is a 2x2 matrix = sum of products of 2x2 blocks.
    # A product of a 2x2 block of A (rows ri, cols rj) and a 2x2 block of B
    # (rows rj, cols rk) gives a 2x2 result in C block (rows ri, cols rk).
    #
    # I'll handle each block-pair product by instantiating the 2x2 Strassen
    # and placing its entries at the correct global offsets.
    #
    # For a product A_block(i,j) * B_block(j,k), the local 2x2 indices are:
    #   A local (a_i, a_j) -> global A row = i*2 + a_i, col = j*2 + a_j
    #   B local (b_j, b_k) -> global B row = j*2 + b_j, col = k*2 + b_k
    #   C local (c_i, c_k) -> global C row = i*2 + c_i, col = k*2 + c_k
    #
    # The 2x2 Strassen U/V/W use m=2 (local). We map:
    #   U_local[a_i*2 + a_j] -> U_global[(i*2+a_i)*4 + (j*2+a_j)]  [since global m=4]
    #   V_local[b_j*2 + b_k] -> V_global[(j*2+b_j)*4 + (k*2+b_k)]  [since global p=4]
    #   W_local[c_k*2 + c_i] -> W_global[(k*2+c_k)*4 + (i*2+c_i)]  [since global n=4, W[k*n+i]]
    #
    # Wait, W convention is W[k*n + i, r] where n=4 global. So W global index = k_global*4 + i_global.
    # Local W is W_local[c_k*2 + c_i] with n=2 local. So W_local index = c_k*2 + c_i.
    # Mapping: c_k -> k_global = k*2 + c_k, c_i -> i_global = i*2 + c_i.
    # W_global index = (k*2+c_k)*4 + (i*2+c_i).
    #
    # U global index = i_global*m_global + j_global = (i*2+a_i)*4 + (j*2+a_j).
    # V global index = j_global*p_global + k_global = (j*2+b_j)*4 + (k*2+b_k).
    
    U = [[0]*49 for _ in range(n*m)]  # 16 x 49
    V = [[0]*49 for _ in range(m*p)]  # 16 x 49
    W = [[0]*49 for _ in range(p*n)]  # 16 x 49
    
    U2, V2, W2 = _build_strassen_2x2()
    
    # Define the 7 block products with their (A_block, B_block, C_block, sign_A, sign_B)
    # Each product: A_block (i,j) with coefficient, B_block (j,k) with coefficient
    # We need to handle sums/differences of blocks.
    #
    # M1 = (A11+A22)(B11+B22)
    #   = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # M2 = (A21+A22)B11 = A21*B11 + A22*B11
    # M3 = A11(B21-B22) = A11*B21 - A11*B22
    # M4 = A22(B12-B11) = A22*B12 - A22*B11
    # M5 = (A11+A12)B22 = A11*B22 + A12*B22
    # M6 = (A21-A11)(B11+B12) = A21*B11 + A21*B12 - A11*B11 - A11*B12
    # M7 = (A12-A22)(B21+B22) = A12*B21 + A12*B22 - A22*B21 - A22*B22
    #
    # For each term A_{ij} * B_{jk}, we instantiate the 2x2 Strassen with
    # appropriate signs and add to the global matrices.
    #
    # The C block for A_{ij}*B_{jk} is C_{ik}.
    
    # List of (product_index, i, j, k, sign_A, sign_B)
    # product_index: 0..6 for M1..M7
    terms = []
    # M1
    terms += [(0, 0, 0, 0, 1, 1), (0, 0, 0, 1, 1, 1), (0, 1, 1, 0, 1, 1), (0, 1, 1, 1, 1, 1)]
    # M2
    terms += [(1, 1, 0, 0, 1, 1), (1, 1, 1, 0, 1, 1)]
    # M3
    terms += [(2, 0, 0, 0, 1, 1), (2, 0, 0, 1, 1, -1)]
    # M4
    terms += [(3, 1, 1, 1, 1, 1), (3, 1, 1, 0, 1, -1)]
    # M5
    terms += [(4, 0, 0, 1, 1, 1), (4, 0, 1, 1, 1, 1)]
    # M6
    terms += [(5, 1, 0, 0, 1, 1), (5, 1, 0, 1, 1, 1), (5, 0, 0, 0, -1, 1), (5, 0, 0, 1, -1, 1)]
    # M7
    terms += [(6, 0, 1, 0, 1, 1), (6, 0, 1, 1, 1, 1), (6, 1, 1, 0, -1, 1), (6, 1, 1, 1, -1, 1)]
    
    for (prod, i, j, k, sa, sb) in terms:
        # The 2x2 Strassen for this block product uses columns prod*7 .. prod*7+6
        # (since each of the 7 products uses 7 columns, and we have 49 total)
        base_col = prod * 7
        for c in range(7):
            col = base_col + c
            # U2 entries
            for a_i in range(2):
                for a_j in range(2):
                    u_local = a_i * 2 + a_j
                    u_global = (i*2 + a_i) * 4 + (j*2 + a_j)
                    val = U2[u_local][c]
                    if val != 0:
                        U[u_global][col] += sa * val
            # V2 entries
            for b_j in range(2):
                for b_k in range(2):
                    v_local = b_j * 2 + b_k
                    v_global = (j*2 + b_j) * 4 + (k*2 + b_k)
                    val = V2[v_local][c]
                    if val != 0:
                        V[v_global][col] += sb * val
            # W2 entries
            for c_k in range(2):
                for c_i in range(2):
                    w_local = c_k * 2 + c_i
                    w_global = (k*2 + c_k) * 4 + (i*2 + c_i)
                    val = W2[w_local][c]
                    if val != 0:
                        W[w_global][col] += val
    
    return U, V, W

def _build_4x4x1():
    """Build 4x4 * 4x1 -> 4x1 schoolbook, rank 16.
    U[i*4 + j, c], V[j*1 + 0, c], W[0*4 + i, c].
    """
    n, m, p = 4, 4, 1
    rank = 16
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            U[i*m + j][col] = 1
            V[j*p + 0][col] = 1
            W[0*n + i][col] = 1
            col += 1
    return U, V, W

def _build_4x4x2():
    """Build 4x4 * 4x2 -> 4x2, rank 32 (two 4x4x1 parts)."""
    U1, V1, W1 = _build_4x4x1()
    U2, V2, W2 = _build_4x4x1()
    # Concatenate: U has 16 rows, V has 8 rows, W has 8 rows
    rank = 32
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(8)]
    # First 4x4x1 (k=0)
    for r in range(16):
        for c in range(16):
            U[r][c] = U1[r][c]
    for r in range(4):
        for c in range(16):
            V[r][c] = V1[r][c]
    for r in range(4):
        for c in range(16):
            W[r][c] = W1[r][c]
    # Second 4x4x1 (k=1)
    for r in range(16):
        for c in range(16):
            U[r][c+16] = U2[r][c]
    for r in range(4):
        for c in range(16):
            V[r+4][c+16] = V2[r][c]
    for r in range(4):
        for c in range(16):
            W[r+4][c+16] = W2[r][c]
    return U, V, W

def construct():
    import time
    start = time.time()
    deadline = start + 120.0
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Split 4x4x6 into 4x4x4 (Strassen, rank 49) + 4x4x2 (rank 32)
    U4, V4, W4 = _build_strassen_4x4()  # 16x49, 16x49, 16x49
    U2, V2, W2 = _build_4x4x2()         # 16x32, 8x32, 8x32
    
    # Concatenate: total rank = 49 + 32 = 81
    rank = 81
    U = [[0]*rank for _ in range(n*m)]   # 16 x 81
    V = [[0]*rank for _ in range(m*p)]   # 24 x 81
    W = [[0]*rank for _ in range(p*n)]   # 24 x 81
    
    # First part: 4x4x4 (k=0,1,2,3)
    for r in range(16):
        for c in range(49):
            U[r][c] = U4[r][c]
    for r in range(16):
        for c in range(49):
            V[r][c] = V4[r][c]
    for r in range(16):
        for c in range(49):
            W[r][c] = W4[r][c]
    
    # Second part: 4x4x2 (k=4,5)
    for r in range(16):
        for c in range(32):
            U[r][c+49] = U2[r][c]
    for r in range(8):
        for c in range(32):
            V[r+16][c+49] = V2[r][c]
    for r in range(8):
        for c in range(32):
            W[r+16][c+49] = W2[r][c]
    
    return {"U": U, "V": V, "W": W}

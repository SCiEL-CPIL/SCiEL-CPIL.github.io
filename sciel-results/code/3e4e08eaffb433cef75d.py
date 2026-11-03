"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 115  # safety margin before 120s limit

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    # Use Strassen-based decomposition
    # Split p=7 into p1=4 + p2=3
    # 4x4 x 4x4: rank 49 (Strassen)
    # 4x4 x 4x3: rank 48 (schoolbook) or try Strassen-like
    # Total: 49 + 48 = 97
    
    # Build 4x4 x 4x4 Strassen (rank 49)
    # Build 4x4 x 4x3 schoolbook (rank 48)
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Part 1: 4x4 x 4x4 Strassen
    U1, V1, W1 = _strassen_4x4_4x4()
    U_parts.append(U1)
    V_parts.append(V1)
    W_parts.append(W1)
    
    # Part 2: 4x4 x 4x3 schoolbook
    U2, V2, W2 = _schoolbook_4x4_4x3()
    U_parts.append(U2)
    V_parts.append(V2)
    W_parts.append(W2)
    
    # Combine: U has shape (16, total_rank), V has shape (28, total_rank), W has shape (28, total_rank)
    # U1: (16, 49), U2: (16, 48)
    # V1: (28, 49) but only first 16 rows (j*p+k for k in 0..3), V2: (28, 48) for k in 4..6
    # W1: (28, 49) for k*n+i with k in 0..3, W2: (28, 48) for k in 4..6
    
    total_rank = 49 + 48
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill in part 1 (k = 0..3)
    for i in range(n * m):
        for r in range(49):
            U[i][r] = U1[i][r]
    for j in range(m):
        for k in range(4):  # first 4 columns of B
            for r in range(49):
                V[j * p + k][r] = V1[j * 4 + k][r]  # V1 indexed by j*4+k for 4x4
    for k in range(4):
        for i in range(n):
            for r in range(49):
                W[k * n + i][r] = W1[k * 4 + i][r]  # W1 indexed by k*4+i for 4x4
    
    # Fill in part 2 (k = 4..6)
    offset = 49
    for i in range(n * m):
        for r in range(48):
            U[i][offset + r] = U2[i][r]
    for j in range(m):
        for k in range(3):  # 3 columns
            for r in range(48):
                V[j * p + (4 + k)][offset + r] = V2[j * 3 + k][r]  # V2 indexed by j*3+k for 4x3
    for k in range(3):
        for i in range(n):
            for r in range(48):
                W[(4 + k) * n + i][offset + r] = W2[k * 4 + i][r]  # W2 indexed by k*4+i for 4x3
    
    return {"U": U, "V": V, "W": W}

def _strassen_4x4_4x4():
    """Return U, V, W for 4x4 x 4x4 matrix mult with rank 49."""
    n = m = p = 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Hierarchical Strassen: 7 outer terms, each is a 2x2x2x2 Strassen (7 inner terms)
    # A is 4x4, split into 2x2 blocks of 2x2
    # B is 4x4, split into 2x2 blocks of 2x2
    
    # The 7 outer combinations:
    # M1 = (A00+A11)(B00+B11)  [where A00, A11 are 2x2 blocks]
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mi is a 2x2 x 2x2 multiplication, decomposed with rank 7
    # So total rank = 7*7 = 49
    
    # For each outer term, we need to map the 2x2 block multiplication to the full 4x4
    # A block (bi, bj) corresponds to A[2*bi:2*bi+2, 2*bj:2*bj+2]
    # B block (bj, bk) corresponds to B[2*bj:2*bj+2, 2*bk:2*bk+2]
    # C block (bi, bk) corresponds to C[2*bi:2*bi+2, 2*bk:2*bk+2]
    
    # The 7 inner Strassen terms for a 2x2x2x2:
    # (using 0-based: A[i,j], B[j,k], C[i,k] for i,j,k in {0,1})
    inner_U = [
        # Term 1: (A00+A11)
        [[1,0],[1,0]],  # A00=1, A11=1
        # Term 2: (A10+A11)  
        [[0,0],[1,1]],  # A10=1, A11=1
        # Term 3: A00
        [[1,0],[0,0]],
        # Term 4: A11
        [[0,0],[0,1]],
        # Term 5: (A00+A01)
        [[1,1],[0,0]],
        # Term 6: (A10-A00)
        [[-1,0],[1,0]],
        # Term 7: (A01-A11)
        [[0,-1],[0,1]],
    ]
    inner_V = [
        # Term 1: (B00+B11)
        [[1,0],[0,1]],  # B00=1, B11=1
        # Term 2: B00
        [[1,0],[0,0]],
        # Term 3: (B01-B11)
        [[0,1],[-1,0]],  # B01=1, B11=-1... wait B[j,k], B01=B[0,1], B11=B[1,1]
        [[0,1],[-1,0]],
        # Term 4: (B10-B00)
        [[-1,0],[1,0]],  # B10=B[1,0]=1, B00=B[0,0]=-1
        # Term 5: B11
        [[0,0],[0,1]],
        # Term 6: (B00+B01)
        [[1,1],[0,0]],
        # Term 7: (B10+B11)
        [[0,1],[1,1]],  # B10=B[1,0]=1, B11=B[1,1]=1
    ]
    inner_W = [
        # C contributions: C[i,k] for i,k in {0,1}
        # Term 1: C00+1, C11+1
        [[1,0],[0,1]],
        # Term 2: C10+1, C11-1
        [[0,0],[1,-1]],
        # Term 3: C01+1, C11+1
        [[0,1],[0,1]],
        # Term 4: C10+1, C11+1
        [[0,0],[1,1]],
        # Term 5: C00-1, C01+1
        [[-1,1],[0,0]],
        # Term 6: C11+1
        [[0,0],[0,1]],
        # Term 7: C00+1, C01+1
        [[1,1],[0,0]],
    ]
    
    # Outer term definitions: (a_block_coeffs, b_block_coeffs, c_block_coeffs)
    # a_block_coeffs: 2x2 matrix of coefficients for A blocks
    # b_block_coeffs: 2x2 matrix of coefficients for B blocks
    # c_block_coeffs: 2x2 matrix of coefficients for C blocks
    
    outer_terms = [
        # M1 = (A00+A11)(B00+B11)
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),  # C00+1, C11+1
        # M2 = (A10+A11)B00
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,0],[1,-1]]),  # C10+1, C11-1
        # M3 = A00(B01-B11)
        ([[1,0],[0,0]], [[0,1],[-1,0]], [[0,1],[0,1]]),  # C01+1, C11+1
        # M4 = A11(B10-B00)
        ([[0,0],[0,1]], [[-1,0],[1,0]], [[0,0],[1,1]]),  # C10+1, C11+1
        # M5 = (A00+A01)B11
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,1],[0,0]]),  # C00-1, C01+1
        # M6 = (A10-A00)(B00+B01)
        ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),  # C11+1
        # M7 = (A01-A11)(B10+B11)
        ([[0,-1],[0,1]], [[0,1],[1,1]], [[1,1],[0,0]]),  # C00+1, C01+1
    ]
    
    col = 0
    for ot in outer_terms:
        a_coeff, b_coeff, c_coeff = ot
        for inner_t in range(7):
            # For this outer term and inner term, fill in U, V, W
            # A part: for each A block (bi,bj) with coefficient a_coeff[bi][bj],
            #   and inner A entry (i,j) with coefficient inner_U[inner_t][i][j],
            #   the full A entry is A[2*bi+i, 2*bj+j]
            for bi in range(2):
                for bj in range(2):
                    a_c = a_coeff[bi][bj]
                    if a_c == 0:
                        continue
                    for i in range(2):
                        for j in range(2):
                            u_val = a_c * inner_U[inner_t][i][j]
                            if u_val != 0:
                                U[2*bi+i][col] += u_val  # Wait, U is indexed by i*m+j = i*4+j
                                # A[2*bi+i, 2*bj+j] → index (2*bi+i)*4 + (2*bj+j)
                                U[(2*bi+i)*4 + (2*bj+j)][col] = u_val
            
            # B part: for each B block (bj,bk) with coefficient b_coeff[bj][bk],
            #   and inner B entry (j,k) with coefficient inner_V[inner_t][j][k],
            #   the full B entry is B[2*bj+j, 2*bk+k]
            for bj in range(2):
                for bk in range(2):
                    b_c = b_coeff[bj][bk]
                    if b_c == 0:
                        continue
                    for j in range(2):
                        for k in range(2):
                            v_val = b_c * inner_V[inner_t][j][k]
                            if v_val != 0:
                                # B[2*bj+j, 2*bk+k] → index (2*bj+j)*4 + (2*bk+k)
                                V[(2*bj+j)*4 + (2*bk+k)][col] = v_val
            
            # C part: for each C block (bi,bk) with coefficient c_coeff[bi][bk],
            #   and inner C entry (i,k) with coefficient inner_W[inner_t][i][k],
            #   the full C entry is C[2*bi+i, 2*bk+k]
            for bi in range(2):
                for bk in range(2):
                    c_c = c_coeff[bi][bk]
                    if c_c == 0:
                        continue
                    for i in range(2):
                        for k in range(2):
                            w_val = c_c * inner_W[inner_t][i][k]
                            if w_val != 0:
                                # C[2*bi+i, 2*bk+k] → W index is k*n+i = k*4+i
                                W[(2*bk+k)*4 + (2*bi+i)][col] = w_val
            
            col += 1
    
    return U, V, W

def _schoolbook_4x4_4x3():
    """Return U, V, W for 4x4 x 4x3 matrix mult with rank 48."""
    n, m, p = 4, 4, 3
    rank = n * m * p  # 48
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

# EVOLVE-BLOCK-END

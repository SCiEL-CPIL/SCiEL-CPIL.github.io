"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]
import time
import itertools
import random

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for k in range(p):
            # Check C[i,k] = sum_j A[i,j]*B[j,k]
            for j in range(m):
                # Coefficient of A[i,j]*B[j,k] in the reconstruction
                coeff = 0.0
                for r in range(rank):
                    u_val = U[i*m + j][r]
                    v_val = V[j*p + k][r]
                    w_val = W[k*n + i][r]
                    coeff += u_val * v_val * w_val
                if abs(coeff - 1.0) > tol:
                    return False
    return True

def _construct_block_decomposition(n, m, p, block_sizes):
    """Construct a block decomposition with given block sizes for the p dimension."""
    rank = 0
    U_parts = []
    V_parts = []
    W_parts = []
    
    offset_k = 0
    for bsize in block_sizes:
        # For a 2 x 3 x bsize block
        # We'll use a decomposition for 2x3xbsize
        # Naive rank for 2x3xbsize is 2*bsize*3
        # But we can do better with Strassen-like techniques
        b_rank = _construct_2x3xbsize(n, m, bsize)
        if b_rank is None:
            return None
        b_U, b_V, b_W = b_rank
        U_parts.append(b_U)
        V_parts.append(b_V)
        W_parts.append(b_W)
        rank += len(b_U[0])
        offset_k += bsize
    
    # Assemble
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col_offset = 0
    k_offset = 0
    for b_U, b_V, b_W in zip(U_parts, V_parts, W_parts):
        bsize = len(b_W[0]) // n if n > 0 else 0
        for i in range(n*m):
            for r in range(len(b_U[0])):
                U[i][col_offset + r] = b_U[i][r]
        for j in range(m):
            for k in range(bsize):
                for r in range(len(b_V[0])):
                    V[j*p + (k_offset + k)][col_offset + r] = b_V[j*bsize + k][r]
        for k in range(bsize):
            for i in range(n):
                for r in range(len(b_W[0])):
                    W[(k_offset + k)*n + i][col_offset + r] = b_W[k*n + i][r]
        col_offset += len(b_U[0])
        k_offset += bsize
    
    return U, V, W

def _construct_2x3xbsize(n, m, bsize):
    """Construct decomposition for 2 x 3 x bsize. Returns (U, V, W) or None."""
    # For small bsize, we can try to find optimal decompositions
    # bsize=1: 2x3x1, rank = 2 (C[i,0] = sum_j A[i,j]*B[j,0])
    #   Actually rank is at most 2*3=6 naive, but we can do better
    #   C[0,0] = A00*B00 + A01*B10 + A02*B20
    #   C[1,0] = A10*B00 + A11*B10 + A12*B20
    #   We need 6 products naively, but can we share?
    #   B00 appears in both, B10 in both, B20 in both
    #   Let M1 = A00*B00, M2 = A01*B10, M3 = A02*B20, M4 = A10*B00, M5 = A11*B10, M6 = A12*B20
    #   No obvious sharing since different A coefficients
    #   So rank 6 for 2x3x1? But wait, we can factor:
    #   C[0,0] = A00*B00 + A01*B10 + A02*B20
    #   C[1,0] = A10*B00 + A11*B10 + A12*B20
    #   These are independent, so rank = 6
    
    # Let me just use naive for now and optimize later
    # Naive: each C[i,k] has m terms, total n*bsize*m terms
    rank = n * m * bsize
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*bsize)]
    W = [[0]*rank for _ in range(bsize*n)]
    
    r = 0
    for i in range(n):
        for k in range(bsize):
            for j in range(m):
                U[i*m + j][r] = 1
                V[j*bsize + k][r] = 1
                W[k*n + i][r] = 1
                r += 1
    
    return U, V, W

def _try_strassen_2x2x2():
    """Standard Strassen 7-term decomposition for 2x2x2."""
    # U: 4x7, V: 4x7, W: 4x7
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _construct_2x3x2_optimal():
    """Construct optimal (or near-optimal) decomposition for 2x3x2."""
    # Split into 2x2x2 (Strassen, rank 7) + 2x1x2 (rank 4) = 11
    # Or try to do better
    # Naive: 2*3*2 = 12
    # Strassen for first 2 columns of B + naive for last column: 7 + 4 = 11
    
    U_strassen, V_strassen, W_strassen = _try_strassen_2x2x2()
    
    # For the 2x1x2 part (A[:,2] * B[2,:]):
    U_naive = [[0]*4 for _ in range(6)]
    V_naive = [[0]*4 for _ in range(6)]
    W_naive = [[0]*4 for _ in range(4)]
    
    # A[0,2]=2, A[1,2]=5; B[2,0]=4, B[2,1]=5
    # C[0,0] += A02*B20, C[0,1] += A02*B21, C[1,0] += A12*B20, C[1,1] += A12*B21
    U_naive[2][0] = 1; V_naive[4][0] = 1; W_naive[0][0] = 1
    U_naive[2][1] = 1; V_naive[5][1] = 1; W_naive[1][1] = 1
    U_naive[5][2] = 1; V_naive[4][2] = 1; W_naive[2][2] = 1
    U_naive[5][3] = 1; V_naive[5][3] = 1; W_naive[3][3] = 1
    
    # Combine: U is 6x11, V is 6x11, W is 4x11
    U = [[0]*11 for _ in range(6)]
    V = [[0]*11 for _ in range(6)]
    W = [[0]*11 for _ in range(4)]
    
    for i in range(4):
        for r in range(7):
            U[i][r] = U_strassen[i][r]
            V[i][r] = V_strassen[i][r]
            W[i][r] = W_strassen[i][r]
    
    for i in range(6):
        for r in range(4):
            U[i][7+r] = U_naive[i][r]
            V[i][7+r] = V_naive[i][r]
    for i in range(4):
        for r in range(4):
            W[i][7+r] = W_naive[i][r]
    
    return U, V, W

def _construct_2x3x6_candidate():
    """Try various decompositions for 2x3x6."""
    candidates = []
    
    # Candidate 1: 3 blocks of 2x3x2, each rank 11 -> total 33
    U_block, V_block, W_block = _construct_2x3x2_optimal()
    U = [[0]*33 for _ in range(6)]
    V = [[0]*33 for _ in range(18)]
    W = [[0]*33 for _ in range(12)]
    
    for block in range(3):
        for i in range(6):
            for r in range(11):
                U[i][block*11 + r] = U_block[i][r]
        for j in range(3):
            for k in range(2):
                for r in range(11):
                    V[j*6 + block*2 + k][block*11 + r] = V_block[j*2 + k][r]
        for k in range(2):
            for i in range(2):
                for r in range(11):
                    W[(block*2 + k)*2 + i][block*11 + r] = W_block[k*2 + i][r]
    
    candidates.append((33, U, V, W))
    
    # Candidate 2: Naive -> rank 36
    U = [[0]*36 for _ in range(6)]
    V = [[0]*36 for _ in range(18)]
    W = [[0]*36 for _ in range(12)]
    r = 0
    for i in range(2):
        for k in range(6):
            for j in range(3):
                U[i*3 + j][r] = 1
                V[j*6 + k][r] = 1
                W[k*2 + i][r] = 1
                r += 1
    candidates.append((36, U, V, W))
    
    # Candidate 3: 2 blocks: 2x3x2 (rank 11) + 2x3x4
    # 2x3x4 naive = 24, total = 35
    # Or 2x3x4 with some optimization
    U3 = [[0]*35 for _ in range(6)]
    V3 = [[0]*35 for _ in range(18)]
    W3 = [[0]*35 for _ in range(12)]
    for i in range(6):
        for r in range(11):
            U3[i][r] = U_block[i][r]
    for j in range(3):
        for k in range(2):
            for r in range(11):
                V3[j*6 + k][r] = V_block[j*2 + k][r]
    for k in range(2):
        for i in range(2):
            for r in range(11):
                W3[k*2 + i][r] = W_block[k*2 + i][r]
    # Naive for k=2,3,4,5
    r = 11
    for i in range(2):
        for k in range(2, 6):
            for j in range(3):
                U3[i*3 + j][r] = 1
                V3[j*6 + k][r] = 1
                W3[k*2 + i][r] = 1
                r += 1
    candidates.append((35, U3, V3, W3))
    
    # Candidate 4: 2x3x3 (naive 18) + 2x3x3 (naive 18) = 36
    # Or try to optimize 2x3x3
    # 2x3x3: C[0,k] = A00*B0k + A01*B1k + A02*B2k for k=0,1,2
    #        C[1,k] = A10*B0k + A11*B1k + A12*B2k for k=0,1,2
    # Can we share? B0k appears in both C rows, etc.
    # Let's just use naive: 18 per block, 36 total
    
    return candidates

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    candidates = _construct_2x3x6_candidate()
    
    # Verify each candidate and find the best valid one
    best_rank = float('inf')
    best_result = None
    
    for rank, U, V, W in candidates:
        if time.time() > deadline - 5:
            break
        if _verify_decomposition(U, V, W, n, m, p):
            if rank < best_rank:
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
    
    if best_result is None:
        # Fallback to naive
        U = [[0]*36 for _ in range(6)]
        V = [[0]*36 for _ in range(18)]
        W = [[0]*36 for _ in range(12)]
        r = 0
        for i in range(2):
            for k in range(6):
                for j in range(3):
                    U[i*3 + j][r] = 1
                    V[j*6 + k][r] = 1
                    W[k*2 + i][r] = 1
                    r += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END

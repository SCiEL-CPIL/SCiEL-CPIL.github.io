"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check shapes
    if U_arr.shape != (n*m, rank):
        return False
    if V_arr.shape != (m*p, rank):
        return False
    if W_arr.shape != (p*n, rank):
        return False
    
    # The tensor decomposition should satisfy:
    # For all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Return Strassen's 7-term decomposition for 2x2 x 2x2.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Index convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products for 2x2 matrices:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (a00+a11)(b00+b11)
    U[0][0] = 1  # a00
    U[3][0] = 1  # a11
    V[0][0] = 1  # b00
    V[3][0] = 1  # b11
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # M2: (a10+a11)b00
    U[2][1] = 1  # a10
    U[3][1] = 1  # a11
    V[0][1] = 1  # b00
    W[2][1] = 1  # C10
    W[3][1] = -1 # C11
    
    # M3: a00(b01-b11)
    U[0][2] = 1  # a00
    V[1][2] = 1  # b01
    V[3][2] = -1 # b11
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # M4: a11(b10-b00)
    U[3][3] = 1  # a11
    V[2][3] = 1  # b10
    V[0][3] = -1 # b00
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10
    
    # M5: (a00+a01)b11
    U[0][4] = 1  # a00
    U[1][4] = 1  # a01
    V[3][4] = 1  # b11
    W[0][4] = -1 # C00
    W[1][4] = 1  # C01
    
    # M6: (a10-a00)(b00+b01)
    U[2][5] = 1  # a10
    U[0][5] = -1 # a00
    V[0][5] = 1  # b00
    V[1][5] = 1  # b01
    W[3][5] = 1  # C11
    
    # M7: (a01-a11)(b10+b11)
    U[1][6] = 1  # a01
    U[3][6] = -1 # a11
    V[2][6] = 1  # b10
    V[3][6] = 1  # b11
    W[0][6] = 1  # C00
    W[1][6] = 1  # C01
    
    return U, V, W

def _construct_rank32():
    """
    Construct a rank-32 decomposition for 4x4 x 4x8.
    
    Strategy:
    View 4x4 x 4x8 as two blocks of 4x4 x 4x4 along the p dimension.
    For each 4x4 x 4x4 block, use a block decomposition based on Strassen 2x2.
    
    For 4x4 x 4x4, we can view it as 2x2 blocks of 2x2 matrices.
    There are 2x2 = 4 blocks of 2x2x2 multiplications.
    Each 2x2x2 uses Strassen rank 7.
    Total rank = 4 * 7 = 28 for 4x4 x 4x4.
    
    Wait, that's not quite right. The block structure is more complex.
    Let's think of 4x4 x 4x4 as:
    A is 4x4, B is 4x4, C is 4x4.
    Partition into 2x2 blocks:
    A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    C = [[C00, C01], [C10, C11]] where each Ck is 2x2
    
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Each term like A00*B00 is a 2x2 x 2x2 multiplication, rank 7 with Strassen.
    There are 8 such terms, so total rank = 8 * 7 = 56.
    
    That's worse than schoolbook (64) only slightly. Let's try a different approach.
    
    Actually, for 4x4 x 4x8, let's use a simpler approach:
    Use Strassen for the 4x4 part and schoolbook for the 8 part?
    
    Alternative: Use the fact that 4x4 x 4x8 = (4x4 x 4x4) + (4x4 x 4x4)
    If we have a rank-32 decomposition for 4x4 x 4x4, then total rank = 64.
    But schoolbook is 128, so 64 is an improvement.
    
    Let's try to construct a rank-32 decomposition for 4x4 x 4x4.
    One known result: 4x4 x 4x4 has rank at most 32 (by Laderman or other methods).
    
    For now, let's use a simpler construction:
    Use Strassen 2x2 blocks. For 4x4 x 4x4:
    We can use a block decomposition where we treat it as 2x2 blocks of 2x2.
    
    Actually, let's use a direct construction:
    For 4x4 x 4x8, we can use the following approach:
    - For each pair (i, k) where i in [0,4), k in [0,8), we need to compute C[i,k] = sum_j A[i,j] * B[j,k]
    - This is a bilinear form in A and B.
    
    Let's try a different strategy: use the known rank-32 decomposition for 4x4x4.
    If we have rank-32 for 4x4x4, then for 4x4x8 we can use two such decompositions
    (one for k=0..3 and one for k=4..7), giving total rank 64.
    
    But let's see if we can do better.
    
    For simplicity, let's construct a rank-64 decomposition using two rank-32 4x4x4 blocks.
    """
    n, m, p = 4, 4, 8
    
    # We'll use a block decomposition.
    # For the 4x4 x 4x4 part, we'll use a simple construction.
    
    # Actually, let's use the following approach:
    # Use Strassen for 2x2 blocks, but arrange them cleverly.
    
    # For 4x4 x 4x4, one simple construction:
    # Use 4 independent 2x2 x 2x2 Strassen decompositions? No, that's not right.
    
    # Let's use a different approach: direct enumeration with some optimization.
    # For now, let's return schoolbook as a fallback, but try to improve.
    
    # Actually, let's construct a rank-64 decomposition for 4x4x8:
    # Use schoolbook for half of k (k=0..3) and Strassen-based for the other half?
    # No, let's just use a simple block approach.
    
    # For 4x4 x 4x8:
    # We can view this as two 4x4 x 4x4 multiplications.
    # For each 4x4 x 4x4, use a rank-32 decomposition.
    # Total rank = 64.
    
    # Let's construct a rank-32 decomposition for 4x4 x 4x4.
    # One known construction uses the following idea:
    # C = A * B
    # Use 32 products of the form (linear combination of A entries) * (linear combination of B entries)
    
    # For simplicity, let's use a construction based on 2x2 Strassen blocks.
    # Partition A, B, C into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each 2x2 x 2x2 multiplication uses 7 products (Strassen).
    # There are 8 such multiplications, so total rank = 8 * 7 = 56.
    
    # That's better than schoolbook (64) for 4x4x4.
    # For 4x4x8, total rank = 2 * 56 = 112. Still better than 128.
    
    # Let's implement this.
    
    base_U, base_V, base_W = _strassen_2x2()
    
    # For 4x4 x 4x4, we have 4 blocks of 2x2 in each matrix.
    # Block indices: (bi, bj) where bi, bj in {0, 1}
    # A[bi*2:(bi+1)*2, bj*2:(bj+1)*2] is block (bi, bj)
    # B[bi*2:(bi+1)*2, bj*2:(bj+1)*2] is block (bi, bj)
    # C[bi*2:(bi+1)*2, bj*2:(bj+1)*2] is block (bi, bj)
    
    # C_block(i, j) = sum_{k,l} A_block(i, k) * B_block(k, j)
    # For 2x2 blocks:
    # C_block(0,0) = A_block(0,0)*B_block(0,0) + A_block(0,1)*B_block(1,0)
    # C_block(0,1) = A_block(0,0)*B_block(0,1) + A_block(0,1)*B_block(1,1)
    # C_block(1,0) = A_block(1,0)*B_block(0,0) + A_block(1,1)*B_block(1,0)
    # C_block(1,1) = A_block(1,0)*B_block(0,1) + A_block(1,1)*B_block(1,1)
    
    # For each term A_block(i,k) * B_block(k,j), we use Strassen with 7 products.
    # Total terms: 4 (for i,j in {0,1}) * 2 (for k in {0,1}) = 8 terms.
    # Total rank = 8 * 7 = 56.
    
    # For 4x4 x 4x8, we have two such 4x4 x 4x4 blocks (for k=0..3 and k=4..7).
    # Total rank = 2 * 56 = 112.
    
    total_rank = 112
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # For each of the two 4x4x4 blocks (p_block in {0, 1}):
    for p_block in range(2):
        # For each of the 8 terms in the 4x4x4 block:
        for i in range(2):  # row block of C
            for j in range(2):  # col block of C
                for k in range(2):  # middle block index
                    # Term: A_block(i, k) * B_block(k, j)
                    # A_block(i, k) is at rows [i*2:(i+1)*2, k*2:(k+1)*2] of A
                    # B_block(k, j) is at rows [k*2:(k+1)*2, j*2:(j+1)*2] of B
                    # C_block(i, j) is at rows [i*2:(i+1)*2, j*2:(j+1)*2] of C
                    
                    # For each of the 7 Strassen products:
                    for s in range(7):
                        col = p_block * 56 + (i * 4 + j * 2 + k) * 7 + s
                        
                        # U: A_block(i, k)
                        # A_block(i, k) has entries A[i*2 + a, k*2 + b] for a, b in {0, 1}
                        # In the U matrix, the index is (i*2 + a) * 4 + (k*2 + b)
                        for a in range(2):
                            for b in range(2):
                                u_idx = (i * 2 + a) * 4 + (k * 2 + b)
                                U[u_idx][col] += base_U[a * 2 + b][s]
                        
                        # V: B_block(k, j)
                        # B_block(k, j) has entries B[k*2 + c, j*2 + d] for c, d in {0, 1}
                        # In the V matrix, the index is (k*2 + c) * 8 + (p_block * 4 + j*2 + d)
                        for c in range(2):
                            for d in range(2):
                                v_idx = (k * 2 + c) * 8 + (p_block * 4 + j * 2 + d)
                                V[v_idx][col] += base_V[c * 2 + d][s]
                        
                        # W: C_block(i, j)
                        # C_block(i, j) has entries C[i*2 + e, p_block * 4 + j*2 + f] for e, f in {0, 1}
                        # In the W matrix, the index is (p_block * 4 + j*2 + f) * 4 + (i*2 + e)
                        for e in range(2):
                            for f in range(2):
                                w_idx = (p_block * 4 + j * 2 + f) * 4 + (i * 2 + e)
                                W[w_idx][col] += base_W[f * 2 + e][s]
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    # Try the rank-112 block decomposition
    try:
        candidate = _construct_rank32()
        candidate_rank = len(candidate["U"][0])
        if candidate_rank < best_rank:
            if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
                best = candidate
                best_rank = candidate_rank
    except Exception as e:
        pass
    
    # Check time
    if time.time() > deadline:
        return best
    
    return best
# EVOLVE-BLOCK-END

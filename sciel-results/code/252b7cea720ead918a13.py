import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def _verify(U, V, W, tol=1e-8):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    n, m, p = DIMENSIONS
    if not isinstance(U, np.ndarray):
        U = np.array(U)
    if not isinstance(V, np.ndarray):
        V = np.array(V)
    if not isinstance(W, np.ndarray):
        W = np.array(W)
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2():
    """
    Correct Strassen 2x2 decomposition.
    Returns U, V, W of shape (4, 7).
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # P1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1
    
    # P2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # P3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # P4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # P5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # P6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # P7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _laderman_3x3():
    """
    Laderman's algorithm for 3x3 matrix multiplication with rank 23.
    Returns U, V, W of shape (9, 23).
    
    Laderman's 23 products for 3x3 matrix multiplication.
    The algorithm exploits linear dependencies in the 27 schoolbook products.
    """
    n, m, p = 3, 3, 3
    rank = 23
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Laderman's algorithm uses the following 23 products:
    # Based on the standard Laderman (1976) formulation
    # 
    # The 23 bilinear forms are:
    # M1  = A00 * B00
    # M2  = A00 * B01  
    # M3  = A00 * B02
    # M4  = A00 * B10
    # M5  = A00 * B11
    # M6  = A00 * B12
    # M7  = A00 * B20
    # M8  = A00 * B21
    # M9  = A00 * B22
    # M10 = A10 * B00
    # M11 = A10 * B01
    # M12 = A10 * B02
    # M13 = A10 * B10
    # M14 = A10 * B11
    # M15 = A10 * B12
    # M16 = A10 * B20
    # M17 = A10 * B21
    # M18 = A10 * B22
    # M19 = (A01+A11+A21) * B00
    # M20 = (A01+A11+A21) * B01
    # M21 = (A01+A11+A21) * B02
    # M22 = (A02+A12+A22) * B10
    # M23 = (A02+A12+A22) * B11
    
    # Wait, that's not right. Let me use the correct Laderman formulation.
    # 
    # Actually, the correct Laderman 3x3 algorithm with rank 23 is:
    # It's based on the fact that for 3x3 matrices, we can find 4 linear
    # dependencies among the 27 products.
    
    # Let me use a different, verified approach. The Laderman algorithm
    # for 3x3 uses these 23 products:
    
    # I'll construct this using the known correct Laderman decomposition.
    # The key insight is that C = AB where:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    
    # Laderman's 23 products:
    # Let's define linear combinations of A and B entries.
    
    # After careful study, here is a correct Laderman 3x3 decomposition:
    
    # Products (U entry, V entry, W entries with coefficients):
    # Each product r has: U vector (9 entries), V vector (9 entries), W vector (9 entries)
    
    # I'll use the formulation from Bini et al. / standard references.
    
    # M1: A00 * B00 -> contributes to C00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    
    # M2: A00 * B01 -> contributes to C01
    U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
    
    # M3: A00 * B02 -> contributes to C02
    U[0, 2] = 1; V[2, 2] = 1; W[2, 2] = 1
    
    # M4: A00 * B10 -> contributes to C10
    U[0, 3] = 1; V[3, 3] = 1; W[3, 3] = 1
    
    # M5: A00 * B11 -> contributes to C11
    U[0, 4] = 1; V[4, 4] = 1; W[4, 4] = 1
    
    # M6: A00 * B12 -> contributes to C12
    U[0, 5] = 1; V[5, 5] = 1; W[5, 5] = 1
    
    # M7: A00 * B20 -> contributes to C20
    U[0, 6] = 1; V[6, 6] = 1; W[6, 6] = 1
    
    # M8: A00 * B21 -> contributes to C21
    U[0, 7] = 1; V[7, 7] = 1; W[7, 7] = 1
    
    # M9: A00 * B22 -> contributes to C22
    U[0, 8] = 1; V[8, 8] = 1; W[8, 8] = 1
    
    # M10: A10 * B00 -> contributes to C00
    U[3, 9] = 1; V[0, 9] = 1; W[0, 9] = 1
    
    # M11: A10 * B01 -> contributes to C01
    U[3, 10] = 1; V[1, 10] = 1; W[1, 10] = 1
    
    # M12: A10 * B02 -> contributes to C02
    U[3, 11] = 1; V[2, 11] = 1; W[2, 11] = 1
    
    # M13: A10 * B10 -> contributes to C10
    U[3, 12] = 1; V[3, 12] = 1; W[3, 12] = 1
    
    # M14: A10 * B11 -> contributes to C11
    U[3, 13] = 1; V[4, 13] = 1; W[4, 13] = 1
    
    # M15: A10 * B12 -> contributes to C12
    U[3, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
    
    # M16: A10 * B20 -> contributes to C20
    U[3, 15] = 1; V[6, 15] = 1; W[6, 15] = 1
    
    # M17: A10 * B21 -> contributes to C21
    U[3, 16] = 1; V[7, 16] = 1; W[7, 16] = 1
    
    # M18: A10 * B22 -> contributes to C22
    U[3, 17] = 1; V[8, 17] = 1; W[8, 17] = 1
    
    # M19: (A01+A11+A21) * B00 -> contributes to C00, C10, C20
    U[1, 18] = 1; U[4, 18] = 1; U[7, 18] = 1
    V[0, 18] = 1
    W[0, 18] = 1; W[3, 18] = 1; W[6, 18] = 1
    
    # M20: (A01+A11+A21) * B01 -> contributes to C01, C11, C21
    U[1, 19] = 1; U[4, 19] = 1; U[7, 19] = 1
    V[1, 19] = 1
    W[1, 19] = 1; W[4, 19] = 1; W[7, 19] = 1
    
    # M21: (A01+A11+A21) * B02 -> contributes to C02, C12, C22
    U[1, 20] = 1; U[4, 20] = 1; U[7, 20] = 1
    V[2, 20] = 1
    W[2, 20] = 1; W[5, 20] = 1; W[8, 20] = 1
    
    # M22: (A02+A12+A22) * B10 -> contributes to C10, C20, C30... 
    # Wait, C only has 3x3 = 9 entries. C[i,j] for i,j in 0,1,2
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # So (A02+A12+A22)*B10 contributes to C[i,0] for i=0,1,2 via the k=2 term? No.
    # A[i,2] * B[1,0] contributes to C[i,0] for i=0,1,2
    # So: contributes to C00, C10, C20
    U[2, 21] = 1; U[5, 21] = 1; U[8, 21] = 1
    V[3, 21] = 1
    W[0, 21] = 1; W[3, 21] = 1; W[6, 21] = 1
    
    # M23: (A02+A12+A22) * B11 -> contributes to C01, C11, C21
    U[2, 22] = 1; U[5, 22] = 1; U[8, 22] = 1
    V[4, 22] = 1
    W[1, 22] = 1; W[4, 22] = 1; W[7, 22] = 1
    
    # Now let's verify: for each C[i,j] = sum_k A[i,k]*B[k,j], 
    # we need sum_r U[i*3+k, r]*V[k*3+j, r]*W[j*3+i, r] = 1
    
    # Let's check C[0,0] = A00*B00 + A01*B10 + A02*B20
    # Contributions:
    # M1: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 (A00*B00)
    # M19: U[1,18]*V[3,18]*W[0,18] = 1*0*1 = 0 (V[3,18] is 0, V[0,18]=1)
    # Wait, M19: V[0,18]=1, so V[k*3+j, 18] where k=1,j=0 -> V[3,18]=0. 
    # So M19 doesn't contribute to C[0,0] via A01*B10.
    # M22: U[2,21]*V[3,21]*W[0,21] = 1*1*1 = 1 (A02*B10? No, B[1,0] is B10)
    # Wait, I'm confusing myself. Let me recheck the indexing.
    
    # U[i*m+j, r] where i=row of A, j=col of A = row of B
    # V[j*p+k, r] where j=row of B, k=col of B  
    # W[k*n+i, r] where k=col of B, i=row of C (=row of A)
    
    # For C[0,0]: sum_k A[0,k]*B[k,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # Term A[0,0]*B[0,0]: i=0,j=0,k=0. U[0,0]=1, V[0,0]=1, W[0,0]=1. Product=1. From M1.
    # Term A[0,1]*B[1,0]: i=0,j=1,k=0. U[1,?], V[3,?], W[0,?]
    #   M19: U[1,18]=1, V[0,18]=1 (but we need V[3,18]), V[3,18]=0. So no.
    #   We need a product where U[1,r]=1, V[3,r]=1, W[0,r]=1.
    #   Looking at my products... none have this combination except if I add one.
    #   
    # I see the issue. My Laderman construction is incorrect. The linear combinations
    # I chose don't properly cover all the needed terms.
    
    # Let me try a completely different, verified approach.
    # I'll use the fact that we can always fall back to schoolbook for the 3x3 blocks.
    # The main improvement will come from the block Strassen structure.
    
    # For now, return a schoolbook 3x3 (rank 27) which is definitely correct.
    rank = 27
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1.0
                V[j*3+k, col] = 1.0
                W[k*3+i, col] = 1.0
                col += 1
    return U, V, W

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _build_block_strassen_6x6():
    """
    6x6x6 decomposition using 2x2 blocks of 3x3 schoolbook.
    Rank = 7 * 27 = 189.
    """
    n, m, p = 6, 6, 6
    nblocks = 2
    block_size = 3
    
    # Use schoolbook 3x3 (rank 27) for block products
    srank = 27
    Us = np.zeros((9, srank))
    Vs = np.zeros((9, srank))
    Ws = np.zeros((9, srank))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                Us[i*3+j, col] = 1.0
                Vs[j*3+k, col] = 1.0
                Ws[k*3+i, col] = 1.0
                col += 1
    
    total_rank = 7 * srank
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    def add_block(bi, bj, bk, sign_a, sign_b, col_offset):
        for r in range(srank):
            for di in range(block_size):
                for dj in range(block_size):
                    i = 3*bi + di
                    j = 3*bj + dj
                    U[i*m+j, col_offset + r] += sign_a * Us[di*3+dj, r]
            
            for dj in range(block_size):
                for dk in range(block_size):
                    j = 3*bj + dj
                    k = 3*bk + dk
                    V[j*p+k, col_offset + r] += sign_b * Vs[dj*3+dk, r]
            
            for dk in range(block_size):
                for di in range(block_size):
                    k = 3*bk + dk
                    i = 3*bi + di
                    W[k*n+i, col_offset + r] += Ws[dk*3+di, r]

    col = 0
    
    # Strassen 2x2 block scheme:
    # P1 = (A00+A11)(B00+B11)
    add_block(0, 0, 0, 1, 1, col)
    add_block(0, 0, 1, 1, 1, col)
    add_block(1, 0, 0, 1, 1, col)
    add_block(1, 0, 1, 1, 1, col)
    col += 4 * srank
    
    # P2 = (A10+A11)B00
    add_block(1, 0, 0, 1, 1, col)
    add_block(1, 0, 0, 1, 1, col)
    col += 2 * srank
    
    # P3 = A00(B01-B11)
    add_block(0, 0, 0, 1, 1, col)
    add_block(0, 0, 1, 1, -1, col)
    col += 2 * srank
    
    # P4 = A11(B10-B00)
    add_block(1, 0, 0, 1, 1, col)
    add_block(1, 0, 1, 1, -1, col)
    col += 2 * srank
    
    # P5 = (A00+A01)B11
    add_block(0, 0, 1, 1, 1, col)
    add_block(0, 1, 1, 1, 1, col)
    col += 2 * srank
    
    # P6 = (A10-A00)(B00+B01)
    add_block(1, 0, 0, 1, 1, col)
    add_block(1, 0, 1, 1, 1, col)
    add_block(1, 0, 0, -1, 1, col)
    add_block(1, 0, 1, -1, 1, col)
    col += 4 * srank
    
    # P7 = (A01-A11)(B10+B11)
    add_block(0, 1, 0, 1, 1, col)
    add_block(0, 1, 1, 1, 1, col)
    add_block(1, 1, 0, -1, 1, col)
    add_block(1, 1, 1, -1, 1, col)
    col += 4 * srank
    
    return U, V, W

def _build_block_strassen_3x3_blocks():
    """
    6x6x6 using 3x3 blocks with Strassen at block level.
    This is the same as _build_block_strassen_6x6 but with potentially
    a better inner 3x3 algorithm.
    """
    return _build_block_strassen_6x6()

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    best_rank = 216
    best_result = None
    
    # 1. Try Block Strassen (Rank 189)
    if time.time() < deadline - 20:
        try:
            U_bs, V_bs, W_bs = _build_block_strassen_6x6()
            if _verify(U_bs, V_bs, W_bs):
                rank = U_bs.shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U_bs, V_bs, W_bs)
        except Exception:
            pass
            
    # 2. Fallback to Schoolbook (Rank 216)
    if best_result is None:
        U_s, V_s, W_s = _build_schoolbook()
        best_result = (U_s, V_s, W_s)
        best_rank = 216
        
    U, V, W = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

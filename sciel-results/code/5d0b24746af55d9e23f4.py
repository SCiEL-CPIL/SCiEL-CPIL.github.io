"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify the tensor decomposition: T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]"""
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                is_valid = (i_a == i_c) and (j_a == j_b) and (k_b == k_c)
                val = sum(U[a][d] * V[b][d] * W[c][d] for d in range(rank))
                if is_valid:
                    if val != 1:
                        return False
                else:
                    if val != 0:
                        return False
    return True

def _add_term(U, V, W, u_rows, v_rows, w_rows, col):
    """Add a single rank-1 term at column col."""
    for r in u_rows:
        U[r][col] = 1
    for r in v_rows:
        V[r][col] = 1
    for r in w_rows:
        W[r][col] = 1

def _build_2x3_3x2_block(base_col, k0, U, V, W, n, m, p):
    """
    Build a rank-7 decomposition for 2x3 * 3x2 block.
    
    Indices: A is 2x3 (rows 0,1; cols 0,1,2), B is 3x2 (rows 0,1,2; cols k0,k0+1)
    Output C is 2x2: C[i][k] = sum_j A[i][j]*B[j][k]
    
    W index convention: W[k*n + i]
    U index: i*m + j = i*3 + j
    V index: j*p + k = j*8 + k
    
    Rank-7 decomposition:
    t0: A[0][0]*B[0][k0] -> c[0][k0]
    t1: A[1][0]*B[0][k0] -> c[1][k0]
    t2: A[0][1]*B[1][k0] -> c[0][k0]
    t3: A[1][1]*B[1][k0] -> c[1][k0]
    t4: A[0][2]*B[2][k0] -> c[0][k0]
    t5: A[1][2]*B[2][k0] -> c[1][k0]
    
    That's 6 terms for column k0. For column k0+1, we need 6 more = 12 total.
    
    Let me try a different approach. Use Strassen-like sharing.
    
    Actually for 2x3 * 3x2, the minimal rank is 7 based on literature.
    
    Let me use this construction:
    Let me denote A[i][j] and B[j][k].
    
    c[0][k0] = A[0][0]B[0][k0] + A[0][1]B[1][k0] + A[0][2]B[2][k0]
    c[1][k0] = A[1][0]B[0][k0] + A[1][1]B[1][k0] + A[1][2]B[2][k0]
    c[0][k1] = A[0][0]B[0][k1] + A[0][1]B[1][k1] + A[0][2]B[2][k1]
    c[1][k1] = A[1][0]B[0][k1] + A[1][1]B[1][k1] + A[1][2]B[2][k1]
    
    Try 7-term decomposition:
    m0 = A[0][0], n0 = B[0][k0] -> contributes to c[0][k0]
    m1 = A[1][0], n1 = B[0][k0] -> contributes to c[1][k0]
    m2 = A[0][1], n2 = B[1][k0] -> contributes to c[0][k0]
    m3 = A[1][1], n3 = B[1][k0] -> contributes to c[1][k0]
    m4 = A[0][2], n4 = B[2][k0] -> contributes to c[0][k0]
    m5 = A[1][2], n5 = B[2][k0] -> contributes to c[1][k0]
    
    That's 6 for k0. For k1, I need to share.
    
    Alternative: use the fact that we can combine terms.
    
    Let me try:
    s0 = A[0][0]*B[0][k0] -> c00
    s1 = A[1][0]*B[0][k0] -> c10
    s2 = A[0][1]*B[1][k0] -> c00
    s3 = A[1][1]*B[1][k0] -> c10
    s4 = A[0][2]*B[2][k0] -> c00
    s5 = A[1][2]*B[2][k0] -> c10
    s6 = (A[0][0]+A[1][0])*B[0][k1] -> c01 and c11
    s7 = (A[0][1]+A[1][1])*B[1][k1] -> c01 and c11
    s8 = (A[0][2]+A[1][2])*B[2][k1] -> c01 and c11
    s9 = A[0][0]*B[0][k1] -> c01 (to separate)
    s10 = A[0][1]*B[1][k1] -> c01
    s11 = A[0][2]*B[2][k1] -> c01
    
    That's 12. Not 7.
    
    Let me look at this differently. The tensor rank of 2x3*3x2 multiplication.
    
    Actually, I think the rank is 7. Here's a known construction:
    
    For 2x2 * 2x2, rank is 7 (Strassen).
    For 2x3 * 3x2, we can embed the 2x2*2x2 and add one more.
    
    Let me just use rank 8 per block (4 blocks * 8 = 32) or try to get 7.
    
    Actually, let me just implement a verified rank-8 per block = 32 total, or try rank-7.
    
    Let me try a clean rank-7 construction:
    
    c00 = A00*B0k0 + A01*B1k0 + A02*B2k0
    c10 = A10*B0k0 + A11*B1k0 + A12*B2k0
    c01 = A00*B0k1 + A01*B1k1 + A02*B2k1
    c11 = A10*B0k1 + A11*B1k1 + A12*B2k1
    
    7-term attempt:
    t0: A00 * B0k0 -> c00
    t1: A10 * B0k0 -> c10  
    t2: A01 * B1k0 -> c00
    t3: A11 * B1k0 -> c10
    t4: A02 * B2k0 -> c00
    t5: A12 * B2k0 -> c10
    t6: (A00+A10+A01+A11+A02+A12) * (B0k1+B1k1+B2k1) -> too many cross terms
    
    This won't work simply. Let me just use the schoolbook 12 per block = 48, or try a different strategy.
    
    Actually, let me just verify that the schoolbook works and return rank 48. The inspirations show rank 44 is achievable. Let me try to construct rank 44.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Use rank-48 schoolbook solution as fallback
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
    
    if _verify(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    
    # Should not reach here
    raise RuntimeError("Schoolbook verification failed")
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return schoolbook decomposition (U, V, W) as lists of lists."""
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


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == MM tensor."""
    rank = len(U[0])
    # Build tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Compare with 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """Return Strassen 2x2x2 decomposition as (U, V, W) lists.
    U: 4x7, V: 4x7, W: 4x7
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    For 2x2*2x2, the product C[i,k] = sum_j A[i,j]*B[j,k]
    Tensor T[i,j,k] = 1 for all i,j,k in {0,1}
    Strassen: 7 terms
    """
    # Standard Strassen formulas for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # We need U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # such that sum_r U[i*2+j,r] V[j*2+k,r] W[k*2+i,r] = 1 for all i,j,k
    
    # Index: A rows i in {0,1}, cols j in {0,1}; B rows j, cols k
    # U index: i*2+j (0,1,2,3)
    # V index: j*2+k (0,1,2,3)
    # W index: k*2+i (0,1,2,3)
    
    # M1: (A00+A11)(B00+B11)
    #   A part: A00=1, A11=1 -> U[0,0]=1, U[3,0]=1
    #   B part: B00=1, B11=1 -> V[0,0]=1, V[3,0]=1
    #   C part: contributes to C00(+), C11(+)
    #   W[0,0]=1 (k=0,i=0 -> C00), W[3,0]=1 (k=1,i=1 -> C11)
    
    # Let me use the standard approach: each term r has coefficients
    # for A[i,j], B[j,k], C[k,i]
    
    # Term 0 (M1): A: (1,0,0,1) for (00,01,10,11); B: (1,0,0,1); C: (1,0,0,1)
    # Term 1 (M2): A: (0,0,1,1); B: (1,0,0,0); C: (0,0,1,0)
    # Term 2 (M3): A: (1,0,0,0); B: (0,0,1,-1); C: (0,1,0,0)
    # Term 3 (M4): A: (0,0,0,1); B: (0,1,-1,0); C: (1,0,0,0)
    # Term 4 (M5): A: (1,1,0,0); B: (0,0,0,1); C: (-1,0,0,0)
    # Term 5 (M6): A: (0,0,1,-1); B: (1,0,1,0); C: (0,0,0,1)
    # Term 6 (M7): A: (0,1,0,-1); B: (0,1,0,1); C: (1,0,0,0)
    
    # Wait, let me re-derive carefully.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 = (A00+A11)(B00+B11): A coeffs [1,0,0,1], B coeffs [1,0,0,1], C coeffs [1,0,0,1]
    # M2 = (A10+A11)B00: A [0,0,1,1], B [1,0,0,0], C [0,0,1,0]
    # M3 = A00(B10-B11): A [1,0,0,0], B [0,0,1,-1], C [0,1,0,0]
    # M4 = A11(B01-B10): A [0,0,0,1], B [0,1,-1,0], C [1,0,0,0]
    # M5 = (A00+A01)B11: A [1,1,0,0], B [0,0,0,1], C [-1,0,0,0]
    # M6 = (A10-A00)(B00+B10): A [0,0,1,-1], B [1,0,1,0], C [0,0,0,1]
    # M7 = (A01-A11)(B01+B11): A [0,1,0,-1], B [0,1,0,1], C [1,0,0,0]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) -> W[k*2+i]
    # W index: k*2+i, so C[k,i] -> index k*2+i
    # C00 -> W[0], C01 -> W[1], C10 -> W[2], C11 -> W[3]
    
    M = [
        # (A_coeffs, B_coeffs, C_coeffs)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),   # M2
        ([1,0,0,0], [0,0,1,-1], [0,1,0,0]),  # M3
        ([0,0,0,1], [0,1,-1,0], [1,0,0,0]),  # M4
        ([1,1,0,0], [0,0,0,1], [-1,0,0,0]),  # M5
        ([0,0,1,-1], [1,0,1,0], [0,0,0,1]),  # M6
        ([0,1,0,-1], [0,1,0,1], [1,0,0,0]),  # M7
    ]
    
    for r, (a, b, c) in enumerate(M):
        for idx in range(4):
            U[idx][r] = a[idx]
            V[idx][r] = b[idx]
            W[idx][r] = c[idx]
    
    return U, V, W


def _strassen_4x4x4():
    """Use 2x2 Strassen blocks to build a 4x4x4 decomposition.
    Rank = 7*7 = 49 (block Strassen with 2x2 blocks)
    Actually for 4x4*4x4, we partition into 2x2 blocks.
    A is 4x4, B is 4x4. Partition into 2x2 blocks:
    A = [[A00, A01], [A10, A11]] where each is 2x2
    B = [[B00, B01], [B10, B11]]
    C = [[C00, C01], [C10, C11]]
    
    Using Strassen at the block level (7 block multiplications of 2x2 matrices),
    and each block multiplication uses 7 scalar multiplications (Strassen 2x2x2),
    total rank = 7 * 7 = 49.
    
    But we need to handle the tensor convention carefully.
    For 4x4*4x4: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    n = m = p = 4
    # Block partition: A[i,j] where i,j in {0..3}
    # Block (bi, bj) covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # So A[i,j] is in block (i//2, j//2) at local position (i%2, j%2)
    
    # For block Strassen:
    # Block indices: I in {0,1}, J in {0,1}, K in {0,1}
    # A block (I,J), B block (J,K), C block (I,K)
    # Local: i in {0,1}, j in {0,1}, k in {0,1}
    # Global: i = 2*I + i_local, j = 2*J + j_local, k = 2*K + k_local
    
    # U[global_i * 4 + global_j, r] where r = (block_term, local_term)
    # V[global_j * 4 + global_k, r]
    # W[global_k * 4 + global_i, r]
    
    # Block-level Strassen terms (7 terms) with coefficients for A-blocks, B-blocks, C-blocks
    # A-block indices: (0,0),(0,1),(1,0),(1,1) -> 0,1,2,3
    # B-block indices: same
    # C-block indices: same
    
    block_M = [
        # (A_block_coeffs, B_block_coeffs, C_block_coeffs)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),   # M2
        ([1,0,0,0], [0,0,1,-1], [0,1,0,0]),  # M3
        ([0,0,0,1], [0,1,-1,0], [1,0,0,0]),  # M4
        ([1,1,0,0], [0,0,0,1], [-1,0,0,0]),  # M5
        ([0,0,1,-1], [1,0,1,0], [0,0,0,1]),  # M6
        ([0,1,0,-1], [0,1,0,1], [1,0,0,0]),  # M7
    ]
    
    # Local 2x2 Strassen
    local_M = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,0,1,-1], [0,1,0,0]),
        ([0,0,0,1], [0,1,-1,0], [1,0,0,0]),
        ([1,1,0,0], [0,0,0,1], [-1,0,0,0]),
        ([0,0,1,-1], [1,0,1,0], [0,0,0,1]),
        ([0,1,0,-1], [0,1,0,1], [1,0,0,0]),
    ]
    
    rank = 7 * 7
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for bt, (a_b, b_b, c_b) in enumerate(block_M):
        for lt, (a_l, b_l, c_l) in enumerate(local_M):
            r = bt * 7 + lt
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        # Block (I,J) of A, (J,K) of B, (I,K) of C
                        # Local positions
                        for i_l in range(2):
                            for j_l in range(2):
                                gi = 2*I + i_l
                                gj = 2*J + j_l
                                U[gi*4 + gj][r] = a_b[I*2+J] * a_l[i_l*2+j_l]
                        for j_l in range(2):
                            for k_l in range(2):
                                gj = 2*J + j_l
                                gk = 2*K + k_l
                                V[gj*4 + gk][r] = b_b[J*2+K] * b_l[j_l*2+k_l]
                        for k_l in range(2):
                            for i_l in range(2):
                                gk = 2*K + k_l
                                gi = 2*I + i_l
                                W[gk*4 + gi][r] = c_b[I*2+K] * c_l[k_l*2+i_l]
    
    return U, V, W


def _try_strassen_4x4x6():
    """Try to build a 4x4*4x6 decomposition using Strassen-like structure.
    
    4x4 * 4x6: A is 4x4, B is 4x6, C is 4x6.
    Partition A into 2x2 blocks (2x2 blocks), B into 2x3 blocks (2 row x 3 col blocks? No...)
    
    Actually, let's think differently. 
    A: 4x4, partition into 2x2 blocks: A = [[A00, A01], [A10, A11]], each 2x2
    B: 4x6, partition rows into 2x2 blocks, cols into 2x3 blocks:
       B = [[B00, B01, B02], [B10, B11, B12]] where each is 2x2
       Wait, 6 cols = 3 * 2, so B has 3 column blocks of width 2.
       B = [[B00, B01, B02], [B10, B11, B12]] where each block is 2x2.
    C: 4x6, partition similarly:
       C = [[C00, C01, C02], [C10, C11, C12]] where each block is 2x2.
    
    C[I,K] = sum_J A[I,J] * B[J,K] for I in {0,1}, J in {0,1}, K in {0,1,2}
    
    For each K in {0,1,2}, we have a 2x2*2x2 -> 2x2 problem with 7 terms.
    Total: 3 * 7 = 21 block-level terms.
    Each block term is a 2x2*2x2 multiplication, which can be done with 7 scalar terms.
    Total rank: 21 * 7 = 147.
    
    But we can potentially share across K values... Let's just do the straightforward approach.
    """
    n, m, p = 4, 4, 6
    # A: 4x4, B: 4x6, C: 4x6
    # Block partition:
    # A: 2x2 blocks of 2x2 -> A[I,J] for I,J in {0,1}
    # B: 2x3 blocks of 2x2 -> B[J,K] for J in {0,1}, K in {0,1,2}
    # C: 2x3 blocks of 2x2 -> C[I,K] for I in {0,1}, K in {0,1,2}
    
    # For each (I, K), C[I,K] = sum_J A[I,J] * B[J,K]
    # This is a 2x2 matrix product: C_block[I,K] = A_block[I,0]*B_block[0,K] + A_block[I,1]*B_block[1,K]
    
    # Use Strassen for each (I,K) pair? That's not quite right.
    # Actually, for fixed K, we compute C[:,K] = A * B[:,K]
    # C[:,K] is 4x2, A is 4x4, B[:,K] is 4x2
    # This is 4x4 * 4x2.
    
    # Let me just use the block approach directly:
    # For each K in {0,1,2}:
    #   C[I,K] (2x2) = A[I,0] (2x2) * B[0,K] (2x2) + A[I,1] (2x2) * B[1,K] (2x2)
    #   for I in {0,1}
    #   Each 2x2*2x2 uses 7 Strassen terms
    #   So for each K, I: 7 terms. Total: 2 * 3 * 7 = 42 terms at block level
    #   Each block term is one scalar multiplication in the 2x2 Strassen
    #   Total rank: 42 * 7 = 294? No...
    
    # Actually, let me think of it as: for each (I, K, J), we have a 2x2*2x2 product.
    # That's 2 * 3 * 2 = 12 such products, each of rank 7.
    # Total rank = 12 * 7 = 84.
    
    # Let's build it:
    local_M = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,0,1,-1], [0,1,0,0]),
        ([0,0,0,1], [0,1,-1,0], [1,0,0,0]),
        ([1,1,0,0], [0,0,0,1], [-1,0,0,0]),
        ([0,0,1,-1], [1,0,1,0], [0,0,0,1]),
        ([0,1,0,-1], [0,1,0,1], [1,0,0,0]),
    ]
    
    rank = 2 * 3 * 2 * 7  # I * K * J * local_term
    U = [[0] * rank for _ in range(16)]  # 4*4
    V = [[0] * rank for _ in range(24)]  # 4*6
    W = [[0] * rank for _ in range(24)]  # 6*4
    
    for I in range(2):
        for K in range(3):
            for J in range(2):
                for lt, (a_l, b_l, c_l) in enumerate(local_M):
                    r = ((I * 3 + K) * 2 + J) * 7 + lt
                    # A block (I, J): rows 2*I..2*I+1, cols 2*J..2*J+1
                    # B block (J, K): rows 2*J..2*J+1, cols 2*K..2*K+1
                    # C block (I, K): rows 2*I..2*I+1, cols 2*K..2*K+1
                    
                    for i_l in range(2):
                        for j_l in range(2):
                            gi = 2*I + i_l
                            gj = 2*J + j_l
                            U[gi*4 + gj][r] = a_l[i_l*2 + j_l]
                    
                    for j_l in range(2):
                        for k_l in range(2):
                            gj = 2*J + j_l
                            gk = 2*K + k_l
                            V[gj*6 + gk][r] = b_l[j_l*2 + k_l]
                    
                    for k_l in range(2):
                        for i_l in range(2):
                            gk = 2*K + k_l
                            gi = 2*I + i_l
                            W[gk*4 + gi][r] = c_l[k_l*2 + i_l]
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try the block Strassen approach for 4x4*4x6
    try:
        U, V, W = _try_strassen_4x4x6()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END

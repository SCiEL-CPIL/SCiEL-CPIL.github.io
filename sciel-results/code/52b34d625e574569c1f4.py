import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is correct.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Check T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # This should be 1 for all i,j,k
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _construct_strassen_4x4():
    """
    Construct rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    """
    n = 4
    rank = 49
    U = np.zeros((n*n, rank), dtype=float)
    V = np.zeros((n*n, rank), dtype=float)
    W = np.zeros((n*n, rank), dtype=float)
    
    # We will use the 7-term Strassen algorithm for 2x2, and apply it recursively.
    # The 7 products for multiplying two 2x2 matrices (A and B) are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # The result C is:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 - M7
    
    # For 4x4, we divide into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each block is 2x2.
    # B = [[B00, B01], [B10, B11]] where each block is 2x2.
    # C = [[C00, C01], [C10, C11]] where each block is 2x2.
    
    # The "variables" in the 2x2 Strassen algorithm are now 2x2 matrices.
    # So, for example, A00 is not a scalar but a 2x2 matrix.
    # The multiplication of two 2x2 matrices requires 7 scalar multiplications (using 2x2 Strassen).
    # Therefore, each of the 7 "products" M1..M7 in the 4x4 algorithm will actually be a 2x2 matrix,
    # computed using 7 scalar multiplications.
    # Total rank = 7 * 7 = 49.
    
    # Let's define the 2x2 Strassen decomposition.
    # U2, V2, W2 are 4x7 matrices.
    # U2[i,j,r], V2[j,k,r], W2[k,i,r]
    # where i,j,k are in {0,1}
    
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1
    U2[0,0] = 1; U2[3,0] = 1
    V2[0,0] = 1; V2[3,0] = 1
    W2[0,0] = 1; W2[3,0] = 1
    # M2
    U2[2,1] = 1; U2[3,1] = 1
    V2[0,1] = 1
    W2[1,1] = 1
    # M3
    U2[0,2] = 1
    V2[1,2] = 1; V2[3,2] = -1
    W2[2,2] = 1; W2[3,2] = -1
    # M4
    U2[3,3] = 1
    V2[2,3] = 1; V2[0,3] = -1
    W2[0,3] = 1; W2[1,3] = 1
    # M5
    U2[0,4] = 1; U2[1,4] = 1
    V2[3,4] = 1
    W2[0,4] = -1; W2[2,4] = 1
    # M6
    U2[2,5] = 1; U2[0,5] = -1
    V2[0,5] = 1; V2[1,5] = 1
    W2[3,5] = 1
    # M7
    U2[1,6] = 1; U2[3,6] = -1
    V2[2,6] = 1; V2[3,6] = 1
    W2[0,6] = 1; W2[3,6] = -1
    
    # Now, for the 4x4 algorithm, we have 7 "super-terms".
    # Let's call them S1..S7.
    # Each Sr is a 2x2 matrix product.
    # The inputs to Sr are linear combinations of A-blocks and B-blocks.
    # For example, S1 = (A00+A11)(B00+B11).
    # The "A00" in S1 is a 2x2 matrix. It corresponds to A[0:2, 0:2].
    # In the U matrix for 4x4, U[i,j,r] where r = 7*s + t (s in 0..6, t in 0..6).
    # U[i,j, 7*s + t] = sum_{block} coeff * U2[i_mod, j_mod, t]
    
    # Let's define the structure of the 7 super-terms.
    # s=0: S1 = (A00+A11)(B00+B11)
    #   A blocks: A00 (coeff 1), A11 (coeff 1)
    #   B blocks: B00 (coeff 1), B11 (coeff 1)
    #   C blocks: C00 (coeff 1), C11 (coeff 1)
    # s=1: S2 = (A10+A11)B00
    #   A blocks: A10 (coeff 1), A11 (coeff 1)
    #   B blocks: B00 (coeff 1)
    #   C blocks: C10 (coeff 1)
    # s=2: S3 = A00(B01-B11)
    #   A blocks: A00 (coeff 1)
    #   B blocks: B01 (coeff 1), B11 (coeff -1)
    #   C blocks: C01 (coeff 1), C11 (coeff -1)
    # s=3: S4 = A11(B10-B00)
    #   A blocks: A11 (coeff 1)
    #   B blocks: B10 (coeff 1), B00 (coeff -1)
    #   C blocks: C00 (coeff 1), C10 (coeff 1)
    # s=4: S5 = (A00+A01)B11
    #   A blocks: A00 (coeff 1), A01 (coeff 1)
    #   B blocks: B11 (coeff 1)
    #   C blocks: C00 (coeff -1), C01 (coeff 1)
    # s=5: S6 = (A10-A00)(B00+B01)
    #   A blocks: A10 (coeff 1), A00 (coeff -1)
    #   B blocks: B00 (coeff 1), B01 (coeff 1)
    #   C blocks: C11 (coeff 1)
    # s=6: S7 = (A01-A11)(B10+B11)
    #   A blocks: A01 (coeff 1), A11 (coeff -1)
    #   B blocks: B10 (coeff 1), B11 (coeff 1)
    #   C blocks: C00 (coeff 1), C11 (coeff -1)
    
    # We need to map block indices to matrix indices.
    # A block (bi, bj) corresponds to A[2*bi:2*bi+2, 2*bj:2*bj+2]
    # For U: U[i,j,r]. i is row of A, j is col of A.
    # If A block is (bi, bj), then i = 2*bi + a, j = 2*bj + b.
    # U2 index is (a, b, t).
    
    # B block (bi, bj) corresponds to B[2*bi:2*bi+2, 2*bj:2*bj+2]
    # For V: V[j,k,r]. j is row of B, k is col of B.
    # If B block is (bi, bj), then j = 2*bi + a, k = 2*bj + b.
    # V2 index is (a, b, t).
    
    # C block (ci, cj) corresponds to C[2*ci:2*ci+2, 2*cj:2*cj+2]
    # For W: W[k,i,r]. k is row of C (col of B), i is col of C (row of A).
    # If C block is (ci, cj), then i = 2*ci + a, k = 2*cj + b.
    # Note: W2 index is (b, a, t) because W2 is for 2x2, W2[k*n+i] -> W2[b*2+a] in 2x2 case?
    # Let's re-check W2 indexing.
    # W2[k,i,r] for 2x2. k in {0,1}, i in {0,1}.
    # W2 is stored as (4, 7). Index k*2 + i.
    # So W2[b*2+a, t] corresponds to k=b, i=a.
    
    # Let's list the terms again with indices.
    terms = [
        # s, A_blocks, B_blocks, C_blocks
        # (block_index: coeff)
        (0, [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        (1, [(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        (2, [(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,-1)]),
        (3, [(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        (4, [(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        (5, [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        (6, [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,-1)])
    ]
    
    for s, A_blocks, B_blocks, C_blocks in terms:
        for t in range(7):
            r = s * 7 + t
            
            # Fill U
            for bi, bj, coeff in A_blocks:
                for a in range(2):
                    for b in range(2):
                        u2_val = U2[a*2+b, t]
                        if u2_val != 0:
                            i = 2*bi + a
                            j = 2*bj + b
                            U[i*4+j, r] += coeff * u2_val
                            
            # Fill V
            for bi, bj, coeff in B_blocks:
                for a in range(2):
                    for b in range(2):
                        v2_val = V2[a*2+b, t]
                        if v2_val != 0:
                            j = 2*bi + a
                            k = 2*bj + b
                            V[j*4+k, r] += coeff * v2_val
                            
            # Fill W
            for ci, cj, coeff in C_blocks:
                for a in range(2):
                    for b in range(2):
                        w2_val = W2[b*2+a, t] # k=b, i=a
                        if w2_val != 0:
                            i = 2*ci + a
                            k = 2*cj + b
                            W[k*4+i, r] += coeff * w2_val
                            
    return U, V, W

def construct():
    deadline = time.time() + 115.0
    
    n, m, p = DIMENSIONS
    
    # Try Strassen Rank 49
    try:
        U, V, W = _construct_strassen_4x4()
        if time.time() < deadline:
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass
    
    # Fallback: Schoolbook Rank 64
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

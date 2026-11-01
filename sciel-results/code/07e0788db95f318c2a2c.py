"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_decomposition():
    """
    Return U, V, W for 2x2 matrix multiplication using Strassen's 7-multiplication algorithm.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*m+j, c] for input A[i,j], V[j*p+k, c] for input B[j,k], W[k*n+i, c] for output C.
    n=m=p=2.
    """
    n = m = p = 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen multiplications:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    
    # M1: A00+A11 times B00+B11, contributes +1 to C00, +1 to C11
    # U: A00 (idx 0) = 1, A11 (idx 3) = 1
    # V: B00 (idx 0) = 1, B11 (idx 3) = 1
    # W: C00 (k=0,i=0, idx 0) = 1, C11 (k=1,i=1, idx 3) = 1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2: (A10+A11) * B00, contributes +1 to C10
    # U: A10 (idx 2) = 1, A11 (idx 3) = 1
    # V: B00 (idx 0) = 1
    # W: C10 (k=0,i=1, idx 1) = 1
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1
    
    # M3: A00 * (B01-B11), contributes +1 to C01, -1 to C11
    # U: A00 (idx 0) = 1
    # V: B01 (idx 1) = 1, B11 (idx 3) = -1
    # W: C01 (k=1,i=0, idx 2) = 1, C11 (k=1,i=1, idx 3) = -1
    U[0][2] += 1
    V[1][2] += 1; V[3][2] += -1
    W[2][2] += 1; W[3][2] += -1
    
    # M4: A11 * (B10-B00), contributes +1 to C00, +1 to C10
    # U: A11 (idx 3) = 1
    # V: B10 (idx 2) = 1, B00 (idx 0) = -1
    # W: C00 (k=0,i=0, idx 0) = 1, C10 (k=0,i=1, idx 1) = 1
    U[3][3] += 1
    V[2][3] += 1; V[0][3] += -1
    W[0][3] += 1; W[1][3] += 1
    
    # M5: (A00+A01) * B11, contributes -1 to C00, +1 to C01
    # U: A00 (idx 0) = 1, A01 (idx 1) = 1
    # V: B11 (idx 3) = 1
    # W: C00 (k=0,i=0, idx 0) = -1, C01 (k=1,i=0, idx 2) = 1
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] += -1; W[2][4] += 1
    
    # M6: (A10-A00) * (B00+B01), contributes +1 to C11
    # U: A10 (idx 2) = 1, A00 (idx 0) = -1
    # V: B00 (idx 0) = 1, B01 (idx 1) = 1
    # W: C11 (k=1,i=1, idx 3) = 1
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7: (A01-A11) * (B10+B11), contributes +1 to C00, -1 to C11
    # U: A01 (idx 1) = 1, A11 (idx 3) = -1
    # V: B10 (idx 2) = 1, B11 (idx 3) = 1
    # W: C00 (k=0,i=0, idx 0) = 1, C11 (k=1,i=1, idx 3) = -1
    U[1][6] += 1; U[3][6] += -1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1; W[3][6] += -1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0]) if U else 0
    # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_4x4_decomposition():
    """
    Construct rank-49 decomposition for 4x4 matrix multiplication using two levels of Strassen.
    
    Level 1: Partition 4x4 into 2x2 blocks, use 7 block multiplications.
    Level 2: Each 2x2 block multiplication uses Strassen's 7 scalar multiplications.
    Total: 7 * 7 = 49 scalar multiplications.
    """
    n = m = p = 4
    rank = 49
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 49
    
    # Get the 2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2_decomposition()
    
    # 4x4 matrices partitioned into 2x2 blocks:
    # A = [[A00, A01],     B = [[B00, B01],
    #      [A10, A11]]         [B10, B11]]
    # C = A * B = [[C00, C01],
    #              [C10, C11]]
    
    # Block-level Strassen:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each block multiplication Mr, we need to map the 2x2 Strassen coefficients
    # to the correct 4x4 indices.
    
    # Block A00: rows 0-1, cols 0-1 of A
    # Block A01: rows 0-1, cols 2-3 of A
    # Block A10: rows 2-3, cols 0-1 of A
    # Block A11: rows 2-3, cols 2-3 of A
    
    # Similarly for B and C.
    
    # For each of the 7 block multiplications, we have a 2x2 Strassen decomposition.
    # We need to compose them.
    
    # Let's define the block offsets:
    # A block (bi, bj): row offset = bi*2, col offset = bj*2
    # B block (bi, bj): row offset = bi*2, col offset = bj*2
    # C block (bi, bj): row offset = bi*2, col offset = bj*2
    
    # For each block multiplication r (0..6), and each scalar multiplication s (0..6):
    # global index = r * 7 + s
    
    # We need to determine which A block, B block, and C block each term contributes to.
    
    # Block-level Strassen:
    # M1 = (A00+A11) * (B00+B11) -> contributes to C00 (+1), C11 (+1)
    # M2 = (A10+A11) * B00 -> contributes to C10 (+1)
    # M3 = A00 * (B01-B11) -> contributes to C01 (+1), C11 (-1)
    # M4 = A11 * (B10-B00) -> contributes to C00 (+1), C10 (+1)
    # M5 = (A00+A01) * B11 -> contributes to C00 (-1), C01 (+1)
    # M6 = (A10-A00) * (B00+B01) -> contributes to C11 (+1)
    # M7 = (A01-A11) * (B10+B11) -> contributes to C00 (+1), C11 (-1)
    
    # For each block multiplication, we know:
    # - Which A blocks are involved (with coefficients)
    # - Which B blocks are involved (with coefficients)
    # - Which C blocks are involved (with coefficients)
    
    # Let's define the block-level coefficients:
    # block_mults[r] = (a_blocks, b_blocks, c_blocks)
    # where each is a dict: {block_idx: coefficient}
    
    block_mults = [
        # M1: (A00+A11) * (B00+B11) -> C00(+1), C11(+1)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2: (A10+A11) * B00 -> C10(+1)
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        # M3: A00 * (B01-B11) -> C01(+1), C11(-1)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): -1}),
        # M4: A11 * (B10-B00) -> C00(+1), C10(+1)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5: (A00+A01) * B11 -> C00(-1), C01(+1)
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6: (A10-A00) * (B00+B01) -> C11(+1)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7: (A01-A11) * (B10+B11) -> C00(+1), C11(-1)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): -1}),
    ]
    
    # For each block multiplication r, and each scalar multiplication s in the 2x2 Strassen:
    # The global index is r * 7 + s
    
    # The 2x2 Strassen U2, V2, W2 have indices:
    # U2[i*2+j, s] for A[i,j] in the 2x2 block
    # V2[j*2+k, s] for B[j,k] in the 2x2 block
    # W2[k*2+i, s] for C[k,i] in the 2x2 block
    
    # Wait, I need to be careful about the indexing.
    # In the 2x2 case, n=m=p=2.
    # U2[i*2+j, s] where i is row of A, j is col of A.
    # V2[j*2+k, s] where j is row of B, k is col of B.
    # W2[k*2+i, s] where k is col of B (row of C), i is row of A (col of C).
    
    # Actually, let me re-read the convention.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W[k*n+i, r] where k is the column index of B (which becomes the row index of C)
    # and i is the row index of A (which becomes the column index of C).
    
    # So C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For the 2x2 case:
    # C[i,k] = sum_j sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    
    # Now, when we compose blocks, we need to map:
    # A block (bi, bj) element (i,j) -> global A index (bi*2+i, bj*2+j)
    # B block (bi, bj) element (j,k) -> global B index (bi*2+j, bj*2+k)
    # C block (bi, bj) element (i,k) -> global C index (bi*2+i, bj*2+k)
    # But W uses W[k*n+i, r] = W[k*4+i, r] for the 4x4 case.
    # So C[i,k] corresponds to W[k*4+i, r].
    
    # Let's construct the decomposition:
    
    for r in range(7):  # block multiplication index
        a_blocks, b_blocks, c_blocks = block_mults[r]
        
        for s in range(7):  # scalar multiplication index within the 2x2 Strassen
            global_idx = r * 7 + s
            
            # For U:
            # The A blocks involved in this block multiplication r are a_blocks.
            # For each A block (bi, bj) with coefficient ca:
            #   The 2x2 Strassen U2[i*2+j, s] gives the coefficient for A[i,j] within the block.
            #   The global A index is (bi*2+i)*4 + (bj*2+j) = bi*8 + i*4 + bj*2 + j
            #   Wait, A is 4x4, so index is i*4+j.
            #   Global row = bi*2 + i, global col = bj*2 + j.
            #   Global index = (bi*2+i)*4 + (bj*2+j)
            
            for (bi, bj), ca in a_blocks.items():
                for i in range(2):
                    for j in range(2):
                        u_val = U2[i*2+j][s]
                        if u_val != 0:
                            global_i = bi*2 + i
                            global_j = bj*2 + j
                            global_idx_u = global_i * 4 + global_j
                            U[global_idx_u][global_idx] += ca * u_val
            
            # For V:
            # The B blocks involved in this block multiplication r are b_blocks.
            # For each B block (bi, bj) with coefficient cb:
            #   The 2x2 Strassen V2[j*2+k, s] gives the coefficient for B[j,k] within the block.
            #   The global B index is (bi*2+j)*4 + (bj*2+k)
            
            for (bi, bj), cb in b_blocks.items():
                for j in range(2):
                    for k in range(2):
                        v_val = V2[j*2+k][s]
                        if v_val != 0:
                            global_j = bi*2 + j
                            global_k = bj*2 + k
                            global_idx_v = global_j * 4 + global_k
                            V[global_idx_v][global_idx] += cb * v_val
            
            # For W:
            # The C blocks involved in this block multiplication r are c_blocks.
            # For each C block (bi, bj) with coefficient cc:
            #   The 2x2 Strassen W2[k*2+i, s] gives the coefficient for C[i,k] within the block.
            #   Wait, W2[k*2+i, s] where k is col of B (row of C), i is row of A (col of C).
            #   So C[i,k] within the block.
            #   Global C index: row = bi*2+i, col = bj*2+k.
            #   But W uses W[k*n+i, r] = W[k*4+i, r].
            #   So the global W index is (bj*2+k)*4 + (bi*2+i)
            
            for (bi, bj), cc in c_blocks.items():
                for i in range(2):
                    for k in range(2):
                        w_val = W2[k*2+i][s]
                        if w_val != 0:
                            global_i = bi*2 + i  # row of C
                            global_k = bj*2 + k  # col of C
                            # W index = k*4 + i
                            global_idx_w = global_k * 4 + global_i
                            W[global_idx_w][global_idx] += cc * w_val
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Try the 49-rank Strassen decomposition
    U, V, W = _strassen_4x4_decomposition()
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

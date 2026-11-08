"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x8 matrix.
    
    The matrix multiplication tensor T[i, j, k] corresponds to C[i, k] = sum_j A[i, j] * B[j, k].
    We seek factor matrices U (n*m x r), V (m*p x r), W (p*n x r) such that:
        T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Note the W index convention: W[k*n + i, c]
    
    For n=4, m=4, p=8:
      - U shape: (16, r)
      - V shape: (32, r)  
      - W shape: (32, r)
    
    The schoolbook rank is n*m*p = 128.
    
    We use a structured approach:
    1. Start with the schoolbook decomposition (rank 128).
    2. Try to reduce rank by exploiting symmetries and linear dependencies.
    
    Key insight: For matrix multiplication A(n x m) * B(m x p) = C(n x p),
    the tensor has a special structure. The bilinear complexity is known to be 
    less than n*m*p for many cases.
    
    For 4x4 * 4x8, we can use the fact that this is equivalent to two 4x4*4x4 multiplications
    (splitting the 8 columns into two groups of 4). Each 4x4*4x4 has known bilinear complexity.
    
    The best known bilinear complexity for 4x4 matrix multiplication is 21 (from Winograd's algorithm 
    and subsequent improvements). Actually, for 4x4, the best known is 21.
    
    Wait, let me reconsider. The standard result:
    - 2x2: rank 7 (Strassen)
    - 4x4: rank 21 (using Strassen recursively gives 7^2 = 49, but better algorithms exist)
    
    Actually, the best known bilinear complexity for 4x4 matrix multiplication is 21.
    For 4x4 * 4x8, we can split the 8x1 part into two 4x1 parts, so we need two 4x4*4x4 multiplications.
    If we can share some computations between them, we might do better than 2*21 = 42.
    
    Let me think about this more carefully. We're computing C = A * B where A is 4x4 and B is 4x8.
    This is equivalent to computing C[:, 0:4] = A * B[:, 0:4] and C[:, 4:8] = A * B[:, 4:8].
    
    Using a 4x4*4x4 algorithm of rank r for each, we get rank 2r total if independent.
    With r = 21, that's 42.
    
    But can we do better? Let's try to construct an explicit decomposition.
    
    Actually, let me use a known result: the bilinear complexity of 4x4 matrix multiplication 
    is at most 21. Let me implement the Strassen-based approach for 4x4 which gives rank 49,
    or use a better algorithm.
    
    For simplicity and correctness, let me start with a rank that we can verify:
    - Schoolbook: 128
    - Strassen for 4x4: 7^2 = 49 per 4x4*4x4, so 98 for 4x4*4x8
    - Better algorithms exist but are complex to implement
    
    Let me implement the Strassen algorithm for 4x4*4x4 and use it twice.
    
    Strassen for 2x2*2x2 has rank 7. For 4x4*4x4, we can use Strassen recursively:
    Split 4x4 into four 2x2 blocks, use Strassen on those, giving rank 7*7 = 49.
    
    For 4x4*4x8, split the 4x8 into two 4x4 matrices, giving rank 49*2 = 98.
    
    But wait, we can do better. Let me think about the tensor structure more carefully.
    
    Actually, let me just implement a correct decomposition with rank 98 using Strassen,
    and then see if we can improve.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Start with deadline guard
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 128
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try Strassen-based decomposition for 4x4*4x4, used twice for 4x4*4x8
    # Strassen 2x2*2x2 rank 7 decomposition
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4*4x4, split into 2x2 blocks and apply Strassen
    # This gives rank 49
    
    # Let me construct the Strassen 4x4*4x4 decomposition explicitly
    # A is 4x4, B is 4x4, C is 4x4
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = [[C11, C12], [C21, C22]] where each is 2x2
    
    # Using Strassen on the 2x2 block structure:
    # S1 = A11 + A22, S2 = A21 + A22, S3 = A11, S4 = A22
    # S5 = A11 + A12, S6 = A21 - A11, S7 = A12 - A22
    # T1 = B11 + B22, T2 = B11, T3 = B21 - B22, T4 = B11 - B21
    # T5 = B22, T6 = B11 + B12, T7 = B21 + B22
    
    # M1 = S1*T1, M2 = S2*T2, M3 = S3*T3, M4 = S4*T4
    # M5 = S5*T5, M6 = S6*T6, M7 = S7*T7
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 product, which itself requires rank 7 (Strassen) or rank 4 (schoolbook)
    # If we use Strassen for each 2x2 product, total rank = 7*7 = 49
    
    # Let me build this decomposition step by step
    
    def strassen_2x2_decomposition():
        """
        Returns the Strassen 2x2*2x2 rank-7 decomposition.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Indexing: U[i*2+j, c], V[i*2+j, c], W[i*2+j, c] for 2x2 matrices
        """
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # C = [[c00, c01], [c10, c11]]
        # c00 = a00*b00 + a01*b10
        # c01 = a00*b01 + a01*b11
        # c10 = a10*b00 + a11*b10
        # c11 = a10*b01 + a11*b11
        
        # Strassen:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b00-b10)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        rank = 7
        U = [[0] * rank for _ in range(4)]  # a00, a01, a10, a11
        V = [[0] * rank for _ in range(4)]  # b00, b01, b10, b11
        W = [[0] * rank for _ in range(4)]  # c00, c01, c10, c11
        
        # M1: (a00+a11)(b00+b11)
        U[0][0] = 1; U[2][0] = 1  # a00, a11
        V[0][0] = 1; V[2][0] = 1  # b00, b11
        W[0][0] = 1; W[3][0] = 1  # c00, c11
        
        # M2: (a10+a11)b00
        U[1][1] = 1; U[2][1] = 1  # a10, a11
        V[0][1] = 1  # b00
        W[1][1] = 1; W[3][1] = -1  # c10, c11
        
        # M3: a00(b10-b11)
        U[0][2] = 1  # a00
        V[2][2] = 1; V[3][2] = -1  # b10, b11
        W[1][2] = 1; W[3][2] = 1  # c01, c11
        
        # M4: a11(b00-b10)
        U[2][3] = 1  # a11
        V[0][3] = 1; V[2][3] = -1  # b00, b10
        W[0][3] = 1; W[1][3] = 1  # c00, c10
        
        # M5: (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1  # a00, a01
        V[3][4] = 1  # b11
        W[0][4] = -1; W[1][4] = 1  # c00, c01
        
        # M6: (a10-a00)(b00+b01)
        U[1][5] = 1; U[0][5] = -1  # a10, a00
        V[0][5] = 1; V[1][5] = 1  # b00, b01
        W[3][5] = 1  # c11
        
        # M7: (a01-a11)(b10+b11)
        U[1][6] = 1; U[2][6] = -1  # a01, a11
        V[2][6] = 1; V[3][6] = 1  # b10, b11
        W[0][6] = 1  # c00
        
        return U, V, W
    
    # Now build 4x4*4x4 using Strassen on 2x2 blocks
    def strassen_4x4_decomposition():
        """
        Returns the Strassen-based 4x4*4x4 rank-49 decomposition.
        U: (16, 49), V: (16, 49), W: (16, 49)
        """
        # A is 4x4, indexed as A[i*4+j] for i,j in 0..3
        # Split into 2x2 blocks:
        # A11 = A[0:2, 0:2] = A[0,1,4,5]
        # A12 = A[0:2, 2:4] = A[2,3,6,7]
        # A21 = A[2:4, 0:2] = A[8,9,12,13]
        # A22 = A[2:4, 2:4] = A[10,11,14,15]
        
        # Similarly for B and C.
        
        # The 7 Strassen products at the 4x4 level:
        # S1 = A11 + A22, S2 = A21 + A22, S3 = A11, S4 = A22
        # S5 = A11 + A12, S6 = A21 - A11, S7 = A12 - A22
        # T1 = B11 + B22, T2 = B11, T3 = B21 - B22, T4 = B11 - B21
        # T5 = B22, T6 = B11 + B12, T7 = B21 + B22
        
        # M1 = S1*T1, M2 = S2*T2, M3 = S3*T3, M4 = S4*T4
        # M5 = S5*T5, M6 = S6*T6, M7 = S7*T7
        
        # Each Mi is a 2x2*2x2 product, which we decompose with rank 7 (Strassen).
        # So total rank = 7 * 7 = 49.
        
        rank = 49
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        
        # Get the 2x2 Strassen decomposition
        U2, V2, W2 = strassen_2x2_decomposition()
        
        # Define block mappings
        # For a 2x2 block at position (bi, bj) in a 4x4 matrix, the indices are:
        # (bi*2 + i, bj*2 + j) for i,j in 0..1
        # Flat index: (bi*2 + i) * 4 + (bj*2 + j)
        
        def block_index(bi, bj, i, j):
            return (bi * 2 + i) * 4 + (bj * 2 + j)
        
        # A11: bi=0, bj=0 -> indices 0,1,4,5
        # A12: bi=0, bj=1 -> indices 2,3,6,7
        # A21: bi=1, bj=0 -> indices 8,9,12,13
        # A22: bi=1, bj=1 -> indices 10,11,14,15
        
        # For each of the 7 Strassen products at the 4x4 level,
        # we have a sum/difference of 2x2 blocks for A and B,
        # and then we do a 2x2*2x2 product using Strassen (rank 7).
        
        # Let me define the 7 products more carefully.
        
        # Product 1: S1*T1 where S1 = A11+A22, T1 = B11+B22
        # Product 2: S2*T2 where S2 = A21+A22, T2 = B11
        # Product 3: S3*T3 where S3 = A11, T3 = B21-B22
        # Product 4: S4*T4 where S4 = A22, T4 = B11-B21
        # Product 5: S5*T5 where S5 = A11+A12, T5 = B22
        # Product 6: S6*T6 where S6 = A21-A11, T6 = B11+B12
        # Product 7: S7*T7 where S7 = A12-A22, T7 = B21+B22
        
        # For each product p (0..6), we have:
        # A_part = sum of coefficients * 2x2 blocks
        # B_part = sum of coefficients * 2x2 blocks
        # Then we compute A_part * B_part using Strassen 2x2 (rank 7)
        
        # Let me define the block coefficients for each product
        # Each product has A_blocks = [(block_id, coeff), ...] and B_blocks = [(block_id, coeff), ...]
        
        # Block IDs: 0=A11, 1=A12, 2=A21, 3=A22, similarly for B
        products = [
            # (A_blocks, B_blocks)
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # M1: (A11+A22)(B11+B22)
            ([(2, 1), (3, 1)], [(0, 1)]),            # M2: (A21+A22)B11
            ([(0, 1)], [(2, 1), (3, -1)]),           # M3: A11(B21-B22)
            ([(3, 1)], [(0, 1), (2, -1)]),           # M4: A22(B11-B21)
            ([(0, 1), (1, 1)], [(3, 1)]),            # M5: (A11+A12)B22
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),   # M6: (A21-A11)(B11+B12)
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),   # M7: (A12-A22)(B21+B22)
        ]
        
        # Result combinations:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For each result block (C11, C12, C21, C22), we combine the products.
        
        # Let me build the decomposition column by column.
        # Each product p contributes 7 columns (from Strassen 2x2).
        
        for p in range(7):
            A_blocks = products[p][0]
            B_blocks = products[p][1]
            
            # For each of the 7 Strassen 2x2 factors
            for s in range(7):
                col = p * 7 + s
                
                # Add contributions to U from A_blocks
                for block_id, coeff in A_blocks:
                    # Block block_id in A: bi = block_id // 2, bj = block_id % 2
                    bi = block_id // 2
                    bj = block_id % 2
                    for i in range(2):
                        for j in range(2):
                            idx = block_index(bi, bj, i, j)
                            # U2[i*2+j, s] is the coefficient for the 2x2 Strassen factor
                            U[idx][col] += coeff * U2[i * 2 + j][s]
                
                # Add contributions to V from B_blocks
                for block_id, coeff in B_blocks:
                    bi = block_id // 2
                    bj = block_id % 2
                    for i in range(2):
                        for j in range(2):
                            idx = block_index(bi, bj, i, j)
                            V[idx][col] += coeff * V2[i * 2 + j][s]
                
                # Add contributions to W from result combinations
                # We need to figure out which C block this product contributes to
                # and with what sign
                # C11: M1(+), M4(+), M5(-), M7(+)
                # C12: M3(+), M5(+)
                # C21: M2(+), M4(+)
                # C22: M1(+), M2(-), M3(+), M6(+)
                
                # For product p, determine which C blocks it contributes to
                c_contribs = []
                if p == 0:  # M1
                    c_contribs = [(0, 1), (3, 1)]  # C11(+), C22(+)
                elif p == 1:  # M2
                    c_contribs = [(2, 1), (3, -1)]  # C21(+), C22(-)
                elif p == 2:  # M3
                    c_contribs = [(1, 1), (3, 1)]  # C12(+), C22(+)
                elif p == 3:  # M4
                    c_contribs = [(0, 1), (2, 1)]  # C11(+), C21(+)
                elif p == 4:  # M5
                    c_contribs = [(0, -1), (1, 1)]  # C11(-), C12(+)
                elif p == 5:  # M6
                    c_contribs = [(3, 1)]  # C22(+)
                elif p == 6:  # M7
                    c_contribs = [(0, 1)]  # C11(+)
                
                for c_block_id, coeff in c_contribs:
                    bi = c_block_id // 2
                    bj = c_block_id % 2
                    for i in range(2):
                        for j in range(2):
                            idx = block_index(bi, bj, i, j)
                            W[idx][col] += coeff * W2[i * 2 + j][s]
        
        return U, V, W
    
    # Try the Strassen 4x4*4x4 approach, used twice for 4x4*4x8
    if time.time() < deadline - 20:
        try:
            U44, V44, W44 = strassen_4x4_decomposition()
            
            # Now use this for 4x4*4x8 by splitting B into two 4x4 matrices
            # B[:, 0:4] and B[:, 4:8]
            # C[:, 0:4] = A * B[:, 0:4], C[:, 4:8] = A * B[:, 4:8]
            
            rank_total = 98
            U = [[0] * rank_total for _ in range(16)]
            V = [[0] * rank_total for _ in range(32)]
            W = [[0] * rank_total for _ in range(32)]
            
            # First half: B[:, 0:4] -> indices 0..15 in V, C[:, 0:4] -> indices 0..15 in W
            for i in range(16):
                for c in range(49):
                    U[i][c] = U44[i][c]
                    V[i][c] = V44[i][c]
                    W[i][c] = W44[i][c]
            
            # Second half: B[:, 4:8] -> indices 16..31 in V, C[:, 4:8] -> indices 16..31 in W
            for i in range(16):
                for c in range(49):
                    U[i][c + 49] = U44[i][c]
                    V[i + 16][c + 49] = V44[i][c]
                    W[i + 16][c + 49] = W44[i][c]
            
            # Verify the decomposition
            # The tensor T[i*m+j, j*p+k, k*n+i] should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            # Let me verify with random matrices
            
            np.random.seed(42)
            A_test = np.random.randn(4, 4)
            B_test = np.random.randn(4, 8)
            C_test = A_test @ B_test
            
            # Compute using the decomposition
            C_comp = np.zeros((4, 8))
            for i in range(4):
                for k in range(8):
                    for j in range(4):
                        val = 0
                        for c in range(rank_total):
                            val += U[i * 4 + j][c] * V[j * 8 + k][c] * W[k * 4 + i][c]
                        C_comp[i, k] += val
            
            if np.allclose(C_test, C_comp, atol=1e-9):
                best_rank = rank_total
                best_U, best_V, best_W = U, V, W
            else:
                # Decomposition is incorrect, stick with schoolbook
                pass
        except Exception:
            pass
    
    # Check time and return best result
    if time.time() > deadline - 5:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to improve further if time permits
    # For now, return the best we have
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

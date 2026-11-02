import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-minimal (or lower-rank) tensor decomposition for 4x4 * 4x7 matrix multiplication."""
    n, m, p = 4, 4, 7
    deadline = time.time() + 110  # Leave 10s safety margin
    
    # Strategy: Use Strassen-based block decomposition to reduce rank below schoolbook (112)
    # Split 4x4 * 4x7 into 2x2 blocks
    # A (4x4) split into four 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # B (4x7) split into: B = [[B00, B01, B02], [B10, B11, B12]]
    #   where B00, B10 are 2x2; B01, B11 are 2x2; B02, B12 are 2x3
    # C (4x7) = [[C00, C01, C02], [C10, C11, C12]]
    #   C00 = A00*B00 + A01*B10  (2x2)
    #   C01 = A00*B01 + A01*B11  (2x2)
    #   C02 = A00*B02 + A01*B12  (2x3)
    #   C10 = A10*B00 + A11*B10  (2x2)
    #   C11 = A10*B01 + A11*B11  (2x2)
    #   C12 = A10*B02 + A11*B12  (2x3)
    
    # For 2x2 * 2x2: use Strassen (rank 7) instead of schoolbook (rank 4)
    # For 2x2 * 2x3: need a decomposition. Let's use rank 8 (schoolbook is 6, but we can do better)
    # Actually, 2x2 * 2x3 schoolbook is rank 6. Let's try to find a lower rank.
    # The tensor for 2x2 * 2x3 has 6 entries in output. Minimum rank is at least 6.
    # Let's use schoolbook for 2x2 * 2x3 (rank 6) for now.
    
    # Total rank estimate:
    # 4 blocks of 2x2*2x2 using Strassen: 4 * 7 = 28
    # 2 blocks of 2x2*2x3 using schoolbook: 2 * 6 = 12
    # Total: 40
    
    # But we need to be careful about how we combine these.
    # Each Strassen product for 2x2*2x2 involves 7 scalar multiplications.
    # Each schoolbook product for 2x2*2x3 involves 6 scalar multiplications.
    
    # Let's implement this step by step.
    
    # First, let's define helper functions for Strassen and 2x2*2x3 schoolbook.
    
    def strassen_2x2x2():
        """Return U, V, W matrices for 2x2 * 2x2 using Strassen (rank 7)."""
        # Indices: A is 2x2 (indices 0,1,2,3), B is 2x2 (indices 0,1,2,3), C is 2x2 (indices 0,1,2,3)
        # U has 4 rows (A indices), V has 4 rows (B indices), W has 4 rows (C indices)
        # Each has 7 columns (7 products)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen formulas for [[a,b],[c,d]] * [[e,f],[g,h]]:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        #
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        
        # A indices: a=0, b=1, c=2, d=3
        # B indices: e=0, f=1, g=2, h=3
        # C indices: C00=0, C01=1, C10=2, C11=3
        
        # M1 = (a+d)(e+h)
        U[0, 0] = 1  # a
        U[3, 0] = 1  # d
        V[0, 0] = 1  # e
        V[3, 0] = 1  # h
        W[0, 0] = 1  # C00
        W[3, 0] = 1  # C11
        
        # M2 = (c+d)e
        U[2, 1] = 1  # c
        U[3, 1] = 1  # d
        V[0, 1] = 1  # e
        W[2, 1] = 1  # C10
        W[3, 1] = -1  # C11
        
        # M3 = a(f-h)
        U[0, 2] = 1  # a
        V[1, 2] = 1  # f
        V[3, 2] = -1  # h
        W[1, 2] = 1  # C01
        W[3, 2] = 1  # C11
        
        # M4 = d(g-e)
        U[3, 3] = 1  # d
        V[2, 3] = 1  # g
        V[0, 3] = -1  # e
        W[0, 3] = -1  # C00
        W[2, 3] = 1  # C10
        
        # M5 = (a+b)h
        U[0, 4] = 1  # a
        U[1, 4] = 1  # b
        V[3, 4] = 1  # h
        W[0, 4] = -1  # C00
        W[1, 4] = 1  # C01
        
        # M6 = (c-a)(e+f)
        U[2, 5] = 1  # c
        U[0, 5] = -1  # a
        V[0, 5] = 1  # e
        V[1, 5] = 1  # f
        W[3, 5] = -1  # C11
        
        # M7 = (b-d)(g+h)
        U[1, 6] = 1  # b
        U[3, 6] = -1  # d
        V[2, 6] = 1  # g
        V[3, 6] = 1  # h
        W[0, 6] = 1  # C00
        W[1, 6] = 0  # C01 (no contribution)
        
        return U, V, W
    
    def schoolbook_2x2x3():
        """Return U, V, W matrices for 2x2 * 2x3 using schoolbook (rank 6)."""
        # A is 2x2 (indices 0,1,2,3), B is 2x3 (indices 0,1,2,3,4,5), C is 2x3 (indices 0,1,2,3,4,5)
        # U has 4 rows, V has 6 rows, W has 6 rows, each with 6 columns
        
        U = np.zeros((4, 6))
        V = np.zeros((6, 6))
        W = np.zeros((6, 6))
        
        # Products:
        # P1 = a*e -> C00
        # P2 = a*f -> C01
        # P3 = a*g -> C02
        # P4 = b*h -> C00
        # P5 = b*i -> C01
        # P6 = b*j -> C02
        # P7 = c*e -> C10
        # P8 = c*f -> C11
        # P9 = c*g -> C12
        # P10 = d*h -> C10
        # P11 = d*i -> C11
        # P12 = d*j -> C12
        
        # Wait, that's 12 products for schoolbook. Let me reconsider.
        # 2x2 * 2x3: A has 4 entries, B has 6 entries, C has 6 entries.
        # Schoolbook: for each C[i,j], sum over k of A[i,k]*B[k,j]
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        
        # So we need 12 scalar multiplications for schoolbook.
        # But we can potentially do better. Let's use rank 12 for now.
        
        # Actually, let's just use the full schoolbook with 12 products.
        # U: 4 rows (A indices), V: 6 rows (B indices), W: 6 rows (C indices), 12 columns
        
        U = np.zeros((4, 12))
        V = np.zeros((6, 12))
        W = np.zeros((6, 12))
        
        col = 0
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        U[0, col] = 1; V[0, col] = 1; W[0, col] = 1; col += 1  # A00 * B00
        U[1, col] = 1; V[3, col] = 1; W[0, col] = 1; col += 1  # A01 * B10
        
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        U[0, col] = 1; V[1, col] = 1; W[1, col] = 1; col += 1  # A00 * B01
        U[1, col] = 1; V[4, col] = 1; W[1, col] = 1; col += 1  # A01 * B11
        
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        U[0, col] = 1; V[2, col] = 1; W[2, col] = 1; col += 1  # A00 * B02
        U[1, col] = 1; V[5, col] = 1; W[2, col] = 1; col += 1  # A01 * B12
        
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        U[2, col] = 1; V[0, col] = 1; W[3, col] = 1; col += 1  # A10 * B00
        U[3, col] = 1; V[3, col] = 1; W[3, col] = 1; col += 1  # A11 * B10
        
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        U[2, col] = 1; V[1, col] = 1; W[4, col] = 1; col += 1  # A10 * B01
        U[3, col] = 1; V[4, col] = 1; W[4, col] = 1; col += 1  # A11 * B11
        
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        U[2, col] = 1; V[2, col] = 1; W[5, col] = 1; col += 1  # A10 * B02
        U[3, col] = 1; V[5, col] = 1; W[5, col] = 1; col += 1  # A11 * B12
        
        return U, V, W
    
    # Now combine these block decompositions.
    # We have 4 blocks of 2x2*2x2 (using Strassen, rank 7 each) and 2 blocks of 2x2*2x3 (using schoolbook, rank 12 each).
    # Total rank: 4*7 + 2*12 = 28 + 24 = 52
    
    # But we need to map the block indices to the global indices.
    # Let's build the full U, V, W matrices.
    
    total_rank = 4 * 7 + 2 * 12  # 52
    U_full = np.zeros((n * m, total_rank))  # 16 x 52
    V_full = np.zeros((m * p, total_rank))  # 28 x 52
    W_full = np.zeros((p * n, total_rank))  # 28 x 52
    
    col_offset = 0
    
    # Block (0,0): A00 * B00 + A01 * B10 -> C00 (2x2)
    # A00 is rows 0-1, cols 0-1 of A -> global A indices 0,1,4,5
    # B00 is rows 0-1, cols 0-1 of B -> global B indices 0,1,7,8
    # C00 is rows 0-1, cols 0-1 of C -> global C indices 0,1,4,5
    
    U_str, V_str, W_str = strassen_2x2x2()
    
    # Map A indices: local 0,1,2,3 -> global 0,1,4,5
    a_map = [0, 1, 4, 5]
    # Map B indices: local 0,1,2,3 -> global 0,1,7,8
    b_map = [0, 1, 7, 8]
    # Map C indices: local 0,1,2,3 -> global 0,1,4,5
    c_map = [0, 1, 4, 5]
    
    for c in range(7):
        for i in range(4):
            U_full[a_map[i], col_offset + c] += U_str[i, c]
        for j in range(4):
            V_full[b_map[j], col_offset + c] += V_str[j, c]
        for k in range(4):
            W_full[c_map[k], col_offset + c] += W_str[k, c]
    col_offset += 7
    
    # Block (0,1): A00 * B01 + A01 * B11 -> C01 (2x2)
    # A00: same as above -> global 0,1,4,5
    # B01: rows 0-1, cols 2-3 of B -> global B indices 2,3,9,10
    # C01: rows 0-1, cols 2-3 of C -> global C indices 2,3,6,7
    
    U_str, V_str, W_str = strassen_2x2x2()
    
    a_map = [0, 1, 4, 5]
    b_map = [2, 3, 9, 10]
    c_map = [2, 3, 6, 7]
    
    for c in range(7):
        for i in range(4):
            U_full[a_map[i], col_offset + c] += U_str[i, c]
        for j in range(4):
            V_full[b_map[j], col_offset + c] += V_str[j, c]
        for k in range(4):
            W_full[c_map[k], col_offset + c] += W_str[k, c]
    col_offset += 7
    
    # Block (1,0): A10 * B00 + A11 * B10 -> C10 (2x2)
    # A10: rows 2-3, cols 0-1 of A -> global A indices 8,9,12,13
    # B00: same as first block -> global B indices 0,1,7,8
    # C10: rows 2-3, cols 0-1 of C -> global C indices 8,9,12,13
    
    U_str, V_str, W_str = strassen_2x2x2()
    
    a_map = [8, 9, 12, 13]
    b_map = [0, 1, 7, 8]
    c_map = [8, 9, 12, 13]
    
    for c in range(7):
        for i in range(4):
            U_full[a_map[i], col_offset + c] += U_str[i, c]
        for j in range(4):
            V_full[b_map[j], col_offset + c] += V_str[j, c]
        for k in range(4):
            W_full[c_map[k], col_offset + c] += W_str[k, c]
    col_offset += 7
    
    # Block (1,1): A10 * B01 + A11 * B11 -> C11 (2x2)
    # A10: same as above -> global 8,9,12,13
    # B01: same as second block -> global B indices 2,3,9,10
    # C11: rows 2-3, cols 2-3 of C -> global C indices 10,11,14,15
    
    U_str, V_str, W_str = strassen_2x2x2()
    
    a_map = [8, 9, 12, 13]
    b_map = [2, 3, 9, 10]
    c_map = [10, 11, 14, 15]
    
    for c in range(7):
        for i in range(4):
            U_full[a_map[i], col_offset + c] += U_str[i, c]
        for j in range(4):
            V_full[b_map[j], col_offset + c] += V_str[j, c]
        for k in range(4):
            W_full[c_map[k], col_offset + c] += W_str[k, c]
    col_offset += 7
    
    # Block (0,2): A00 * B02 + A01 * B12 -> C02 (2x3)
    # A00: global 0,1,4,5
    # B02: rows 0-1, cols 4-6 of B -> global B indices 4,5,6,11,12,13
    # C02: rows 0-1, cols 4-6 of C -> global C indices 4,5,6,10,11,12
    
    U_sb, V_sb, W_sb = schoolbook_2x2x3()
    
    a_map = [0, 1, 4, 5]
    b_map = [4, 5, 6, 11, 12, 13]
    c_map = [4, 5, 6, 10, 11, 12]
    
    for c in range(12):
        for i in range(4):
            U_full[a_map[i], col_offset + c] += U_sb[i, c]
        for j in range(6):
            V_full[b_map[j], col_offset + c] += V_sb[j, c]
        for k in range(6):
            W_full[c_map[k], col_offset + c] += W_sb[k, c]
    col_offset += 12
    
    # Block (1,2): A10 * B02 + A11 * B12 -> C12 (2x3)
    # A10: global 8,9,12,13
    # B02: same as above -> global B indices 4,5,6,11,12,13
    # C12: rows 2-3, cols 4-6 of C -> global C indices 16,17,18,22,23,24
    
    U_sb, V_sb, W_sb = schoolbook_2x2x3()
    
    a_map = [8, 9, 12, 13]
    b_map = [4, 5, 6, 11, 12, 13]
    c_map = [16, 17, 18, 22, 23, 24]
    
    for c in range(12):
        for i in range(4):
            U_full[a_map[i], col_offset + c] += U_sb[i, c]
        for j in range(6):
            V_full[b_map[j], col_offset + c] += V_sb[j, c]
        for k in range(6):
            W_full[c_map[k], col_offset + c] += W_sb[k, c]
    col_offset += 12
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        """Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k."""
        # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Should equal 1 for all i,j,k
        
        # For efficiency, check a subset
        import random
        random.seed(42)
        
        num_checks = min(1000, n * m * p)
        for _ in range(num_checks):
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            
            val = 0
            for c in range(U.shape[1]):
                val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
            
            if abs(val - 1) > 1e-10:
                return False
        return True
    
    if not verify(U_full, V_full, W_full, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # We'll use Strassen's algorithm for 4x4x4 matrix multiplication.
    # The 4x4 matrices are partitioned into 2x2 blocks.
    # Strassen uses 7 block multiplications instead of 4.
    # Each block multiplication is a 2x2x2 multiplication.
    # We can use Strassen for the 2x2x2 as well (rank 7), giving total rank 7*7=49.
    # Or use schoolbook for 2x2x2 (rank 4), giving total rank 7*4=28.
    
    # Let's use Strassen for both levels: rank 49 for 4x4x4.
    # For 4x4x9: we have 9 columns. We can group them as 4+4+1.
    # First 4 columns: rank 49
    # Next 4 columns: rank 49  
    # Last 1 column: schoolbook rank 4*4*1 = 16
    # Total: 49 + 49 + 16 = 114
    
    # Actually, let me use schoolbook for the inner 2x2x2 to get rank 28 per 4x4x4 block.
    # 28 + 28 + 16 = 72. That's better.
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * rank for _ in range(m * p)]  # 36 x 72
    W = [[0] * rank for _ in range(p * n)]  # 36 x 72
    
    def add_strassen_2x2(U, V, W, a_off, b_off, c_off, col_start):
        """Add Strassen 2x2x2 decomposition starting at column col_start."""
        c = col_start
        # M1 = (A11+A22)(B11+B22)
        U[a_off + 0][c] = 1; U[a_off + 3][c] = 1
        V[b_off + 0][c] = 1; V[b_off + 3][c] = 1
        W[c_off + 0][c] = 1; W[c_off + 3][c] = 1
        c += 1
        # M2 = (A21+A22)B11
        U[a_off + 2][c] = 1; U[a_off + 3][c] = 1
        V[b_off + 0][c] = 1
        W[c_off + 2][c] = 1; W[c_off + 3][c] = -1
        c += 1
        # M3 = A11(B21-B22)
        U[a_off + 0][c] = 1
        V[b_off + 2][c] = 1; V[b_off + 3][c] = -1
        W[c_off + 1][c] = 1; W[c_off + 3][c] = 1
        c += 1
        # M4 = A22(B12-B11)
        U[a_off + 3][c] = 1
        V[b_off + 1][c] = 1; V[b_off + 0][c] = -1
        W[c_off + 0][c] = 1; W[c_off + 2][c] = 1
        c += 1
        # M5 = (A11+A12)B22
        U[a_off + 0][c] = 1; U[a_off + 1][c] = 1
        V[b_off + 3][c] = 1
        W[c_off + 0][c] = -1; W[c_off + 1][c] = 1
        c += 1
        # M6 = (A21-A11)(B11+B12)
        U[a_off + 2][c] = 1; U[a_off + 0][c] = -1
        V[b_off + 0][c] = 1; V[b_off + 1][c] = 1
        W[c_off + 3][c] = 1
        c += 1
        # M7 = (A12-A22)(B21+B22)
        U[a_off + 1][c] = 1; U[a_off + 3][c] = -1
        V[b_off + 2][c] = 1; V[b_off + 3][c] = 1
        W[c_off + 0][c] = 1
        c += 1
        return c
    
    def add_4x4x4_strassen_schoolbook(U, V, W, a_off, b_off, c_off, col_start):
        """
        Add 28-rank decomposition for 4x4x4 using Strassen for outer 2x2 blocks
        and schoolbook for inner 2x2x2 multiplications.
        
        A, B, C are 4x4 matrices partitioned into 2x2 blocks:
        A = [[A11, A12], [A21, A22]]
        B = [[B11, B12], [B21, B22]]
        C = [[C11, C12], [C21, C22]]
        
        Strassen products:
        M1 = (A11+A22)(B11+B22)
        M2 = (A21+A22)B11
        M3 = A11(B21-B22)
        M4 = A22(B12-B11)
        M5 = (A11+A12)B22
        M6 = (A21-A11)(B11+B12)
        M7 = (A12-A22)(B21+B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        c = col_start
        
        # For each Strassen product, we need to compute a 2x2 matrix product.
        # Each 2x2 product uses 4 scalar multiplications (schoolbook).
        
        # M1 = (A11+A22)(B11+B22)
        # A11+A22 is a 2x2 matrix with entries:
        #   (a00+a20, a01+a21)
        #   (a10+a30, a11+a31)
        # B11+B22 is a 2x2 matrix with entries:
        #   (b00+b20, b01+b21)
        #   (b10+b30, b11+b31)
        # The 2x2 product has 4 scalar multiplications.
        
        # Let me define a helper to add a 2x2 schoolbook product
        def add_2x2_schoolbook(U, V, W, u_entries, v_entries, w_entries, col):
            """
            u_entries: list of 4 (U_row, coeff) pairs for the 2x2 A matrix
            v_entries: list of 4 (V_row, coeff) pairs for the 2x2 B matrix  
            w_entries: list of 4 (W_row, coeff) pairs for the 2x2 C matrix
            Returns new col index
            """
            # A = [[a00, a01], [a10, a11]]
            # B = [[b00, b01], [b10, b11]]
            # C = [[c00, c01], [c10, c11]]
            # c00 = a00*b00 + a01*b10
            # c01 = a00*b01 + a01*b11
            # c10 = a10*b00 + a11*b10
            # c11 = a10*b01 + a11*b11
            
            # 4 scalar multiplications:
            # s1 = a00*b00
            # s2 = a01*b10
            # s3 = a00*b01
            # s4 = a01*b11
            # s5 = a10*b00
            # s6 = a11*b10
            # s7 = a10*b01
            # s8 = a11*b11
            
            # Wait, that's 8. Schoolbook for 2x2x2 is rank 4? No, it's rank 4 only with Strassen.
            # Schoolbook for 2x2x2 is rank 4: 
            # c00 = a00*b00 + a01*b10
            # c01 = a00*b01 + a01*b11
            # c10 = a10*b00 + a11*b10
            # c11 = a10*b01 + a11*b11
            # That's 8 scalar multiplications, not 4!
            
            # I was confused. Schoolbook 2x2x2 has rank 4? No, it has rank 4 only in the sense
            # that there are 4 output entries, but each requires 2 multiplications.
            # Total scalar multiplications = 4*2 = 8? No wait.
            
            # Actually, for 2x2 matrix multiplication:
            # c00 = a00*b00 + a01*b10  (2 mults)
            # c01 = a00*b01 + a01*b11  (2 mults)
            # c10 = a10*b00 + a11*b10  (2 mults)
            # c11 = a10*b01 + a11*b11  (2 mults)
            # Total: 8 scalar multiplications.
            
            # But Strassen reduces this to 7.
            
            # So schoolbook 2x2x2 has rank 8, not 4!
            # My earlier calculation was wrong.
            
            # Let me reconsider. The rank of 2x2x2 matrix multiplication is 7 (Strassen).
            # Schoolbook is 8.
            
            # So for 4x4x4 with Strassen outer + schoolbook inner: 7*8 = 56
            # With Strassen outer + Strassen inner: 7*7 = 49
            
            # Let me use Strassen for both: rank 49 per 4x4x4 block.
            # Total: 49 + 49 + 16 = 114
            
            # But wait, I can do better. Let me just use schoolbook for the whole 4x4x4.
            # That's rank 64 (4*4*4 = 64).
            # For 4x4x9: 64*2 + 16 = 144. That's the baseline.
            
            # Hmm, let me just implement Strassen properly for 4x4x4 with rank 49.
            return col
        
        # I'm getting confused. Let me start fresh with a clean implementation.
        return c
    
    # Let me just use schoolbook for simplicity and correctness.
    # Rank = 4*4*9 = 144
    
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

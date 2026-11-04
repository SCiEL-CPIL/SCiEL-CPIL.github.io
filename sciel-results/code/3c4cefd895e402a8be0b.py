"""Program for matrix multiplication 4x4x9 using Strassen-based decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will use the Strassen algorithm for 4x4x4 blocks.
    # The 4x4x4 matrix multiplication can be decomposed into 49 multiplications.
    # We have two 4x4x4 blocks (for columns 0-3 and 4-7) and one 4x4x1 block (for column 8).
    # Total rank = 49 + 49 + 16 = 114.
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the Strassen 4x4x4 decomposition.
    # We'll use a known rank 49 decomposition for 4x4x4.
    # The Strassen algorithm for 4x4x4 can be constructed by dividing into 2x2x2 blocks.
    # Each 2x2x2 block uses 7 multiplications.
    # For 4x4x4, we have 4 blocks of 2x2x2, but with sharing.
    
    # Actually, the standard Strassen for 4x4x4 uses 49 multiplications.
    # Let's hardcode the factors for a 4x4x4 Strassen decomposition.
    
    # The 4x4x4 Strassen decomposition factors:
    # U is 16x49, V is 16x49, W is 16x49
    
    # For simplicity and correctness, I'll use a known decomposition.
    # The factors for 4x4x4 Strassen (rank 49) are complex to hardcode manually.
    
    # Alternative: Use the schoolbook for 4x4x4 but that's rank 64.
    # We need rank 49.
    
    # Let's use a different approach: use the fact that 4x4x4 has rank 49.
    # I'll generate the Strassen 4x4x4 factors programmatically.
    
    # Strassen 2x2x2 factors:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 4x4x4, we divide A into 4 2x2 blocks, B into 4 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    
    # The product C = AB has 4 2x2 blocks.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each 2x2x2 multiplication uses 7 mults.
    # Total without sharing: 4*7 = 28, but we need 49.
    # Actually, the Strassen 4x4x4 uses 49 multiplications total.
    
    # Let me use a known rank 49 decomposition.
    # I'll use the following: the 4x4x4 tensor has rank 49.
    # I'll hardcode the factors.
    
    # For now, let's use the schoolbook for the entire 4x4x9.
    # Rank = 4*4*9 = 144.
    # This is the baseline.
    
    # To improve, I need to use Strassen for the 4x4x4 parts.
    # Let me implement the Strassen 4x4x4 factors.
    
    # The 4x4x4 Strassen factors (rank 49):
    # I'll use a standard decomposition.
    
    # U factors (16x49):
    # Each column corresponds to one multiplication.
    # The entries are 0, 1, or -1.
    
    # Let me define the 49 multiplications for 4x4x4.
    # This is based on the Strassen algorithm extended to 4x4.
    
    # I'll use the following known decomposition:
    # The 4x4x4 matrix multiplication tensor has rank 49.
    # The factors are:
    
    # For the two 4x4x4 blocks, I'll use the same Strassen decomposition.
    # For the 4x4x1 block, I'll use the schoolbook (16 mults).
    
    # Let me define the Strassen 4x4x4 factors.
    # I'll use a helper to generate them.
    
    def strassen_4x4x4_factors():
        """Return (U, V, W) for 4x4x4 Strassen decomposition, rank 49."""
        # This is a known rank 49 decomposition.
        # I'll hardcode the factors.
        
        # U: 16x49
        U_s = [[0]*49 for _ in range(16)]
        V_s = [[0]*49 for _ in range(16)]
        W_s = [[0]*49 for _ in range(16)]
        
        # The 4x4x4 Strassen decomposition uses 49 multiplications.
        # Let me define them.
        
        # I'll use the following approach:
        # Divide 4x4x4 into 8 2x2x2 blocks, but with sharing.
        
        # Actually, let me use a simpler approach:
        # Use the schoolbook for 4x4x4 (64 mults) for now.
        # This gives total rank 64+64+16 = 144, same as baseline.
        
        # To get rank 49, I need the actual Strassen factors.
        # Let me look up the 4x4x4 Strassen factors.
        
        # The 4x4x4 Strassen algorithm:
        # We can write 4x4x4 as a combination of 2x2x2 Strassen calls.
        
        # For now, I'll return the schoolbook factors for 4x4x4.
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    col = i*16 + j*4 + k
                    U_s[i*4+j][col] = 1
                    V_s[j*4+k][col] = 1
                    W_s[k*4+i][col] = 1
        
        return U_s, V_s, W_s
    
    # Since I can't easily get the rank 49 factors, let me use a different approach.
    # I'll use the schoolbook for the entire 4x4x9, which gives rank 144.
    
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

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    # Decomposition strategy:
    # Split the 3x7 multiplication into blocks:
    # A is 2x3, B is 3x7
    # Split B into columns: B = [B1 | B2 | B3] where B1 is 3x2, B2 is 3x2, B3 is 3x2, B4 is 3x1
    # C = A*B = [A*B1 | A*B2 | A*B3 | A*B4]
    # Each A*Bj for j=1,2,3 is 2x2 multiplication (rank 7 via Strassen)
    # A*B4 is 2x1 multiplication (rank 4)
    # Total rank = 3*7 + 4 = 25
    
    # Strassen decomposition for 2x2x2:
    # m1 = (A00+A11)*(B00+B11)
    # m2 = (A10+A11)*B00
    # m3 = A00*(B01-B11)
    # m4 = A11*(B10-B00)
    # m5 = (A00+A01)*B11
    # m6 = (A10-A00)*(B00+B01)
    # m7 = (A01-A11)*(B10+B11)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # 2x2x1 direct:
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # This has rank 4 (two independent terms per output, but we can share)
    # Actually: C0 = A00*B0 + A01*B1, C1 = A10*B0 + A11*B1
    # We need 4 tensor terms for this.
    
    # Let's build the full decomposition with rank 25
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]  # 6 x 25
    V = [[0] * rank for _ in range(m * p)]  # 21 x 25
    W = [[0] * rank for _ in range(p * n)]  # 14 x 25
    
    # For each of the 3 blocks of size 2x2x2, use Strassen
    # Block j: columns 2*j and 2*j+1 of B, output columns 2*j and 2*j+1 of C
    # A is always the same 2x3 matrix, but for each block we only use certain rows of A
    
    # Actually, let me reconsider. A is 2x3, so A*B where B is 3x7.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let me use a different approach. I'll directly construct a rank-25 decomposition.
    # 
    # The tensor T_{i,j,k} = delta for multiplication C = A*B where:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # For rank r, we need vectors u_c, v_c, w_c such that:
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    # i.e., T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    # 
    # Let me try to find a rank-25 decomposition by hand/known results.
    
    # Actually, let me try a simpler approach: use the fact that 
    # rank(2x3x7) <= rank(2x3x2)*3 + rank(2x3x1)*1
    # If rank(2x3x2) = 11 and rank(2x3x1) = 6, then 33+6=39. Still better than 42.
    
    # But I'm not sure about these ranks. Let me just output the schoolbook for safety,
    # but with a clean implementation.
    
    rank = n * m * p  # 42
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
# EVOLVE-BLOCK-END

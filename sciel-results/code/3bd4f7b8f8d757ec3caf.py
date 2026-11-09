"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Attempt a rank-15 decomposition using Strassen-like bilinear identities
    # For 2x3x3, we group terms to exploit (a+b)(c+d) = ac+ad+bc+bd
    
    rank = 15
    # U: 6x15, V: 9x15, W: 6x15
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strategy: Use 3 Strassen-like identities to reduce 18 terms to 15
    # Each identity saves 1 multiplication by combining 4 terms into 3
    
    # Define the decomposition coefficients
    # We'll use a structured pattern based on splitting the 3x3 matrix
    
    # Coefficients for U (input A indices: 0,1,2 for row 0; 3,4,5 for row 1)
    # Coefficients for V (input B indices: 0,1,2 for col 0; 3,4,5 for col 1; 6,7,8 for col 2)
    # Coefficients for W (output indices: 0,1,2 for col 0; 3,4,5 for col 1)
    
    # Base terms (9 terms for the 3x3 part of each row)
    # Row 0 of A: a0, a1, a2
    # Row 1 of A: a3, a4, a5
    
    # Let's try a specific rank-15 ansatz
    # This is a known structure for rectangular matrix multiplication
    
    # Term 0-2: Handle column 0 of B with Strassen identity
    # M0 = (a0+a1)(b0+b1) -> covers a0b0, a0b1, a1b0, a1b1
    # But we need to be careful with the 2x3 structure
    
    # Actually, let me use a verified rank-15 decomposition for 2x3x3
    # Reference: Known results for rectangular matrix multiplication
    
    # I'll construct a specific integer-valued decomposition
    # Using the pattern from Winograd-style optimizations
    
    # For now, use a conservative rank-16 approach that's guaranteed to work
    # by combining pairs of terms
    
    # Let me try rank=16 first as a safe improvement
    rank = 16
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Keep 14 individual terms and combine 4 terms into 3 using one Strassen identity
    # That gives 14 + 3 = 17... let me try two identities: 12 + 3 + 3 = 18? No.
    
    # Let me use a different approach: known rank for 2x3x3 is 15 or 16
    # I'll implement a rank-16 decomposition
    
    # Individual terms for most positions
    term_id = 0
    # Terms for C[0,0] = a0b0 + a1b1 + a2b2
    # Terms for C[0,1] = a0b3 + a1b4 + a2b5
    # Terms for C[0,2] = a0b6 + a1b7 + a2b8
    # Terms for C[1,0] = a3b0 + a4b1 + a5b2
    # Terms for C[1,1] = a3b3 + a4b4 + a5b5
    # Terms for C[1,2] = a3b6 + a4b7 + a5b8
    
    # Use Strassen identity on (a0+a1)(b0+b1) for the top-left 2x2 block of B
    # This combines: a0b0, a0b1, a1b0, a1b1 into 3 terms
    
    # But we only have a0b0, a1b1 in C[0,0] and a0b3, a1b4 in C[0,1]
    # The cross terms a0b1, a1b0 don't appear directly
    
    # Let me try a different grouping. Use (a0+a2)(b0+b2) for diagonal terms
    # This is getting complex. Let me just implement a known rank-16 solution.
    
    # Simple approach: keep all 18 terms but verify they're correct
    # Then try to merge some
    
    # For safety, let me return a valid rank-18 solution first, then optimize
    rank = 18
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

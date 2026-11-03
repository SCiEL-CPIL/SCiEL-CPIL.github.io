"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 5
    # Using a finite-field coding approach to find low-rank decomposition
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
    # We seek rank-r decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    
    # Strategy: Use structured construction with Strassen-like ideas
    # For 3x3x3, Strassen gives rank 7. For 3x3x5, we can use:
    # - 3 Strassen blocks for the 3x3x3 part (but they share structure)
    # - Additional terms for the 2 extra columns
    
    # Better: Use the fact that rank(3,3,5) can be bounded by considering
    # the tensor as a linear code and finding minimal spanning set.
    
    # Explicit construction using a known rank-34 method:
    # Split into 3+2 columns. For 3x3x3 use rank-23 (Strassen variant).
    # For the remaining 2 columns, use 11 terms that share structure.
    # Total: 23 + 11 = 34
    
    # Actually, let me use a verified construction:
    # rank(3,3,5) <= 34 via the following explicit formula.
    
    # I'll implement a rank-34 construction based on:
    # Terms 0-22: 3x3x3 decomposition (rank 23)
    # Terms 23-33: 11 terms for columns 3,4
    
    # Strassen-like 3x3x3 (rank 23):
    # Using the standard Strassen decomposition extended to 3x3
    
    # Let me use a simpler but valid approach: 
    # rank(3,3,5) = 34 via explicit construction
    
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # 9 x 34
    V = [[0] * rank for _ in range(m * p)]  # 15 x 34
    W = [[0] * rank for _ in range(p * n)]  # 15 x 34
    
    # Construction based on structured tensor decomposition
    # Using the following pattern for 3x3x5:
    
    # Phase 1: Handle 3x3x3 sub-tensor with rank-23 terms
    # Using a 3x3x3 decomposition (not optimal Strassen rank 7, but a valid rank-23)
    # Actually, let me use the schoolbook for 3x3x3 (rank 27) and optimize the 2 extra columns
    
    # Let me try a different approach: use the known fact that
    # rank(3,3,5) <= 34 and implement an explicit construction.
    
    # Based on the structure, I'll use:
    # - 20 terms that cover most of the 3x3x3 part
    # - 14 terms that handle the remaining parts including the 2 extra columns
    
    # For now, let me implement a valid rank-40 construction that I can verify:
    
    # Using the following construction:
    # For each of the 5 output columns, we need 3*3=9 input combinations
    # Total naive: 45 terms
    # We can merge terms that share the same (i,j) or (j,k) patterns
    
    # Explicit rank-40 construction:
    # Terms 0-8: Column 0 (9 terms)
    # Terms 9-17: Column 1 (9 terms)  
    # Terms 18-26: Column 2 (9 terms)
    # Terms 27-35: Column 3 (9 terms, but merge 3 with previous)
    # Terms 36-39: Column 4 (4 terms, using shared structure)
    
    # This gives 9+9+9+6+4 = 37, not 40. Let me be more careful.
    
    # Let me just use a clean rank-45 (schoolbook) for correctness,
    # but I'll try to find a better one.
    
    # For safety, return schoolbook rank-45:
    rank = 45
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

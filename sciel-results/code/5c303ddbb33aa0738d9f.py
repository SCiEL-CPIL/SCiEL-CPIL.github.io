"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank estimate: 2 * (4 * 7 + 16) = 88
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # a_off: offset in A (row, col)
    # b_off: offset in B (row, col)
    # c_off: offset in C (row, col)
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # Strassen intermediate products
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Indices in U (n*m), V (m*p), W (p*n)
        # A indices: (i*m + j)
        # B indices: (j*p + k)
        # C indices: (k*n + i)
        
        # M1
        # U: A11+A22 -> (a_r, a_c) and (a_r+1, a_c+1)
        # V: B11+B22 -> (b_r, b_c) and (b_r+1, b_c+1)
        # W: C11, C12, C21, C22 contributions
        
        # We need to add 7 terms to U, V, W
        
        # Term 1: M1
        # U[ a_r*m + a_c ] += 1
        # U[ (a_r+1)*m + (a_c+1) ] += 1
        # V[ b_r*p + b_c ] += 1
        # V[ (b_r+1)*p + (b_c+1) ] += 1
        # W[ b_c*n + a_r ] += 1 (C11)
        # W[ (b_c+1)*n + a_r ] += 1 (C12)
        # W[ b_c*n + (a_r+1) ] += 1 (C21)
        # W[ (b_c+1)*n + (a_r+1) ] += 1 (C22)
        
        # Wait, W index is k*n + i.
        # C11 is i=a_r, k=b_c. Index: b_c*n + a_r.
        # C12 is i=a_r, k=b_c+1. Index: (b_c+1)*n + a_r.
        # C21 is i=a_r+1, k=b_c. Index: b_c*n + (a_r+1).
        # C22 is i=a_r+1, k=b_c+1. Index: (b_c+1)*n + (a_r+1).
        
        # But Strassen formula:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So we add coefficients to W.
        
        # M1
        U[a_r*m + a_c][col] += 1
        U[(a_r+1)*m + (a_c+1)][col] += 1
        V[b_r*p + b_c][col] += 1
        V[(b_r+1)*p + (b_c+1)][col] += 1
        W[b_c*n + a_r][col] += 1
        W[(b_c+1)*n + a_r][col] += 1
        W[b_c*n + (a_r+1)][col] += 1
        W[(b_c+1)*n + (a_r+1)][col] += 1
        
        col += 1
        
        # M2 = (A21+A22)B11
        U[(a_r+1)*m + a_c][col] += 1
        U[(a_r+1)*m + (a_c+1)][col] += 1
        V[b_r*p + b_c][col] += 1
        W[b_c*n + (a_r+1)][col] += 1 # C21
        W[(b_c+1)*n + (a_r+1)][col] += -1 # C22
        
        col += 1
        
        # M3 = A11(B12-B22)
        U[a_r*m + a_c][col] += 1
        V[b_r*p + (b_c+1)][col] += 1
        V[(b_r+1)*p + (b_c+1)][col] += -1
        W[(b_c+1)*n + a_r][col] += 1 # C12
        W[(b_c+1)*n + (a_r+1)][col] += 1 # C22
        
        col += 1
        
        # M4 = A22(B21-B11)
        U[(a_r+1)*m + (a_c+1)][col] += 1
        V[(b_r+1)*p + b_c][col] += 1
        V[b_r*p + b_c][col] += -1
        W[b_c*n + a_r][col] += 1 # C11
        W[b_c*n + (a_r+1)][col] += 1 # C21
        
        col += 1
        
        # M5 = (A11+A12)B22
        U[a_r*m + a_c][col] += 1
        U[a_r*m + (a_c+1)][col] += 1
        V[(b_r+1)*p + (b_c+1)][col] += 1
        W[b_c*n + a_r][col] += -1 # C11
        W[(b_c+1)*n + a_r][col] += 1 # C12
        
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[(a_r+1)*m + a_c][col] += 1
        U[a_r*m + a_c][col] += -1
        V[b_r*p + b_c][col] += 1
        V[b_r*p + (b_c+1)][col] += 1
        W[(b_c+1)*n + (a_r+1)][col] += 1 # C22
        
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a_r*m + (a_c+1)][col] += 1
        U[(a_r+1)*m + (a_c+1)][col] += -1
        V[(b_r+1)*p + b_c][col] += 1
        V[(b_r+1)*p + (b_c+1)][col] += 1
        W[b_c*n + a_r][col] += 1 # C11
        W[(b_c+1)*n + a_r][col] += -1 # C12
        
        col += 1

    # Helper to add trivial 1x4x4 block
    # a_r: row in A (0-indexed), a_c: col start in A
    # b_r: row start in B, b_c: col start in B
    # c_r: row in C, c_c: col start in C
    def add_trivial_1x4x4(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        for j in range(4):
            for k in range(4):
                U[a_r*m + (a_c + j)][col] += 1
                V[(b_r + j)*p + (b_c + k)][col] += 1
                W[(c_c + k)*n + c_r][col] += 1
                col += 1

    # Process first half of B (cols 0-3)
    # A rows 0-1, cols 0-3
    # B rows 0-3, cols 0-3
    # C rows 0-1, cols 0-3
    
    # 4 Strassen blocks
    add_strassen(0, 0, 0, 0, 0, 0) # A11 B11 -> C11
    add_strassen(0, 2, 2, 0, 0, 0) # A12 B21 -> C11
    add_strassen(0, 0, 0, 2, 0, 2) # A11 B12 -> C12
    add_strassen(0, 2, 2, 2, 0, 2) # A12 B22 -> C12
    
    # 1 Trivial block
    add_trivial_1x4x4(2, 0, 0, 0, 2, 0) # A_bot B_left -> C_bot

    # Process second half of B (cols 4-7)
    # A rows 0-1, cols 0-3
    # B rows 0-3, cols 4-7
    # C rows 0-1, cols 4-7
    
    add_strassen(0, 0, 0, 4, 0, 4) # A11 B11' -> C11'
    add_strassen(0, 2, 2, 4, 0, 4) # A12 B21' -> C11'
    add_strassen(0, 0, 0, 6, 0, 6) # A11 B12' -> C12'
    add_strassen(0, 2, 2, 6, 0, 6) # A12 B22' -> C12'
    
    add_trivial_1x4x4(2, 0, 0, 4, 2, 4) # A_bot B_left' -> C_bot'

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

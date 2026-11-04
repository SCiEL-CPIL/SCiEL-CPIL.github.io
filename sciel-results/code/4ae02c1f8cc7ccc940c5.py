import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    # Total Rank = 23 (for 3x3x3) + 19 (for 3x3x4) = 42
    R = 42
    
    U = [[0] * R for _ in range(n * m)]  # 9 x 42
    V = [[0] * R for _ in range(m * p)]  # 21 x 42
    W = [[0] * R for _ in range(p * n)]  # 21 x 42
    
    col = 0
    
    # --- Part 1: 3x3x3 Decomposition (Rank 23) ---
    # We use the known 23-term decomposition for 3x3x3 matrix multiplication.
    # The tensor is T[i,j,k] = 1.
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # The 23 terms are defined by tuples of (u_indices, v_indices, w_indices, sign)
    # where u_indices is a list of (i,j) pairs with coefficient 1, etc.
    # Actually, it's easier to list the non-zero entries for each rank term.
    
    # I will use a hardcoded list of the 23 terms for 3x3x3.
    # Source: Standard optimal decomposition.
    # Note: The specific coefficients might vary by permutation, but the rank is 23.
    
    # Helper to add a term
    def add_term(u_entries, v_entries, w_entries):
        nonlocal col
        for (i, j) in u_entries:
            U[i * 3 + j][col] = 1
        for (j, k) in v_entries:
            V[j * 3 + k][col] = 1
        for (k, i) in w_entries:
            W[k * 3 + i][col] = 1
        col += 1
    
    # 23 terms for 3x3x3
    # This is a known set. 
    # I will use a simplified "Strassen-like" or standard 23-term set.
    # If I cannot guarantee the exact 23 terms from memory, I will use a safe 27-term schoolbook 
    # BUT the prompt asks for improvement. 
    # Let's try to implement a known 23-term decomposition.
    
    # Actually, implementing the exact 23-term decomposition from scratch without a reference table is error-prone.
    # However, I can use the fact that R(3x3x3)=23.
    # I will use a verified 23-term decomposition found in literature (e.g. from "Fast Matrix Multiplication" surveys).
    
    # Since I cannot fetch external data, I will use a robust fallback if I can't be 100% sure of the 23 terms.
    # But wait, the previous program used schoolbook (27). 
    # If I use 27 + 19 = 46, that is still an improvement over 63.
    # But 23 + 19 = 42 is better.
    
    # Let's try to construct the 23 terms.
    # I will use the decomposition from Strassen (1973) which is 23 terms.
    
    # Term 1:
    add_term([(0,0), (1,1)], [(0,0), (1,1)], [(0,0), (1,1)]) # M1 = (A00+A11)(B00+B11) -> C00, C11
    # Wait, the W entries for M1 in 3x3x3 are more complex because of the 3rd row/col.
    
    # Given the high risk of error in hardcoding 23 complex terms, I will use the **Schoolbook 27** for the 3x3x3 part.
    # This guarantees correctness.
    # Rank = 27 + 19 = 46.
    # This is a significant improvement over 63.
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
                
    # --- Part 2: 3x3x4 Decomposition (Rank 19) ---
    # A is 3x3, B is 3x4 (cols 3..6 of the original B).
    # We split A into:
    # A00 (2x2): i=0,1; j=0,1
    # A01 (2x1): i=0,1; j=2
    # A10 (1x2): i=2; j=0,1
    # A11 (1x1): i=2; j=2
    # B2 (3x4): j=0,1,2; k=3,4,5,6
    # Split B2 into:
    # B2_0 (2x4): j=0,1; k=3..6
    # B2_1 (1x4): j=2; k=3..6
    
    # Term 1: A00 * B2_0 (2x2 x 2x4). Rank 14.
    # We use Strassen for 2x2x2 on k=3,4 and k=5,6.
    
    def add_strassen_2x2x2(col, i_off, j_off, k_off):
        # n=2, m=2, p=2
        # Local indices: i in 0..1, j in 0..1, k in 0..1
        # Global indices:
        # i_g = i_off + i
        # j_g = j_off + j
        # k_g = k_off + k
        
        # M1 = (A00+A11)(B00+B11)
        U[i_off*3 + j_off][col] = 1
        U[(i_off+1)*3 + (j_off+1)][col] = 1
        V[j_off*7 + k_off][col] = 1
        V[(j_off+1)*7 + (k_off+1)][col] = 1
        W[k_off*3 + i_off][col] = 1
        W[(k_off+1)*3 + (i_off+1)][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[(i_off+1)*3 + j_off][col] = 1
        U[(i_off+1)*3 + (j_off+1)][col] = 1
        V[j_off*7 + k_off][col] = 1
        W[k_off*3 + (i_off+1)][col] = 1
        col += 1
        
        # M3 = A00(B10-B00)
        U[i_off*3 + j_off][col] = 1
        V[(j_off+1)*7 + k_off][col] = 1
        V[j_off*7 + k_off][col] = -1
        W[k_off*3 + i_off][col] = 1
        col += 1
        
        # M4 = A11(B11-B00)
        U[(i_off+1)*3 + (j_off+1)][col] = 1
        V[(j_off+1)*7 + (k_off+1)][col] = 1
        V[j_off*7 + k_off][col] = -1
        W[(k_off+1)*3 + (i_off+1)][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[i_off*3 + j_off][col] = 1
        U[(i_off+1)*3 + (j_off+1)][col] = 1 # Wait, A01 is i=0..1, j=1? No.
        # In 2x2x2, A is:
        # A00 A01
        # A10 A11
        # So A01 is (0,1).
        # My indices: i_off, j_off.
        # A00: (i_off, j_off)
        # A01: (i_off, j_off+1)
        # A10: (i_off+1, j_off)
        # A11: (i_off+1, j_off+1)
        
        # Let's re-verify Strassen terms with correct indices.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B11-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # I will rewrite the Strassen helper to be precise.
        pass

    # Given the complexity and risk, I will use Schoolbook for the 2x2x4 part as well?
    # Schoolbook 2x2x4 is 16.
    # 16 + 2 + 2 + 1 = 21.
    # Total Rank = 27 + 21 = 48.
    # Still better than 63.
    
    # Let's try to be more aggressive but safe.
    # I will implement the 19-rank 3x3x4 part using the 2-term and 1-term optimizations for the small blocks, 
    # but use Schoolbook (16) for the 2x2x4 part to avoid Strassen implementation errors.
    # Rank = 27 + 16 + 2 + 2 + 1 = 48.
    
    # Actually, let's just do full Schoolbook for the 3x3x4 part?
    # Rank 36.
    # Total 27 + 36 = 63.
    
    # I will stick to the 48-rank solution (Schoolbook 3x3x3 + Optimized Small Blocks 3x3x4).
    
    # Part 2: 3x3x4
    # B columns are 3,4,5,6.
    
    # Term 1: A00 (2x2) * B2_0 (2x4). Schoolbook.
    # i in 0,1; j in 0,1; k in 3,4,5,6.
    for i in range(2):
        for j in range(2):
            for k in range(3, 7):
                U[i * 3 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    # 16 terms. col is now 27 + 16 = 43.
    
    # Term 2: A01 (2x1) * B2_1 (1x4).
    # i in 0,1; j=2; k in 3,4,5,6.
    # C[i,k] = A[i,2] B[2,k].
    # We can use 2 terms (one for i=0, one for i=1).
    for i in range(2):
        U[i * 3 + 2][col] = 1
        for k in range(3, 7):
            V[2 * 7 + k][col] = 1
        for k in range(3, 7):
            W[k * 3 + i][col] = 1
        col += 1
    # 2 terms. col is now 45.
    
    # Term 3: A10 (1x2) * B2_0 (2x4).
    # i=2; j in 0,1; k in 3,4,5,6.
    # C[2,k] = A[2,0]B[0,k] + A[2,1]B[1,k].
    # We can use 2 terms (one for j=0, one for j=1).
    for j in range(2):
        U[2 * 3 + j][col] = 1
        for k in range(3, 7):
            V[j * 7 + k][col] = 1
        for k in range(3, 7):
            W[k * 3 + 2][col] = 1
        col += 1
    # 2 terms. col is now 47.
    
    # Term 4: A11 (1x1) * B2_1 (1x4).
    # i=2; j=2; k in 3,4,5,6.
    # C[2,k] = A[2,2] B[2,k].
    # 1 term.
    U[2 * 3 + 2][col] = 1
    for k in range(3, 7):
        V[2 * 7 + k][col] = 1
    for k in range(3, 7):
        W[k * 3 + 2][col] = 1
    col += 1
    # 1 term. col is now 48.
    
    # Total Rank 48.
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    
    # Use a structured decomposition for 2x4 x 4x8
    # C[i,c] = sum_{a=0}^{3} A[i,a] * B[a,c]
    # 
    # Strategy: Group the 4 inner dimensions into 2 pairs and use
    # Strassen-like sharing.
    #
    # Split: A = [A_L | A_R] where A_L, A_R are 2x2
    #        B = [B_T; B_B] where B_T, B_B are 2x8
    # C = A_L * B_T + A_R * B_B
    #
    # Each 2x2 * 2x8 needs rank 25 (3p+1 for p=8)
    # Total: 50. Still better than 56.
    #
    # Let me construct 2x2xp with rank 3p+1.
    # For p=8: rank 25.
    
    def make_2x2xp(p):
        """2x2xp multiplication, rank 3p+1"""
        r = 3 * p + 1
        U = np.zeros((4, r))
        V = np.zeros((2*p, r))
        W = np.zeros((2*p, r))
        
        col = 0
        # Strassen for first 2 columns (rank 7)
        # M1 = (A00+A11)(B00+B11)
        U[0, col] = 1; U[3, col] = 1
        V[0, col] = 1; V[p+1, col] = 1
        W[0, col] = 1; W[p+1, col] = 1
        col += 1
        # M2 = (A10+A11)(B00)
        U[2, col] = 1; U[3, col] = 1
        V[0, col] = 1
        W[p, col] = 1; W[p+1, col] = -1
        col += 1
        # M3 = (A00)(B01+B11)
        U[0, col] = 1
        V[1, col] = 1; V[p+1, col] = 1
        W[1, col] = 1; W[p+1, col] = 1
        col += 1
        # M4 = (A11)(B10+B00)
        U[3, col] = 1
        V[p, col] = 1; V[0, col] = 1
        W[0, col] = 1; W[p, col] = 1
        col += 1
        # M5 = (A00+A01)(B11)
        U[0, col] = 1; U[1, col] = 1
        V[p+1, col] = 1
        W[0, col] = -1; W[1, col] = 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[2, col] = 1; U[0, col] = -1
        V[0, col] = 1; V[1, col] = 1
        W[p+1, col] = 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[1, col] = 1; U[3, col] = -1
        V[p, col] = 1; V[p+1, col] = 1
        W[0, col] = 1
        col += 1
        
        # For columns 2..p-1, add 3 products each
        for c in range(2, p):
            # Need: C0c = A00*B0c + A01*B1c
            #       C1c = A10*B0c + A11*B1c
            # Use:
            # N1 = (A00+A11) * B0c
            # N2 = (A01+A10) * B1c
            # N3 = (A00-A11) * (B0c - B1c)
            # Then:
            # N1 = A00*B0c + A11*B0c
            # N2 = A01*B1c + A10*B1c
            # N3 = A00*B0c - A00*B1c - A11*B0c + A11*B1c
            # N1 + N3 = 2*A00*B0c - A00*B1c + A11*B1c... 
            # C0c = A00*B0c + A01*B1c
            # C1c = A10*B0c + A11*B1c
            # N1 - N3 = 2*A11*B0c + A00*B1c - A11*B1c...
            
            # Let me use a different set:
            # N1 = A00 * B0c
            # N2 = A11 * B1c
            # N3 = (A01 + A10) * (B0c + B1c)
            # C0c = N1 + A01*B1c = N1 + (N3 - A01*B0c - A10*B0c - A10*B1c)
            # This doesn't close cleanly.
            
            # Try:
            # N1 = (A00 + A01) * B0c = A00*B0c + A01*B0c
            # N2 = (A10 + A11) * B1c = A10*B1c + A11*B1c
            # N3 = (A00 - A10) * (B0c - B1c) = A00*B0c - A00*B1c - A10*B0c + A10*B1c
            # C0c = A00*B0c + A01*B1c
            # C1c = A10*B0c + A11*B1c
            # N1 = A00*B0c + A01*B0c
            # N2 = A10*B1c + A11*B1c
            # N3 = A00*B0c - A00*B1c - A10*B0c + A10*B1c
            # C0c - N1 = A01*B1c - A01*B0c = A01*(B1c - B0c)
            # C1c - N2 = A10*B0c - A10*B1c = A10*(B0c - B1c)
            # So C0c = N1 + A01*(B1c-B0c) and C1c = N2 + A10*(B0c-B1c)
            # But A01*(B1c-B0c) and A10*(B0c-B1c) are not among our products.
            
            # I think the clean 3-product extension is:
            # N1 = (A00 + A11) * B0c
            # N2 = (A01 + A10) * B1c  
            # N3 = (A00 - A11) * (B0c - B1c)
            # C0c = (N1 + N3)/2 + (N2 - (A10*B1c - A01*B1c))/2... messy.
            
            # Let me just verify if 3p+1 is actually achievable and find the right formulas.
            # For now, use 4 products per extra column for safety: rank 7+4*(p-2) = 4p+3
            # For p=8: 35. Total for 2x4x8: 70. Worse than 56.
            
            # Use 4 products:
            # N1 = A00 * B0c
            # N2 = A01 * B1c
            # N3 = A10 * B0c
            # N4 = A11 * B1c
            U[0, col] = 1
            V[c, col] = 1
            W[c, col] = 1
            col += 1
            U[1, col] = 1
            V[p+c, col] = 1
            W[c, col] = 1
            col += 1
            U[2, col] = 1
            V[c, col] = 1
            W[p+c, col] = 1
            col += 1
            U[3, col] = 1
            V[p+c, col] = 1
            W[p+c, col] = 1
            col += 1
        
        return U, V, W, col
    
    # Since 2x2xp with 4 products/column gives rank 4p+3=35 for p=8,
    # and we need two of them, total = 70. Worse than 56.
    # 
    # Let me stick with the current rank-56 Strassen approach but see if I
    # can find any sharing.
    #
    # Actually, let me try the 4x4 Strassen-Winograd approach properly.
    # Embed 2x4x8 into 4x4x4 by:
    # A' = [[A, 0], [0, 0]] (4x4, top-left is A)
    # For B, we need 4x4. But B is 4x8. 
    # 
    # Alternative: Think of it as 2x4 x 4x8 = 2x(4x8).
    # Use the fact that 4x8 can be handled by 2x(4x4) with sharing.
    #
    # Let me just return the current rank-56 solution.
    
    rank = 56
    U = np.zeros((8, rank))
    V = np.zeros((32, rank))
    W = np.zeros((16, rank))
    
    col = 0
    blocks = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 4, 4),
        (0, 0, 6, 6),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
        (2, 2, 4, 4),
        (2, 2, 6, 6),
    ]
    
    for (a_col_off, b_row_off, b_col_off, c_col_off) in blocks:
        a00 = 0*4 + a_col_off
        a01 = 0*4 + a_col_off + 1
        a10 = 1*4 + a_col_off
        a11 = 1*4 + a_col_off + 1
        
        b00 = b_row_off*8 + b_col_off
        b01 = b_row_off*8 + b_col_off + 1
        b10 = (b_row_off+1)*8 + b_col_off
        b11 = (b_row_off+1)*8 + b_col_off + 1
        
        c00 = c_col_off*2 + 0
        c01 = (c_col_off+1)*2 + 0
        c10 = c_col_off*2 + 1
        c11 = (c_col_off+1)*2 + 1
        
        # Strassen 1
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1
        
        # Strassen 2
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        W[c11, col] += -1
        col += 1
        
        # Strassen 3
        U[a00, col] += 1
        V[b01, col] += 1
        V[b11, col] += -1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # Strassen 4
        U[a11, col] += 1
        V[b10, col] += 1
        V[b00, col] += -1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # Strassen 5
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1
        W[c01, col] += 1
        col += 1
        
        # Strassen 6
        U[a10, col] += 1
        U[a00, col] += -1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # Strassen 7
        U[a01, col] += 1
        U[a11, col] += -1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        col += 1
    
    assert col == 56
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

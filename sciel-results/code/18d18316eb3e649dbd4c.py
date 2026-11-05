"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-54 decomposition for 2x5 * 5x6 matrix multiplication.
    Uses Strassen's algorithm on 2x2x2 sub-blocks to reduce from 60 to 54 rank.
    """
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    U = []
    V = []
    W = []
    
    def add_strassen_block(i0, j0, k0):
        """Add a Strassen 2x2x2 block at given offsets.
        U index: (i0+row)*m + (j0+col)
        V index: (j0+row)*p + (k0+col)
        W index: (k0+col)*n + (i0+row)
        """
        # Strassen 7 multiplications.
        # Each term: list of (U_pos, U_coeff), list of (V_pos, V_coeff), list of (W_pos, W_coeff)
        # Positions: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        
        strassen_terms = [
            # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
            ([((0,0),1),((1,1),1)], [((0,0),1),((0,1),1),((1,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
            # M2 = (A10+A11)B00 -> C10 += 1
            ([((1,0),1),((1,1),1)], [((0,0),1)], [((1,0),1)]),
            # M3 = A00(B10-B11) -> C01 += 1, C11 += 1
            ([((0,0),1)], [((1,0),1),((1,1),-1)], [((0,1),1),((1,1),1)]),
            # M4 = A11(B01-B00) -> C00 += 1, C10 += 1
            ([((1,1),1)], [((0,1),1),((0,0),-1)], [((0,0),1),((1,0),1)]),
            # M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
            ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,0),-1),((0,1),1)]),
            # M6 = (A11-A01)(B00+B01) -> C11 += 1
            ([((1,1),1),((0,1),-1)], [((0,0),1),((0,1),1)], [((1,1),1)]),
            # M7 = (A01-A11)(B01+B11) -> C00 += 1
            ([((0,1),1),((1,1),-1)], [((0,1),1),((1,1),1)], [((0,0),1)]),
        ]
        
        for u_terms, v_terms, w_terms in strassen_terms:
            U_row = [0] * (n * m)
            V_row = [0] * (m * p)
            W_row = [0] * (p * n)
            
            for (r, c), coeff in u_terms:
                U_row[(i0 + r) * m + (j0 + c)] = coeff
            for (r, c), coeff in v_terms:
                V_row[(j0 + r) * p + (k0 + c)] = coeff
            for (r, c), coeff in w_terms:
                W_row[(k0 + c) * n + (i0 + r)] = coeff
            
            U.append(U_row)
            V.append(V_row)
            W.append(W_row)
    
    def add_schoolbook_term(i, j, k):
        U_row = [0] * (n * m)
        V_row = [0] * (m * p)
        W_row = [0] * (p * n)
        U_row[i * m + j] = 1
        V_row[j * p + k] = 1
        W_row[k * n + i] = 1
        U.append(U_row)
        V.append(V_row)
        W.append(W_row)
    
    # Apply Strassen to all 2x2x2 blocks
    for i0 in [0]:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen_block(i0, j0, k0)
    
    # Schoolbook for remaining terms (j=4)
    for i in range(n):
        for k in range(p):
            add_schoolbook_term(i, 4, k)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

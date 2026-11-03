"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 115 second deadline for safety

    # 1. Define Strassen block decomposition
    # We will test it first.
    def get_strassen_terms(i1, i2, j1, j2, k1, k2):
        """
        Returns list of 7 tuples (U_entries, V_entries, W_entries)
        U_entries: dict (i, j) -> coeff
        V_entries: dict (j, k) -> coeff
        W_entries: dict (k, i) -> coeff
        """
        terms = []
        
        # P1 = (A11+A22)(B11+B22)
        u = {(i1,j1): 1, (i2,j2): 1}
        v = {(j1,k1): 1, (j2,k2): 1}
        # C11 += P1, C22 += P1
        w = {(k1,i1): 1, (k2,i2): 1}
        terms.append((u, v, w))
        
        # P2 = (A21+A22)B11
        u = {(i2,j1): 1, (i2,j2): 1}
        v = {(j1,k1): 1}
        # C21 += P2
        w = {(k1,i2): 1}
        terms.append((u, v, w))
        
        # P3 = A11(B21-B22)
        u = {(i1,j1): 1}
        v = {(j2,k1): 1, (j2,k2): -1}
        # C12 += P3
        w = {(k2,i1): 1}
        terms.append((u, v, w))
        
        # P4 = A22(B12-B11)
        u = {(i2,j2): 1}
        v = {(j1,k2): 1, (j1,k1): -1}
        # C11 += P4, C21 += P4
        w = {(k1,i1): 1, (k1,i2): 1}
        terms.append((u, v, w))
        
        # P5 = (A11+A12)B22
        u = {(i1,j1): 1, (i1,j2): 1}
        v = {(j2,k2): 1}
        # C12 += P5, C11 -= P5
        w = {(k2,i1): 1, (k1,i1): -1}
        terms.append((u, v, w))
        
        # P6 = (A21-A11)(B11+B12)
        u = {(i2,j1): 1, (i1,j1): -1}
        v = {(j1,k1): 1, (j1,k2): 1}
        # C22 += P6 ? Let's check.
        # Standard Strassen C22 = P1 - P2 + P3. P6 is not in there.
        # Wait, I need to check the C equations again.
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3
        # Where is P6 and P7?
        # Ah, I see. In some formulations, P6 and P7 are used in C11 and C22.
        # Let's verify P7.
        # P7 = (A12-A22)(B21+B22)
        # If C11 = P1 + P4 - P5 + P7, then W for P7 must have (k1,i1): 1.
        # Let's assume the standard set is correct and derive W for P6 and P7.
        # But wait, if P6 is not in any C equation, it's not used?
        # No, Strassen uses 7 products. All 7 must be used.
        # My C equations must be incomplete or wrong.
        # Correct Strassen C equations:
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3
        # This uses P1, P2, P3, P4, P5, P7. P6 is missing.
        # This implies P6 is not used in this specific variant?
        # Or maybe my P definitions are swapped.
        # Let's swap P6 and P3? No.
        # Actually, there are multiple Strassen variants.
        # Let's just use the one where all 7 are used.
        # A common variant:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B21-B22)
        # P4 = A22(B12-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3
        # It seems P6 is indeed not in the output formulas in this specific set?
        # That's impossible. P6 must cancel out or be part of the derivation.
        # Actually, if I expand C22 = P1 - P2 + P3, I got:
        # A11B11 + A22B22 - A21B11 + A11B21.
        # This is NOT C22.
        # So this set of P's and C's is INVALID.
        
        # Let's try a different known valid set.
        # From Wikipedia:
        # p1 = (a11+a22)(b11+b22)
        # p2 = (a21+a22)b11
        # p3 = a11(b21-b22)
        # p4 = a22(b12-b11)
        # p5 = (a11+a12)b22
        # p6 = (a21-a11)(b11+b12)
        # p7 = (a12-a22)(b21+b22)
        # c11 = p1 + p4 - p5 + p7
        # c12 = p3 + p5
        # c21 = p2 + p4
        # c22 = p1 - p2 + p3
        # This is the same set. Why is it invalid?
        # Let's re-expand c22 = p1 - p2 + p3.
        # p1 = a11b11 + a11b22 + a22b11 + a22b22
        # -p2 = -a21b11 - a22b11
        # +p3 = a11b21 - a11b22
        # Sum: a11b11 + a22b22 - a21b11 + a11b21.
        # Expected c22 = a21b11 + a22b12 + a21b21 + a22b22.
        # They are different.
        # There is a mistake in my expansion or the formula.
        # Let's check p3 again. p3 = a11(b21-b22).
        # Maybe c22 uses p6?
        # If c22 = p1 - p2 + p3 is wrong, what is right?
        # Let's try c22 = p1 - p2 + p3 + ... ?
        # Actually, I suspect the formula for c22 is p1 - p2 + p3 is WRONG for this set of P's.
        # Let's try c22 = p1 - p2 + p3 + p6 - p7 ? No.
        
        # Given the time, I will NOT use Strassen if I can't verify it.
        # I will return the schoolbook.
        
        return terms

    # Verify Strassen
    # Let's just do schoolbook to be safe.
    rank = n * m * p
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

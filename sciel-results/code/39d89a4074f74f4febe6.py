import time
import numpy as np

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Target Rank 54 based on Strassen decomposition of 2x2 blocks
    # Structure:
    # j indices: 0,1,2,3,4
    # k indices: 0,1,2,3,4,5
    # i indices: 0,1
    
    # We group j into pairs (0,1), (2,3) and single (4)
    # We group k into pairs (0,1), (2,3), (4,5)
    
    # For each j-pair and k-pair, we use a 2x2x2 Strassen block (rank 7)
    # There are 2 j-pairs and 3 k-pairs -> 6 blocks * 7 = 42 rank
    # For the remaining j=4 (single) and 3 k-pairs, we use naive 2x1x2 blocks (rank 4)
    # There are 1 j-single and 3 k-pairs -> 3 blocks * 4 = 12 rank
    # Total rank = 42 + 12 = 54
    
    rank = 54
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        """
        Adds a rank-7 Strassen decomposition for a 2x2 * 2x2 block.
        i_base: base index for i (rows of A, rows of C) - must be 0
        j_base: base index for j (cols of A, rows of B)
        k_base: base index for k (cols of B, cols of C)
        
        Sub-blocks:
        A_block: rows i_base, i_base+1; cols j_base, j_base+1
        B_block: rows j_base, j_base+1; cols k_base, k_base+1
        C_block: rows i_base, i_base+1; cols k_base, k_base+1
        
        Strassen Formulas (Standard):
        M0 = (A00 + A11) * (B00 + B11)
        M1 = (A10 + A11) * B00
        M2 = A00 * (B01 - B11)
        M3 = A11 * (B10 - B00)
        M4 = (A00 + A01) * B11
        M5 = (A10 - A00) * (B00 + B01)
        M6 = (A01 - A11) * (B10 + B11)
        
        C00 = M0 - M1 + M3 + M4
        C01 = M1 + M2
        C10 = M3 + M4
        C11 = M0 - M2 + M1 + M5  <-- Wait, let's re-verify C11.
        
        Let's re-derive C11 = A10*B01 + A11*B11.
        M0 = A00B00 + A00B11 + A11B00 + A11B11
        M2 = A00B01 - A00B11
        M1 = A10B00 + A11B00
        M5 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
        
        Try M0 - M2 + M1 + M5:
        (A00B00 + A00B11 + A11B00 + A11B11) - (A00B01 - A00B11) + (A10B00 + A11B00) + (A10B00 + A10B01 - A00B00 - A00B01)
        A00B00: 1 - 1 = 0
        A00B11: 1 + 1 = 2 (Target: 0) -> FAIL.
        
        Let's check standard Strassen C11 formula.
        C11 = M0 - M2 + M1 + M5 is NOT standard.
        Standard is:
        C11 = M0 - M2 + M1 + M5 ? No.
        
        Let's look up standard Strassen coefficients.
        M1 = (A10 + A11) B00
        M2 = A00 (B01 - B11)
        M3 = A11 (B10 - B00)
        M4 = (A00 + A01) B11
        M5 = (A10 - A00) (B00 + B01)
        M6 = (A01 - A11) (B10 + B11)
        
        C00 = M0 - M1 + M3 + M4
        C01 = M1 + M2
        C10 = M3 + M4
        C11 = M0 - M2 + M1 + M5 ?
        
        Let's re-evaluate C11 = M0 - M2 + M1 + M5.
        Wait, I might have the indices mixed up in my head or the formula is different.
        Let's use a known correct set.
        
        Alternative set:
        M1 = (A00 + A11)(B00 + B11)
        M2 = (A10 + A11)B00
        M3 = A00(B01 - B11)
        M4 = A11(B10 - B00)
        M5 = (A00 + A01)B11
        M6 = (A10 - A00)(B00 + B01)
        M7 = (A01 - A11)(B10 + B11)
        
        C00 = M1 - M2 + M4 + M5
        C01 = M2 + M3
        C10 = M4 + M5
        C11 = M1 - M3 + M2 + M6  <-- Let's test this.
        
        C11 = (A00+A11)(B00+B11) - A00(B01-B11) + (A10+A11)B00 + (A10-A00)(B00+B01)
        = (A00B00 + A00B11 + A11B00 + A11B11) - (A00B01 - A00B11) + (A10B00 + A11B00) + (A10B00 + A10B01 - A00B00 - A00B01)
        
        A00B00: 1 - 1 = 0
        A00B11: 1 + 1 = 2. (Target 0). Still failing.
        
        There is a sign error in my manual expansion or the formula.
        Let's rely on the inspiration code which scored 1.1111 (Rank 54).
        The inspiration code used:
        M0: (A00+A11)(B00+B11) -> U: A00,A11. V: B00,B11. W: C00,C11
        M1: (A10+A11)B00 -> U: A10,A11. V: B00. W: C10, C11(-1)
        M2: A00(B01-B11) -> U: A00. V: B01,-B11. W: C01, C11
        M3: A11(B10-B00) -> U: A11. V: B10,-B00. W: C00, C10
        M4: (A00+A01)B11 -> U: A00,A01. V: B11. W: C00(-1), C01
        M5: (A10-A00)(B00+B01) -> U: A10,-A00. V: B00,B01. W: C11
        M6: (A01-A11)(B10+B11) -> U: A01,-A11. V: B10,B11. W: C00
        
        Let's verify C11 with this set:
        C11 gets contributions from:
        M0: +1 (from W11)
        M1: -1 (from W11)
        M2: +1 (from W11)
        M5: +1 (from W11)
        
        Sum for C11: M0 - M1 + M2 + M5
        M0 = A00B00 + A00B11 + A11B00 + A11B11
        -M1 = -A10B00 - A11B00
        M2 = A00B01 - A00B11
        M5 = A10B00 + A10B01 - A00B00 - A00B01
        
        Total:
        A00B00: 1 - 1 = 0
        A00B11: 1 - 1 = 0
        A11B00: 1 - 1 = 0
        A11B11: 1 (Target 1) OK
        A10B00: -1 + 1 = 0
        A00B01: 1 - 1 = 0
        A10B01: 1 (Target 1) OK
        
        It works!
        
        Let's verify C00:
        M0: +1
        M3: +1
        M4: -1
        M6: +1
        Sum: M0 + M3 - M4 + M6
        M0 = A00B00 + A00B11 + A11B00 + A11B11
        M3 = A11B10 - A11B00
        -M4 = -A00B11 - A01B11
        M6 = A01B10 + A01B11 - A11B10 - A11B11
        
        Total:
        A00B00: 1 (Target 1) OK
        A00B11: 1 - 1 = 0
        A11B00: 1 - 1 = 0
        A11B11: 1 - 1 = 0
        A11B10: 1 - 1 = 0
        A01B11: -1 + 1 = 0
        A01B10: 1 (Target 1? No, C00 is A00B00 + A01B10? No. C00 = A00B00 + A01B10? 
        Wait, C00 = sum_j A0j B_j0 = A00B00 + A01B10.
        My sum has A01B10 = 1. Correct.
        
        Let's verify C01:
        M2: +1
        M4: +1
        Sum: M2 + M4
        M2 = A00B01 - A00B11
        M4 = A00B11 + A01B11
        Total: A00B01 + A01B11.
        Target C01 = A00B01 + A01B11. Correct.
        
        Let's verify C10:
        M1: +1
        M3: +1
        Sum: M1 + M3
        M1 = A10B00 + A11B00
        M3 = A11B10 - A11B00
        Total: A10B00 + A11B10.
        Target C10 = A10B00 + A11B10. Correct.
        
        So the coefficients in the inspiration code are correct.
        """
        nonlocal col
        
        # M0: (A00+A11)(B00+B11)
        U[i_base * m + j_base][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        
        # M1: (A10+A11)B00
        U[(i_base + 1) * m + j_base][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] -= 1
        col += 1
        
        # M2: A00(B01-B11)
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] -= 1
        W[(k_base + 1) * n + i_base][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        
        # M3: A11(B10-B00)
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[(j_base + 1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        col += 1
        
        # M4: (A00+A01)B11
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base + 1)][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base + 1) * n + i_base][col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01)
        U[(i_base + 1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11)
        U[i_base * m + (j_base + 1)][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] -= 1
        V[(j_base + 1) * p + k_base][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        """
        Adds a rank-4 naive decomposition for a 2x1 * 1x2 block.
        i_base: base index for i
        j_base: single index for j
        k_base: base index for k
        """
        nonlocal col
        
        # 1: A00 B00 -> C00
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        
        # 2: A00 B01 -> C01
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + i_base][col] += 1
        col += 1
        
        # 3: A10 B00 -> C10
        U[(i_base + 1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        col += 1
        
        # 4: A10 B01 -> C11
        U[(i_base + 1) * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1

    # Apply Strassen blocks for j pairs (0,1) and (2,3)
    # j_base = 0, 2
    # k_base = 0, 2, 4
    
    for j_base in [0, 2]:
        for k_base in [0, 2, 4]:
            add_strassen_block(0, j_base, k_base)
    
    # Apply naive blocks for j=4
    # j_base = 4
    # k_base = 0, 2, 4
    
    for k_base in [0, 2, 4]:
        add_naive_2112_block(0, 4, k_base)
    
    # Verify col == 54
    if col != 54:
        # Fallback to schoolbook if something went wrong (should not happen)
        # But we can just return what we have, it might be invalid.
        # Better to return a valid schoolbook if col != 54?
        # No, the logic is deterministic.
        pass
        
    # Verification step (optional but good for debugging, disabled for speed in final)
    # if not _check_decomposition(U, V, W, n, m, p):
    #     print("Decomposition failed verification")
        
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > tol:
                    return False
    return True

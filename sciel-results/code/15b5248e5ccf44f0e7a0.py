"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS

    # Strassen 2x2: rank 7
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    #
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22) B11
    # P3 = A11 (B12 - B22)
    # P4 = A22 (B21 - B11)
    # P5 = (A11 + A12) B22
    # P6 = (A21 - A11) (B11 + B12)
    # P7 = (A12 - A22) (B21 + B22)
    
    # For each product P_r, we define:
    # U entries: coefficients for A entries (i, j)
    # V entries: coefficients for B entries (j, k)
    # W entries: coefficients for C entries (k, i)
    
    # Strassen products defined as (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict mapping (i,j) -> coeff
    # V_coeffs: dict mapping (j,k) -> coeff
    # W_coeffs: dict mapping (k,i) -> coeff
    
    strassen_products = [
        # P1 = (A11 + A22)(B11 + B22)
        # contributes to C11 with +1, C22 with +1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2 = (A21 + A22) B11
        # contributes to C21 with +1, C22 with -1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        # P3 = A11 (B12 - B22)
        # contributes to C12 with +1, C22 with +1
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        # P4 = A22 (B21 - B11)
        # contributes to C11 with +1, C21 with +1
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # P5 = (A11 + A12) B22
        # contributes to C11 with -1, C12 with +1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        # P6 = (A21 - A11) (B11 + B12)
        # contributes to C22 with +1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7 = (A12 - A22) (B21 + B22)
        # contributes to C11 with +1
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    rank = 7 * 8  # 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(8):
        # A is 2x4, B is 4x8
        # Each block multiplies 2x2 (from A) by 2x2 (from B) to get 2x2 (in C)
        # A columns: 2*block to 2*block+1
        # B columns: 4*block to 4*block+1... wait, B is 4x8
        # A is 2x4, so A has columns 0-3. B is 4x8, so B has rows 0-3, cols 0-7.
        # C is 2x8.
        # For block b (0-7):
        #   A rows 0-1, cols 2*(b//2) to 2*(b//2)+1
        #   B rows 2*(b//2) to 2*(b//2)+1, cols 2*(b%2) to 2*(b%2)+1... 
        #   
        # Actually, let me think more carefully.
        # A is 2x4. B is 4x8. C is 2x8.
        # We can split this into 8 blocks of 2x2 by 2x2.
        # For output column pair (2*qc, 2*qc+1) of C, and input row pair (2*ar, 2*ar+1) of A... 
        # 
        # Let's index: C[i, k] = sum_j A[i,j] B[j,k]
        # i in {0,1}, k in {0..7}, j in {0..3}
        # 
        # For each output column block (k0, k0+1) where k0 = 2*qc, qc in {0,1,2,3}:
        #   And each intermediate row block (j0, j0+1) where j0 = 2*ar, ar in {0,1}:
        #     This contributes to C[i, k0..k0+1] for i in {0,1}
        #     Using A[i, j0..j0+1] and B[j0..j0+1, k0..k0+1]
        #
        # So there are 4 * 2 = 8 blocks.
        # Block index: b = qc * 2 + ar
        #   qc = b // 2, ar = b % 2
        #   A columns: 2*ar, 2*ar+1
        #   B rows: 2*ar, 2*ar+1
        #   B columns: 2*qc, 2*qc+1
        #   C columns: 2*qc, 2*qc+1
        #   C rows: 0, 1
        
        qc = block // 2
        ar = block % 2
        a_col0 = 2 * ar
        b_row0 = 2 * ar
        b_col0 = 2 * qc
        c_col0 = 2 * qc
        
        for u_coeff, v_coeff, w_coeff in strassen_products:
            # Fill U column col
            for (li, lj), coeff in u_coeff.items():
                gi = li
                gj = a_col0 + lj
                U[gi * m + gj][col] += coeff
            # Fill V column col
            for (lj, lk), coeff in v_coeff.items():
                gj = b_row0 + lj
                gk = b_col0 + lk
                V[gj * p + gk][col] += coeff
            # Fill W column col
            for (lk, li), coeff in w_coeff.items():
                gk = b_col0 + lk
                gi = li
                W[gk * n + gi][col] += coeff
            col += 1
        

    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

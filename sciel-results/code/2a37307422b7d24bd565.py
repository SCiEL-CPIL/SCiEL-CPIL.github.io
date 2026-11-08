"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
TIME_LIMIT = 120.0

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    # Schoolbook decomposition of rank n*m*p = 32
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Embedded Strassen decomposition of rank 28
    # Partition A (2x4) as [A1, A2] where A1, A2 are 2x2
    # Partition B (4x4) as [[B11, B12], [B21, B22]] where each is 2x2
    # C = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    # Each 2x2x2 multiplication uses Strassen (rank 7)
    # Total rank = 4 * 7 = 28
    
    # Strassen's 7 intermediate products for 2x2 matrices:
    # Given X (2x2) and Y (2x2), compute X*Y using:
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y10-Y11)
    # M4 = X11*(Y01-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y10)
    # M7 = (X01-X11)*(Y01+Y11)
    # Result:
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # For each block multiplication (4 total), we have 7 Strassen terms
    # Block 0: A1 * B11 (contributes to C[:, 0:2])
    # Block 1: A2 * B21 (contributes to C[:, 0:2])
    # Block 2: A1 * B12 (contributes to C[:, 2:4])
    # Block 3: A2 * B22 (contributes to C[:, 2:4])
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # shape (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # shape (8, 28)
    
    # Strassen coefficients for the 7 products
    # For product X*Y where X, Y are 2x2:
    # X indices: X00=(0,0), X01=(0,1), X10=(1,0), X11=(1,1)
    # Y indices: Y00=(0,0), Y01=(0,1), Y10=(1,0), Y11=(1,1)
    # 
    # M1 = (X00+X11)*(Y00+Y11): U factors: X00:+1, X11:+1; V factors: Y00:+1, Y11:+1
    # M2 = (X10+X11)*Y00: U factors: X10:+1, X11:+1; V factors: Y00:+1
    # M3 = X00*(Y10-Y11): U factors: X00:+1; V factors: Y10:+1, Y11:-1
    # M4 = X11*(Y01-Y00): U factors: X11:+1; V factors: Y01:+1, Y00:-1
    # M5 = (X00+X01)*Y11: U factors: X00:+1, X01:+1; V factors: Y11:+1
    # M6 = (X10-X00)*(Y00+Y10): U factors: X10:+1, X00:-1; V factors: Y00:+1, Y10:+1
    # M7 = (X01-X11)*(Y01+Y11): U factors: X01:+1, X11:-1; V factors: Y01:+1, Y11:+1
    #
    # Output reconstruction:
    # Z00 = M1 + M4 - M5 + M7: W factors for each M
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Strassen term definitions: list of (u_factors, v_factors, w_factors)
    # u_factors: dict mapping (row, col) -> coeff for X
    # v_factors: dict mapping (row, col) -> coeff for Y
    # w_factors: dict mapping (row, col) -> coeff for Z
    
    strassen_terms = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        # M3
        ([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,1,1)]),
        # M4
        ([(1,1,1)], [(0,1,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)], [(1,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)], [(0,0,1)]),
    ]
    
    # Block definitions: (X_offset, Y_offset, Z_offset)
    # X is A (2x4), Y is B (4x4), Z is C (2x4)
    # A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    # C[:, 0:2] = A1*B11 + A2*B21, C[:, 2:4] = A1*B12 + A2*B22
    
    blocks = [
        # (A_col_offset, B_row_offset, B_col_offset, C_col_offset)
        (0, 0, 0, 0),  # A1 * B11 -> C[:, 0:2]
        (2, 2, 0, 0),  # A2 * B21 -> C[:, 0:2]
        (0, 0, 2, 2),  # A1 * B12 -> C[:, 2:4]
        (2, 2, 2, 2),  # A2 * B22 -> C[:, 2:4]
    ]
    
    col = 0
    for (a_col_off, b_row_off, b_col_off, c_col_off) in blocks:
        for (u_fac, v_fac, w_fac) in strassen_terms:
            # U factors: A indices
            for (r, c, coeff) in u_fac:
                U[r * m + (a_col_off + c)][col] = coeff
            
            # V factors: B indices
            for (r, c, coeff) in v_fac:
                V[(b_row_off + r) * p + (b_col_off + c)][col] = coeff
            
            # W factors: C indices
            for (r, c, coeff) in w_fac:
                W[(c_col_off + c) * n + r][col] = coeff
            
            col += 1
    
    # Verify the decomposition
    try:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Build tensor T
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Check: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        T_recon = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        
        if np.allclose(T, T_recon, atol=1e-10):
            return {"U": U, "V": V, "W": W}
        else:
            # Return schoolbook if Strassen verification fails
            return {"U": U_school, "V": V_school, "W": W_school}
    except Exception:
        return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END

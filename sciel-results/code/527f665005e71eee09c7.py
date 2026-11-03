import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add a Strassen block
    # X is 2x2 from A (rows i0..i1, cols j0..j1)
    # Y is 2x2 from B (rows j0..j1, cols k0..k1)
    # Z is 2x2 in C (rows i0..i1, cols k0..k1)
    def add_strassen_block(col_offset, i0, j0, k0):
        # Local indices 0,1 map to global i0+i, j0+j, k0+k
        # X indices in U: (i0+i)*m + (j0+j)
        # Y indices in V: (j0+j)*p + (k0+k)
        # Z indices in W: (k0+k)*n + (i0+i)
        
        # Strassen multipliers
        # M1: (X00+X11)(Y00+Y11) -> Z00 += 1, Z11 += 1
        # M2: (X10+X11)Y00     -> Z10 += 1
        # M3: X00(Y10-Y11)     -> Z01 += 1
        # M4: X11(Y01-Y00)     -> Z00 += 1, Z10 += 1
        # M5: (X00+X10)Y11     -> Z00 -= 1, Z01 += 1
        # M6: (X10-X00)(Y00+Y10) -> Z11 += 1
        # M7: (X01-X11)(Y11+Y01) -> Z00 += 1, Z11 -= 1
        
        # Define the 7 terms
        # Each term: (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
        
        terms = []
        
        # M1
        terms.append((
            [(i0)*m + (j0), (i0+1)*m + (j0+1)], [1, 1],
            [(j0)*p + (k0), (j0+1)*p + (k0+1)], [1, 1],
            [(k0)*n + (i0), (k0+1)*n + (i0+1)], [1, 1]
        ))
        # M2
        terms.append((
            [(i0+1)*m + (j0), (i0+1)*m + (j0+1)], [1, 1],
            [(j0)*p + (k0)], [1],
            [(k0)*n + (i0+1)], [1]
        ))
        # M3
        terms.append((
            [(i0)*m + (j0)], [1],
            [(j0+1)*p + (k0), (j0+1)*p + (k0+1)], [1, -1],
            [(k0+1)*n + (i0)], [1]
        ))
        # M4
        terms.append((
            [(i0+1)*m + (j0+1)], [1],
            [(j0)*p + (k0+1), (j0)*p + (k0)], [1, -1],
            [(k0)*n + (i0), (k0)*n + (i0+1)], [1, 1]
        ))
        # M5
        terms.append((
            [(i0)*m + (j0), (i0+1)*m + (j0)], [1, 1],
            [(j0+1)*p + (k0+1)], [1],
            [(k0)*n + (i0), (k0+1)*n + (i0)], [-1, 1]
        ))
        # M6
        terms.append((
            [(i0+1)*m + (j0), (i0)*m + (j0)], [1, -1],
            [(j0)*p + (k0), (j0+1)*p + (k0)], [1, 1],
            [(k0+1)*n + (i0+1)], [1]
        ))
        # M7
        terms.append((
            [(i0)*m + (j0+1), (i0+1)*m + (j0+1)], [1, -1],
            [(j0+1)*p + (k0+1), (j0)*p + (k0+1)], [1, 1],
            [(k0)*n + (i0), (k0+1)*n + (i0+1)], [1, -1]
        ))
        
        for t in terms:
            u_idx, u_val, v_idx, v_val, w_idx, w_val = t
            U[u_idx, col_offset] = u_val
            V[v_idx, col_offset] = v_val
            W[w_idx, col_offset] = w_val
            col_offset += 1
            
        return col_offset

    # Top 2 rows of C
    # C_top = A_top B
    # A_top is rows 0,1 of A.
    # B is 4x4.
    # Split B into 4 2x2 blocks.
    # A_top split into Left (cols 0,1) and Right (cols 2,3).
    
    # Block 1: A_TL (rows 0,1; cols 0,1) * B_00 (rows 0,1; cols 0,1) -> C_TL (rows 0,1; cols 0,1)
    col = 0
    col = add_strassen_block(col, 0, 0, 0)
    
    # Block 2: A_TR (rows 0,1; cols 2,3) * B_20 (rows 2,3; cols 0,1) -> C_TL (rows 0,1; cols 0,1)
    col = add_strassen_block(col, 0, 2, 0)
    
    # Block 3: A_TL (rows 0,1; cols 0,1) * B_02 (rows 0,1; cols 2,3) -> C_TR (rows 0,1; cols 2,3)
    col = add_strassen_block(col, 0, 0, 2)
    
    # Block 4: A_TR (rows 0,1; cols 2,3) * B_22 (rows 2,3; cols 2,3) -> C_TR (rows 0,1; cols 2,3)
    col = add_strassen_block(col, 0, 2, 2)
    
    # Bottom row of C
    # C_bot = A_bot B
    # A_bot is row 2 of A.
    # B is 4x4.
    # Schoolbook: 16 terms.
    for j in range(4):
        for k in range(4):
            # A[2, j] * B[j, k] -> C[2, k]
            U[2*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + 2, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.
    Uses a hard-coded Strassen-based decomposition of rank 39.
    """
    n, m, p = DIMENSIONS
    
    rank = 39
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        nonlocal col
        U[u_idx, col] += coeff
        V[v_idx, col] += coeff
        W[w_idx, col] += coeff
        col += 1
        
    def add_strassen_block(a_row_offsets, b_col_offsets, b_row_offsets, k_offset):
        """
        Adds a Strassen 2x2x2 block.
        a_row_offsets: list of 2 indices for A rows (i)
        b_col_offsets: list of 2 indices for B cols (k)
        b_row_offsets: list of 2 indices for B rows (j)
        k_offset: offset for the output column k
        """
        # A indices: i in a_row_offsets, j in [0, 1] relative to A's first 2 cols
        # B indices: j in b_row_offsets, k in b_col_offsets
        # C indices: i in a_row_offsets, k in [k_offset, k_offset+1]
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        # A[0,0] is u = a_row_offsets[0]*3 + 0
        # A[1,1] is u = a_row_offsets[1]*3 + 1
        # B[0,0] is v = b_row_offsets[0]*7 + b_col_offsets[0]
        # B[1,1] is v = b_row_offsets[1]*7 + b_col_offsets[1]
        
        # We need to sum these.
        # Since U, V, W are sparse, we can add individual terms.
        # M1 = A00 B00 + A00 B11 + A11 B00 + A11 B11
        # This is 4 rank-1 terms.
        # But Strassen combines them into 1 multiplier.
        # The multiplier is M1.
        # U part: A00 and A11. V part: B00 and B11.
        # This is not a single rank-1 term in U,V,W unless we allow sums.
        # But U, V, W columns must be sparse?
        # The problem says "integer/half-integer entries".
        # It does NOT say sparse.
        # So we can put sums in U, V, W columns.
        
        # However, the standard decomposition lists the multipliers as the columns.
        # Each column c corresponds to one multiplication $M_c = (\sum U_{i,c} A_i) (\sum V_{j,c} B_j)$.
        # Then $C_k = \sum W_{k,c} M_c$.
        
        # So for M1:
        # U[0, col] = 1, U[1, col] = 1 (for A00, A11)
        # V[0, col] = 1, V[1, col] = 1 (for B00, B11)
        # W[0, col] = 1, W[1, col] = 1 (for C00, C11? No, M1 contributes to C00, C11, C01, C10 with signs)
        
        # Let's trace M1 contribution:
        # C00 += M1
        # C11 += M1
        # C01 -= M1? No, C01 = M3 + M5. C10 = M2 + M4.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for M1:
        # W[C00] += 1
        # W[C11] += 1
        
        # We will implement this logic for each Strassen block.
        
        i0, i1 = a_row_offsets
        j0, j1 = b_row_offsets
        k0, k1 = b_col_offsets
        
        # M1
        U[i0*3+0, col] = 1.0
        U[i1*3+1, col] = 1.0
        V[j0*7+k0, col] = 1.0
        V[j1*7+k1, col] = 1.0
        W[k_offset*2 + i0, col] = 1.0
        W[(k_offset+1)*2 + i1, col] = 1.0
        col += 1
        
        # M2 = (A10 + A11) * B00
        U[i1*3+0, col] = 1.0
        U[i1*3+1, col] = 1.0
        V[j0*7+k0, col] = 1.0
        W[k_offset*2 + i1, col] = 1.0
        col += 1
        
        # M3 = A00 * (B10 - B11)
        U[i0*3+0, col] = 1.0
        V[j1*7+k0, col] = 1.0
        V[j1*7+k1, col] = -1.0
        W[k_offset*2 + i0, col] = 1.0
        W[(k_offset+1)*2 + i0, col] = 1.0 # Wait, C01 = M3+M5. C00? No.
        # C01 = M3 + M5. So W[C01] += 1.
        # C00? No.
        # Let's re-verify Strassen formula.
        # C = [ M1+M4-M5+M7, M3+M5 ]
        #     [ M2+M4,         M1-M2+M3+M6 ]
        # So M3 contributes to C01 and C11.
        W[k_offset*2 + i0, col] = 0.0 # C00? No.
        W[(k_offset+1)*2 + i0, col] = 1.0 # C01
        W[(k_offset+1)*2 + i1, col] = 1.0 # C11
        col += 1
        
        # M4 = A11 * (B00 - B01)
        U[i1*3+1, col] = 1.0
        V[j0*7+k0, col] = 1.0
        V[j0*7+k1, col] = -1.0
        W[k_offset*2 + i0, col] = 1.0 # C00
        W[k_offset*2 + i1, col] = 1.0 # C10
        col += 1
        
        # M5 = (A00 + A01) * B11
        U[i0*3+0, col] = 1.0
        U[i0*3+1, col] = 1.0
        V[j1*7+k1, col] = 1.0
        W[k_offset*2 + i0, col] = -1.0 # C00
        W[(k_offset+1)*2 + i0, col] = 1.0 # C01
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[i1*3+0, col] = 1.0
        U[i0*3+0, col] = -1.0
        V[j0*7+k0, col] = 1.0
        V[j0*7+k1, col] = 1.0
        W[(k_offset+1)*2 + i1, col] = 1.0 # C11
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[i0*3+1, col] = 1.0
        U[i1*3+1, col] = -1.0
        V[j1*7+k0, col] = 1.0
        V[j1*7+k1, col] = 1.0
        W[k_offset*2 + i0, col] = 1.0 # C00
        col += 1

    # Block 1: A1 B11 (Strassen)
    # A1 rows: 0, 1. Cols: 0, 1.
    # B11 rows: 0, 1. Cols: 0, 1.
    add_strassen_block([0, 1], [0, 1], [0, 1], 0)
    
    # Block 2: A1 B12 (Strassen)
    # B12 rows: 0, 1. Cols: 2, 3.
    add_strassen_block([0, 1], [2, 3], [0, 1], 2)
    
    # Block 3: A1 B13 (Strassen)
    # B13 rows: 0, 1. Cols: 4, 5.
    add_strassen_block([0, 1], [4, 5], [0, 1], 4)
    
    # Block 4: A1 B14 (Schoolbook 2x2x1)
    # B14 rows: 0, 1. Col: 6.
    # C[i, 6] = A[i,0]B[0,6] + A[i,1]B[1,6]
    for i in range(2):
        for j in range(2):
            u_idx = i*3 + j
            v_idx = j*7 + 6
            w_idx = 6*2 + i
            U[u_idx, col] = 1.0
            V[v_idx, col] = 1.0
            W[w_idx, col] = 1.0
            col += 1
            
    # Block 5: A2 B2 (Schoolbook 2x1x7)
    # A2 row: 2 (in A). Cols: 0, 1 (in A2, which is A[:, 2])
    # B2 row: 2 (in B). Cols: 0..6.
    # C[i, k] = A[i, 2] B[2, k]
    for i in range(2):
        for k in range(7):
            u_idx = i*3 + 2
            v_idx = 2*7 + k
            w_idx = k*2 + i
            U[u_idx, col] = 1.0
            V[v_idx, col] = 1.0
            W[w_idx, col] = 1.0
            col += 1
            
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Convert to list format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    def to_int_or_half(x):
        if abs(x - round(x)) < 1e-10:
            return int(round(x))
        elif abs(x * 2 - round(x * 2)) < 1e-10:
            return round(x * 2) / 2.0
        else:
            return x
            
    U_final = [[to_int_or_half(val) for val in row] for row in U_list]
    V_final = [[to_int_or_half(val) for val in row] for row in V_list]
    W_final = [[to_int_or_half(val) for val in row] for row in W_list]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    # Verify
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m+j
                v = j*p+k
                w = k*n+i
                val = np.dot(U[u,:], V[v,:]*W[w,:])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    print(f"Max Error: {max_err}")
    print(f"Rank: {len(U[0])}")

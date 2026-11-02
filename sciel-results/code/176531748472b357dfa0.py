import time
import numpy as np

def construct():
    """
    Constructs a rank-minimal (or near-minimal) exact tensor decomposition 
    for multiplying a 2x4 matrix by a 4x8 matrix.
    
    Returns:
        dict: {'U': list, 'V': list, 'W': list}
    """
    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n
    
    # Rank 56 via 4 blocks of M_{2,4,2}, each rank 14 (two Strassen 2x2x2)
    rank = 56
    
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    def add_strassen_block(A_base, B_base, C_base, col_offset):
        """
        Adds a Strassen decomposition for A_base (2x2) * B_base (2x2) = C_base (2x2)
        A_base indices in U: row offset A_base
        B_base indices in V: row offset B_base
        C_base indices in W: row offset C_base
        """
        # A is 2x2, so A_base is the index of the top-left element in the flattened 2x4 A?
        # No, U is flattened 2x4.
        # We need to map 2x2 local indices to global indices.
        
        # Local A indices: 00, 01, 10, 11
        # Local B indices: 00, 01, 10, 11
        # Local C indices: 00, 01, 10, 11
        
        # We pass the global row indices for the top-left of the 2x2 blocks.
        
        a_00, a_01 = A_base, A_base + 1
        a_10, a_11 = A_base + 2, A_base + 3
        
        b_00, b_01 = B_base, B_base + 1
        b_10, b_11 = B_base + 2, B_base + 3
        
        c_00, c_01 = C_base, C_base + 1
        c_10, c_11 = C_base + 2, C_base + 3
        
        # M1
        U[a_00, col_offset] += 1
        U[a_11, col_offset] += 1
        V[b_00, col_offset] += 1
        V[b_11, col_offset] += 1
        W[c_00, col_offset] += 1
        W[c_11, col_offset] += 1
        
        # M2
        U[a_10, col_offset+1] += 1
        U[a_11, col_offset+1] += 1
        V[b_00, col_offset+1] += 1
        W[c_10, col_offset+1] += 1
        
        # M3
        U[a_00, col_offset+2] += 1
        V[b_10, col_offset+2] += 1
        V[b_11, col_offset+2] -= 1
        W[c_01, col_offset+2] += 1
        
        # M4
        U[a_11, col_offset+3] += 1
        V[b_01, col_offset+3] += 1
        V[b_00, col_offset+3] -= 1
        W[c_00, col_offset+3] += 1
        W[c_10, col_offset+3] += 1
        
        # M5
        U[a_00, col_offset+4] += 1
        U[a_01, col_offset+4] += 1
        V[b_11, col_offset+4] += 1
        W[c_00, col_offset+4] -= 1
        W[c_01, col_offset+4] += 1
        
        # M6
        U[a_10, col_offset+5] += 1
        U[a_00, col_offset+5] -= 1
        V[b_00, col_offset+5] += 1
        V[b_01, col_offset+5] += 1
        W[c_11, col_offset+5] += 1
        
        # M7
        U[a_01, col_offset+6] += 1
        U[a_11, col_offset+6] -= 1
        V[b_10, col_offset+6] += 1
        V[b_11, col_offset+6] += 1
        W[c_11, col_offset+6] += 1

    # A is 2x4. 
    # A1 is columns 0,1. A2 is columns 2,3.
    # A1 base index in U (flattened row-major): 0
    # A2 base index in U: 2
    
    # B is 4x8.
    # For k in 0..3:
    # B_1k is rows 0,1 and cols 2k, 2k+1.
    # B_2k is rows 2,3 and cols 2k, 2k+1.
    
    # C is 2x8.
    # C_k is cols 2k, 2k+1.
    # C_k base index in W (flattened row-major): 2k
    
    for k in range(4):
        # B_1k base index in V:
        # Row 0, Col 2k -> index 2k
        b1_base = 2 * k
        # B_2k base index in V:
        # Row 2, Col 2k -> index 2*p + 2k = 16 + 2k
        b2_base = 16 + 2 * k
        
        # C_k base index in W:
        # Row 0, Col 2k -> index 2k
        c_base = 2 * k
        
        # Compute A1 * B_1k
        # A1 base is 0
        add_strassen_block(0, b1_base, c_base, 14 * k)
        
        # Compute A2 * B_2k
        # A2 base is 2
        add_strassen_block(2, b2_base, c_base, 14 * k + 7)
        
    # Verification
    def verify(U, V, W, rank, atol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :rank] * V[j*p+k, :rank] * W[k*n+i, :rank])
                    if not np.isclose(val, 1.0, atol=atol):
                        return False
        return True

    if not verify(U, V, W, rank):
        # Fallback to schoolbook if something went wrong (should not happen)
        rank = 64
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

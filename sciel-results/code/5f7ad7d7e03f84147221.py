"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # We use a block decomposition strategy.
    # Split intermediate dimension m=4 into two groups of 2.
    # C = A1 @ B1 + A2 @ B2
    # where A1, A2 are 2x2, B1, B2 are 2x5.
    # For each 2x2 @ 2x5 block, we split the output columns into 2, 2, 1.
    # 2x2 @ 2x2 uses Strassen (rank 7).
    # 2x2 @ 2x1 uses schoolbook (rank 4).
    # Total per block: 7 + 7 + 4 = 18.
    # Total rank: 18 + 18 = 36.
    
    rank = 36
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen_2x2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """Add Strassen decomposition for 2x2 @ 2x2 -> 2x2.
        A rows a_r0, a_r0+1; cols a_c0, a_c0+1
        B rows b_r0, b_r0+1; cols b_c0, b_c0+1
        C rows c_r0, c_r0+1; cols c_c0, c_c0+1
        U index: i*m + j
        V index: j*p + k
        W index: k*n + i
        """
        nonlocal col
        u00 = a_r0 * m + a_c0
        u01 = a_r0 * m + a_c0 + 1
        u10 = (a_r0 + 1) * m + a_c0
        u11 = (a_r0 + 1) * m + a_c0 + 1
        
        v00 = b_r0 * p + b_c0
        v01 = b_r0 * p + b_c0 + 1
        v10 = (b_r0 + 1) * p + b_c0
        v11 = (b_r0 + 1) * p + b_c0 + 1
        
        w00 = c_c0 * n + c_r0
        w01 = (c_c0 + 1) * n + c_r0
        w10 = c_c0 * n + c_r0 + 1
        w11 = (c_c0 + 1) * n + c_r0 + 1
        
        # M1 = (a00+a11)(b00+b11) -> C00 += M1, C11 += M1
        U[u00, col] += 1; U[u11, col] += 1
        V[v00, col] += 1; V[v11, col] += 1
        W[w00, col] += 1; W[w11, col] += 1
        col += 1
        
        # M2 = (a10+a11)b00 -> C10 += M2, C11 -= M2
        U[u10, col] += 1; U[u11, col] += 1
        V[v00, col] += 1
        W[w10, col] += 1; W[w11, col] -= 1
        col += 1
        
        # M3 = a00(b01-b11) -> C01 += M3, C11 += M3
        U[u00, col] += 1
        V[v01, col] += 1; V[v11, col] -= 1
        W[w01, col] += 1; W[w11, col] += 1
        col += 1
        
        # M4 = a11(b10-b00) -> C00 += M4, C10 += M4
        U[u11, col] += 1
        V[v10, col] += 1; V[v00, col] -= 1
        W[w00, col] += 1; W[w10, col] += 1
        col += 1
        
        # M5 = (a00+a01)b11 -> C00 -= M5, C01 += M5
        U[u00, col] += 1; U[u01, col] += 1
        V[v11, col] += 1
        W[w00, col] -= 1; W[w01, col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01) -> C11 += M6
        U[u10, col] += 1; U[u00, col] -= 1
        V[v00, col] += 1; V[v01, col] += 1
        W[w11, col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11) -> C00 += M7
        U[u01, col] += 1; U[u11, col] -= 1
        V[v10, col] += 1; V[v11, col] += 1
        W[w00, col] += 1
        col += 1

    def add_schoolbook_2x2_2x1(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """Add schoolbook for 2x2 @ 2x1 -> 2x1.
        A rows a_r0, a_r0+1; cols a_c0, a_c0+1
        B rows b_r0, b_r0+1; col b_c0
        C rows c_r0, c_r0+1; col c_c0
        """
        nonlocal col
        u00 = a_r0 * m + a_c0
        u01 = a_r0 * m + a_c0 + 1
        u10 = (a_r0 + 1) * m + a_c0
        u11 = (a_r0 + 1) * m + a_c0 + 1
        
        v0 = b_r0 * p + b_c0
        v1 = (b_r0 + 1) * p + b_c0
        
        w0 = c_c0 * n + c_r0
        w1 = c_c0 * n + c_r0 + 1
        
        # C0 = A00*B0 + A01*B1
        U[u00, col] = 1; V[v0, col] = 1; W[w0, col] = 1
        col += 1
        U[u01, col] = 1; V[v1, col] = 1; W[w0, col] = 1
        col += 1
        
        # C1 = A10*B0 + A11*B1
        U[u10, col] = 1; V[v0, col] = 1; W[w1, col] = 1
        col += 1
        U[u11, col] = 1; V[v1, col] = 1; W[w1, col] = 1
        col += 1

    # Block 1: A[:, 0:2] @ B[0:2, :]
    # A rows 0,1 cols 0,1; B rows 0,1 cols 0,1,2,3,4
    # Split B into cols 0,1 (2x2), cols 2,3 (2x2), col 4 (2x1)
    add_strassen_2x2(0, 0, 0, 0, 0, 0)      # A[:,0:2] @ B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_2x2(0, 0, 0, 2, 0, 2)      # A[:,0:2] @ B[0:2, 2:4] -> C[0:2, 2:4]
    add_schoolbook_2x2_2x1(0, 0, 0, 4, 0, 4) # A[:,0:2] @ B[0:2, 4:5] -> C[0:2, 4:5]
    
    # Block 2: A[:, 2:4] @ B[2:4, :]
    # A rows 0,1 cols 2,3; B rows 2,3 cols 0,1,2,3,4
    # Split B into cols 0,1 (2x2), cols 2,3 (2x2), col 4 (2x1)
    add_strassen_2x2(0, 2, 2, 0, 0, 0)      # A[:,2:4] @ B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_2x2(0, 2, 2, 2, 0, 2)      # A[:,2:4] @ B[2:4, 2:4] -> C[0:2, 2:4]
    add_schoolbook_2x2_2x1(0, 2, 2, 4, 0, 4) # A[:,2:4] @ B[2:4, 4:5] -> C[0:2, 4:5]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

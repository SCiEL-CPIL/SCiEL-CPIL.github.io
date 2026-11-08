import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # We will construct a decomposition of rank 58
    # Using two Strassen blocks of size 2x2x2
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {2,3}, k in {2,3}
    # Remaining terms: schoolbook
    
    rank = 58
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen terms for a 2x2x2 block
    # Indices: i in {0,1}, j in {j1, j2}, k in {k1, k2}
    def add_strassen_block(j1, j2, k1, k2, start_col):
        # M1 = (A0j1 + A1j2)(Bj1k1 + Bj2k2)
        # U: A0j1=1, A1j2=1
        # V: Bj1k1=1, Bj2k2=1
        # W: C0k1=1, C1k2=1
        U[0 * m + j1][start_col] = 1
        U[1 * m + j2][start_col] = 1
        V[j1 * p + k1][start_col] = 1
        V[j2 * p + k2][start_col] = 1
        W[k1 * n + 0][start_col] = 1
        W[k2 * n + 1][start_col] = 1
        col_ret = start_col + 1
        
        # M2 = (A1j1 + A1j2)Bj1k1  -- Wait, standard Strassen uses specific pairs
        # Let's use the standard Strassen formulas mapped to local indices 0,1 for j and k
        # Local j0=j1, j1=j2. Local k0=k1, k1=k2.
        # M1 = (A0j0 + A1j1)(Bj0k0 + Bj1k1)
        # M2 = (A1j0 + A1j1)Bj0k0
        # M3 = A0j0(Bj0k1 - Bj1k1)
        # M4 = A1j1(Bj1k0 - Bj0k0)
        # M5 = (A0j0 + A0j1)Bj1k1
        # M6 = (A1j0 - A0j0)(Bj0k0 + Bj0k1)
        # M7 = (A0j1 - A1j1)(Bj1k0 + Bj1k1)
        
        # We already added M1 above?
        # M1: A0j1 is A[0, j1]. A1j2 is A[1, j2].
        # My M1 above used A[0, j1] and A[1, j2].
        # Standard M1 uses A[0, j0] and A[1, j1].
        # Let's restart the block addition carefully.
        
        # Reset the column we just used? No, we are building it.
        # Let's implement a helper that adds 7 terms.
        
        pass
        
    # Let's just hardcode the 7 terms for each block to be safe and clear.
    
    def add_block(j1, j2, k1, k2, start_col):
        c = start_col
        # M1: (A_0,j1 + A_1,j2) * (B_j1,k1 + B_j2,k2)
        U[0*m + j1][c] = 1; U[1*m + j2][c] = 1
        V[j1*p + k1][c] = 1; V[j2*p + k2][c] = 1
        W[k1*n + 0][c] = 1; W[k2*n + 1][c] = 1
        c += 1
        
        # M2: (A_1,j1 + A_1,j2) * B_j1,k1
        U[1*m + j1][c] = 1; U[1*m + j2][c] = 1
        V[j1*p + k1][c] = 1
        W[k1*n + 1][c] = 1
        c += 1
        
        # M3: A_0,j1 * (B_j1,k2 - B_j2,k2)
        U[0*m + j1][c] = 1
        V[j1*p + k2][c] = 1; V[j2*p + k2][c] = -1
        W[k2*n + 0][c] = 1
        c += 1
        
        # M4: A_1,j2 * (B_j2,k1 - B_j1,k1)
        U[1*m + j2][c] = 1
        V[j2*p + k1][c] = 1; V[j1*p + k1][c] = -1
        W[k1*n + 1][c] = 1
        c += 1
        
        # M5: (A_0,j1 + A_0,j2) * B_j2,k2
        U[0*m + j1][c] = 1; U[0*m + j2][c] = 1
        V[j2*p + k2][c] = 1
        W[k2*n + 0][c] = -1
        c += 1
        
        # M6: (A_1,j1 - A_0,j1) * (B_j1,k1 + B_j1,k2)
        U[1*m + j1][c] = 1; U[0*m + j1][c] = -1
        V[j1*p + k1][c] = 1; V[j1*p + k2][c] = 1
        W[k1*n + 1][c] = 1; W[k2*n + 1][c] = 1
        c += 1
        
        # M7: (A_0,j2 - A_1,j2) * (B_j2,k1 + B_j2,k2)
        U[0*m + j2][c] = 1; U[1*m + j2][c] = -1
        V[j2*p + k1][c] = 1; V[j2*p + k2][c] = 1
        W[k1*n + 0][c] = 1; W[k2*n + 0][c] = 1
        c += 1
        
        return c

    # Block 1: j in {0,1}, k in {0,1}
    col = add_block(0, 1, 0, 1, 0)
    
    # Block 2: j in {2,3}, k in {2,3}
    col = add_block(2, 3, 2, 3, col)
    
    # Remaining terms
    # Iterate over all i, j, k
    # If (j,k) is in block 1 or block 2, skip
    # Block 1 indices: j in {0,1}, k in {0,1}
    # Block 2 indices: j in {2,3}, k in {2,3}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this (j,k) pair is covered by a Strassen block
                # Note: Strassen blocks cover specific (j,k) pairs for ALL i in the block (0,1)
                # Since n=2, all i are covered.
                covered = False
                if j in [0, 1] and k in [0, 1]:
                    covered = True
                if j in [2, 3] and k in [2, 3]:
                    covered = True
                
                if not covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    # The total rank used is 'col'
    # It should be 58
    # If col < 58, we can truncate or just return the smaller rank
    # If col > 58, something is wrong
    
    # Return the matrices, trimmed to 'col'
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

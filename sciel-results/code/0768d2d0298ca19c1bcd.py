"""Matrix multiplication 2,3,8: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix mult."""
    start_time = time.time()
    deadline = start_time + 120  # 120s deadline with margin

    n, m, p = DIMENSIONS
    
    # Rank 44 decomposition using 4 blocks of 2x3x2 (rank 11 each)
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx, col] = val
        V[v_idx, col] = val
        W[w_idx, col] = val
        col += 1
        
    # Strassen decomposition for 2x2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # 1: (a11+a22)(b11+b22)
    # 2: (a21+a22)b11
    # 3: a11(b12-b22)
    # 4: a22(b11-b12)
    # 5: (a11+a12)b22
    # 6: (a21-a11)(b11+b12)
    # 7: (a12-a22)(b21+b22)
    
    # We have 4 blocks corresponding to columns 0-1, 2-3, 4-5, 6-7 of the 8-column output.
    # Each block is a 2x3 * 3x2 multiplication.
    # Block r (0 to 3) uses columns 2r, 2r+1 of the output (and corresponding inputs).
    
    col = 0
    for block in range(4):
        # Input A is full 2x3.
        # Input B is 3x2 (columns 2*block, 2*block+1 of the full 3x8 matrix).
        # Output C is 2x2 (columns 2*block, 2*block+1 of the full 2x8 matrix).
        
        # Indices in U (2x3): row i, col j. Index i*3 + j.
        # Indices in V (3x8): row j, col k. Index j*8 + k.
        # Indices in W (8x2): row k, col i. Index k*2 + i. (Wait, W is p*n = 8*2 = 16).
        # W index: k*n + i.
        
        # 1. Strassen part for A[:, 0:2] * B[0:2, 2*block:2*block+2]
        # A_11 = U[0], A_12 = U[1], A_21 = U[3], A_22 = U[4]
        # B_11 = V[0*8 + 2*block], B_12 = V[0*8 + 2*block+1]
        # B_21 = V[1*8 + 2*block], B_22 = V[1*8 + 2*block+1]
        
        # Term 1: (A11 + A22)(B11 + B22)
        U[0, col] = 1; U[4, col] = 1
        V[2*block, col] = 1; V[8+2*block, col] = 1
        W[2*block, col] = 1; W[2*block+1, col] = 1 # C11, C12? No.
        # W index k*n + i.
        # C11 is k=2*block, i=0. W[2*block, col] = 1.
        # C12 is k=2*block+1, i=0. W[2*block+1, col] = 1.
        # C21 is k=2*block, i=1. W[2*block+2, col] = 1.
        # C22 is k=2*block+1, i=1. W[2*block+3, col] = 1.
        
        # Wait, Strassen output:
        # C11 = M1 - M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 - M3 + M6
        
        # So we need to add these linear combinations.
        # Instead of adding 7 terms and then combining, we add the 7 products as separate rank-1 terms?
        # No, the rank-1 terms are the products themselves.
        # The final W matrix entries must reflect the linear combination.
        # So for each of the 7 Strassen products, we add 1 to the appropriate W entries.
        
        # Let's define the 7 Strassen products for this block.
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B11-B12)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        
        # Map to U, V, W contributions.
        
        # P1
        U[0, col] = 1; U[4, col] = 1
        V[2*block, col] = 1; V[8+2*block, col] = 1
        W[2*block, col] = 1 # C11
        W[2*block+1, col] = 1 # C12
        W[2*block+2, col] = 1 # C21
        W[2*block+3, col] = 1 # C22
        col += 1
        
        # P2
        U[3, col] = 1; U[4, col] = 1
        V[2*block, col] = 1
        W[2*block+2, col] = 1 # C21
        W[2*block+3, col] = -1 # C22 (from -M2)
        col += 1
        
        # P3
        U[0, col] = 1
        V[8+2*block+1, col] = 1 # B12
        V[8+2*block+1, col] -= 1 # Wait, B12-B22.
        # V index for B12 is 1*8 + 2*block + 1? No.
        # B is 3x8. Row j, Col k.
        # B12 is j=0, k=2*block+1. Index 2*block+1.
        # B22 is j=1, k=2*block+1. Index 8+2*block+1.
        # So V[2*block+1, col] = 1
        # V[8+2*block+1, col] = -1
        V[2*block+1, col] = 1
        V[8+2*block+1, col] = -1
        W[2*block+1, col] = 1 # C12
        W[2*block+3, col] = -1 # C22 (from -M3)
        col += 1
        
        # P4
        U[4, col] = 1
        V[2*block, col] = 1 # B11
        V[8+2*block+1, col] = -1 # B12? No, B11-B12.
        # B11 is j=0, k=2*block. Index 2*block.
        # B12 is j=0, k=2*block+1. Index 2*block+1.
        V[2*block, col] = 1
        V[2*block+1, col] = -1
        W[2*block, col] = -1 # C11 (from -M4)
        W[2*block+2, col] = 1 # C21
        col += 1
        
        # P5
        U[0, col] = 1; U[1, col] = 1
        V[8+2*block+1, col] = 1 # B22
        W[2*block, col] = -1 # C11 (from -M5)
        W[2*block+1, col] = 1 # C12
        col += 1
        
        # P6
        U[3, col] = 1; U[0, col] = -1
        V[2*block, col] = 1 # B11
        V[2*block+1, col] = 1 # B12
        W[2*block+3, col] = 1 # C22 (from M6)
        col += 1
        
        # P7
        U[1, col] = 1; U[4, col] = -1
        V[8+2*block, col] = 1 # B21
        V[8+2*block+1, col] = 1 # B22
        W[2*block, col] = 1 # C11 (from M7)
        col += 1
        
        # 2. The 4 cross terms for A[:, 2] * B[2, 2*block:2*block+2]
        # A13 = U[2], A23 = U[5]
        # B31 = V[16+2*block], B32 = V[16+2*block+1]
        
        # P8: A13 B31 -> C11
        U[2, col] = 1
        V[16+2*block, col] = 1
        W[2*block, col] = 1
        col += 1
        
        # P9: A13 B32 -> C12
        U[2, col] = 1
        V[16+2*block+1, col] = 1
        W[2*block+1, col] = 1
        col += 1
        
        # P10: A23 B31 -> C21
        U[5, col] = 1
        V[16+2*block, col] = 1
        W[2*block+2, col] = 1
        col += 1
        
        # P11: A23 B32 -> C22
        U[5, col] = 1
        V[16+2*block+1, col] = 1
        W[2*block+3, col] = 1
        col += 1

    # Convert to lists
    U_list = [[int(U[i, j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[int(V[i, j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[int(W[i, j]) for j in range(rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

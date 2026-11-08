"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

import numpy as np
import time
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 by 4x8 matrix multiplication.
    Returns {'U': list, 'V': list, 'W': list} where:
      U has shape (n*m, rank) = (8, rank)
      V has shape (m*p, rank) = (32, rank)
      W has shape (p*n, rank) = (16, rank)
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # We will construct a rank 56 decomposition using Strassen's algorithm on 2x2 blocks.
    # M(2,4,8) is split into two M(2,4,4) multiplications (left and right halves of B).
    # Each M(2,4,4) is split into four M(2,2,2) multiplications.
    # Each M(2,2,2) uses Strassen (rank 7).
    # Total rank = 2 * 4 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # We need to map these to the global indices.
    # Global indices:
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]
    
    # Let's define a helper to add a Strassen block for a specific 2x2x2 sub-problem.
    # Sub-problem: A_sub (2x2), B_sub (2x2), C_sub (2x2).
    # A_sub rows: i in [i0, i0+2), cols: j in [j0, j0+2)
    # B_sub rows: j in [j0, j0+2), cols: k in [k0, k0+2)
    # C_sub rows: i in [i0, i0+2), cols: k in [k0, k0+2)
    
    def add_strassen_block(i0, j0, k0, base_rank):
        # 7 multiplications
        # M1: (A[i0, j0] + A[i0+1, j0+1]) * (B[j0, k0] + B[j0+1, k0+1])
        # Indices:
        # A indices: i0*m + j0, (i0+1)*m + (j0+1)
        # B indices: j0*p + k0, (j0+1)*p + (k0+1)
        # C indices: k0*n + i0, (k0+1)*n + (i0+1) -- Wait, W index is k*n + i.
        
        # Let's define the 7 terms.
        # Each term is a product of a linear combination of A and a linear combination of B.
        # U entries are coefficients for A.
        # V entries are coefficients for B.
        # W entries are coefficients for C.
        
        # Term 1: M1
        # U: +1 at i0*m+j0, +1 at (i0+1)*m+(j0+1)
        # V: +1 at j0*p+k0, +1 at (j0+1)*p+(k0+1)
        # W: +1 at k0*n+i0 (for C[i0, k0]? No, C[i, k] = sum ...
        # The output C[i, k] gets contributions from M1, M4, M5, M7.
        # C[i0, k0] += M1 - M4 + M5 - M7 ? No.
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # So for each M_x, we add +1 or -1 to the appropriate W entries.
        
        # M1 contributes to C[0,0] and C[1,1].
        # C[0,0] index: k0*n + i0.
        # C[1,1] index: (k0+1)*n + (i0+1).
        
        # M1:
        U[i0*m + j0][base_rank] = 1
        U[(i0+1)*m + (j0+1)][base_rank] = 1
        V[j0*p + k0][base_rank] = 1
        V[(j0+1)*p + (k0+1)][base_rank] = 1
        W[k0*n + i0][base_rank] = 1
        W[(k0+1)*n + (i0+1)][base_rank] = 1
        
        # M2: (c+d)e = (A[1,0] + A[1,1]) * B[0,0]
        # C[1,0] += M2
        U[i0*m + j0+1][base_rank+1] = 1 # Wait, A[1,0] is i0+1, j0. A[1,1] is i0+1, j0+1.
        # Correction:
        # A_sub = [[A[i0,j0], A[i0,j0+1]], [A[i0+1,j0], A[i0+1,j0+1]]]
        # c = A[i0+1, j0], d = A[i0+1, j0+1]
        # e = B[j0, k0]
        U[(i0+1)*m + j0][base_rank+1] = 1
        U[(i0+1)*m + (j0+1)][base_rank+1] = 1
        V[j0*p + k0][base_rank+1] = 1
        W[(k0)*n + (i0+1)][base_rank+1] = 1 # C[1,0] index: k0*n + (i0+1)
        
        # M3: a(f-h) = A[0,0] * (B[0,1] - B[1,1])
        # C[0,1] += M3
        U[i0*m + j0][base_rank+2] = 1
        V[j0*p + (k0+1)][base_rank+2] = 1
        V[(j0+1)*p + (k0+1)][base_rank+2] = -1
        W[(k0+1)*n + i0][base_rank+2] = 1 # C[0,1] index: (k0+1)*n + i0
        
        # M4: d(g-e) = A[1,1] * (B[1,0] - B[0,0])
        # C[0,0] += -M4, C[1,0] += M4
        U[(i0+1)*m + (j0+1)][base_rank+3] = 1
        V[(j0+1)*p + k0][base_rank+3] = 1
        V[j0*p + k0][base_rank+3] = -1
        W[k0*n + i0][base_rank+3] = -1
        W[k0*n + (i0+1)][base_rank+3] = 1
        
        # M5: (a+b)h = (A[0,0] + A[0,1]) * B[1,1]
        # C[0,0] += -M5, C[0,1] += M5
        U[i0*m + j0][base_rank+4] = 1
        U[i0*m + (j0+1)][base_rank+4] = 1
        V[(j0+1)*p + (k0+1)][base_rank+4] = 1
        W[k0*n + i0][base_rank+4] = -1
        W[(k0+1)*n + i0][base_rank+4] = 1
        
        # M6: (c-a)(e+f) = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        # C[1,1] += M6
        U[(i0+1)*m + j0][base_rank+5] = 1
        U[i0*m + j0][base_rank+5] = -1
        V[j0*p + k0][base_rank+5] = 1
        V[j0*p + (k0+1)][base_rank+5] = 1
        W[(k0+1)*n + (i0+1)][base_rank+5] = 1
        
        # M7: (b-d)(g+h) = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        # C[0,0] += M7
        U[i0*m + (j0+1)][base_rank+6] = 1
        U[(i0+1)*m + (j0+1)][base_rank+6] = -1
        V[(j0+1)*p + k0][base_rank+6] = 1
        V[(j0+1)*p + (k0+1)][base_rank+6] = 1
        W[k0*n + i0][base_rank+6] = 1
        
        return base_rank + 7

    # We have 4 blocks for the left half of B (k=0..3) and 4 for the right half (k=4..7).
    # Block 1: i0=0, j0=0, k0=0
    # Block 2: i0=0, j0=2, k0=0
    # Block 3: i0=0, j0=0, k0=2
    # Block 4: i0=0, j0=2, k0=2
    # Wait, A is 2x4. B is 4x4.
    # A = [A11 A12; A21 A22] with 2x2 blocks?
    # A rows 0,1. Cols 0,1,2,3.
    # A11: rows 0,1 cols 0,1. (i0=0, j0=0)
    # A12: rows 0,1 cols 2,3. (i0=0, j0=2)
    # B is 4x4.
    # B11: rows 0,1 cols 0,1. (j0=0, k0=0)
    # B12: rows 0,1 cols 2,3. (j0=0, k0=2)
    # B21: rows 2,3 cols 0,1. (j0=2, k0=0)
    # B22: rows 2,3 cols 2,3. (j0=2, k0=2)
    
    # C = A B
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21  -- Wait, A is 2x4.
    # A has 2 rows. So A11 and A12 are the top row blocks?
    # No, A is 2x4.
    # A = [[a00 a01 a02 a03], [a10 a11 a12 a13]]
    # We can split A into A_left (2x2) and A_right (2x2).
    # A_left: cols 0,1. A_right: cols 2,3.
    # B is 4x4.
    # B_top (2x4) and B_bottom (2x4)?
    # B = [[b00 b01 b02 b03], [b10 b11 b12 b13], [b20 b21 b22 b23], [b30 b31 b32 b33]]
    # C = A_left * B_top + A_right * B_bottom
    # C is 2x4.
    
    # So we have two terms:
    # Term 1: A_left (2x2) * B_top (2x4)
    # Term 2: A_right (2x2) * B_bottom (2x4)
    
    # A_left * B_top:
    # B_top is 2x4. Split into B11 (2x2) and B12 (2x2).
    # A_left * B11 (2x2x2)
    # A_left * B12 (2x2x2)
    
    # A_right * B_bottom:
    # B_bottom is 2x4. Split into B21 (2x2) and B22 (2x2).
    # A_right * B21 (2x2x2)
    # A_right * B22 (2x2x2)
    
    # So 4 Strassen blocks.
    
    # Block 1: A_left * B11
    # i0=0, j0=0, k0=0
    # A indices: i in 0,1; j in 0,1.
    # B indices: j in 0,1; k in 0,1.
    # C indices: i in 0,1; k in 0,1.
    cur_rank = 0
    cur_rank = add_strassen_block(0, 0, 0, cur_rank)
    
    # Block 2: A_left * B12
    # i0=0, j0=0, k0=2
    # A indices: i in 0,1; j in 0,1.
    # B indices: j in 0,1; k in 2,3.
    # C indices: i in 0,1; k in 2,3.
    cur_rank = add_strassen_block(0, 0, 2, cur_rank)
    
    # Block 3: A_right * B21
    # i0=0, j0=2, k0=0
    # A indices: i in 0,1; j in 2,3.
    # B indices: j in 2,3; k in 0,1.
    # C indices: i in 0,1; k in 0,1.
    cur_rank = add_strassen_block(0, 2, 0, cur_rank)
    
    # Block 4: A_right * B22
    # i0=0, j0=2, k0=2
    # A indices: i in 0,1; j in 2,3.
    # B indices: j in 2,3; k in 2,3.
    # C indices: i in 0,1; k in 2,3.
    cur_rank = add_strassen_block(0, 2, 2, cur_rank)
    
    # This covers B cols 0..3.
    # Now for B cols 4..7.
    # B is 4x8.
    # B_top_right (2x4) and B_bottom_right (2x4).
    # A_left * B_top_right
    # A_right * B_bottom_right
    
    # Block 5: A_left * B_top_right (k=4..7)
    # i0=0, j0=0, k0=4
    cur_rank = add_strassen_block(0, 0, 4, cur_rank)
    
    # Block 6: A_left * B_top_right (k=6..7)
    # i0=0, j0=0, k0=6
    cur_rank = add_strassen_block(0, 0, 6, cur_rank)
    
    # Block 7: A_right * B_bottom_right (k=4..7)
    # i0=0, j0=2, k0=4
    cur_rank = add_strassen_block(0, 2, 4, cur_rank)
    
    # Block 8: A_right * B_bottom_right (k=6..7)
    # i0=0, j0=2, k0=6
    cur_rank = add_strassen_block(0, 2, 6, cur_rank)
    
    # Total rank should be 8 * 7 = 56.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

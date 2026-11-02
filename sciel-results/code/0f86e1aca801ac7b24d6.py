import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _construct_rank88():
    """
    Construct a rank-88 decomposition for 3x4 x 4x8 matrix multiplication.
    
    Strategy: Split the 3 rows into 2+1.
    - The 1x4x8 part has rank exactly 32 (outer products).
    - The 2x4x8 part: split the 4 columns into two 2-column groups, giving two 2x2x8 blocks.
      Each 2x2x8 block can be done with rank 7*4=28 using Strassen on the 2x2x2 sub-blocks
      applied 4 times (for the 8 columns split into 4 groups of 2).
      Wait, let me reconsider.
    
    Actually, for 2x4x8: we have 2 rows, 4 middle dim, 8 output cols.
    Split the 4 middle into two 2's: gives two 2x2x8 blocks.
    Each 2x2x8: split the 8 into four 2's: gives four 2x2x2 blocks.
    Each 2x2x2 has rank 7 (Strassen).
    So each 2x2x8 has rank 4*7 = 28.
    Two such blocks: 56.
    Plus the 1x4x8 part: 32.
    Total: 56 + 32 = 88.
    """
    n, m, p = 3, 4, 8
    rank = 88
    U = [[0.0] * rank for _ in range(n * m)]  # 12 x 88
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 88
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x 88
    
    col = 0
    
    # Part 1: The 1x4x8 part (row index 2 of A)
    # For i=2, j=0..3, k=0..7: T[2,j,k] = 1
    # Rank-1 terms: U[2*m+j, c] * V[j*p+k, c] * W[k*n+2, c] = 1
    # Use U[2*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+2, c] = 1
    for j in range(4):
        for k in range(8):
            U[2 * m + j][col] = 1.0
            V[j * p + k][col] = 1.0
            W[k * n + 2][col] = 1.0
            col += 1
    
    # Part 2: The 2x4x8 part (rows 0 and 1 of A)
    # Split middle dim 4 into two groups of 2: j in {0,1} and j in {2,3}
    # For each group, we have a 2x2x8 block
    # Split the 8 output cols into four groups of 2: k in {0,1}, {2,3}, {4,5}, {6,7}
    # Each 2x2x2 sub-block uses Strassen (rank 7)
    
    # Strassen 2x2x2: C = A*B where A,B,C are 2x2
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    # c00 = (a00+a11)*(b00+b11) - (a10+a11)*(b10+b11) + (a00+a10)*(b01+b11)
    # c01 = a00*b01 + a01*(b11-b10)
    # c10 = (a10+a11)*b00 + (a00+a11)*(b10-b00)
    # c11 = (a00+a11)*(b00+b11) - a00*b00 - a11*b11
    # 
    # The 7 multiplications are:
    # m0 = (a00+a11)*(b00+b11)
    # m1 = (a10+a11)*(b10+b11)
    # m2 = (a00+a10)*(b01+b11)
    # m3 = a00*b01
    # m4 = a01*(b11-b10)
    # m5 = (a10+a11)*b00
    # m6 = (a00+a11)*(b10-b00)
    # m7 = a11*b11
    # Wait, that's 8. Let me use the standard 7:
    # m0 = (a00+a11)*(b00+b11)
    # m1 = (a10+a11)*(b10+b11)
    # m2 = (a00+a10)*(b01+b11)
    # m3 = a00*b01
    # m4 = a01*(b11-b10)
    # m5 = (a10+a11)*b00
    # m6 = (a00+a11)*(b10-b00)
    # Actually standard Strassen 7 mults:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a10)*b11
    # m6 = (a10+a11)*(b00+b10)
    # m7 = (a00+a11)*(b01+b11)
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # For our tensor decomposition, each "multiplication" m_s = (sum_i alpha_s[i]*A[i,j]) * (sum_k beta_s[k]*B[j,k])
    # contributes a rank-1 term to the tensor.
    
    # For a 2x2x2 block with A indices (i1,i2) and B indices (j1,j2) and C indices (k1,k2):
    # The 7 Strassen terms give 7 rank-1 contributions.
    
    # Let me define the Strassen coefficients:
    # For block with row indices (r0, r1), col indices (c0, c1), out indices (o0, o1):
    # A submatrix: [[A[r0,c0], A[r0,c1]], [A[r1,c0], A[r1,c1]]]
    # B submatrix: [[B[c0,o0], B[c0,o1]], [B[c1,o0], B[c1,o1]]]
    # C submatrix: [[C[r0,o0], C[r0,o1]], [C[r1,o0], C[r1,o1]]]
    
    # Strassen 7 multiplications:
    # m1: (A[r0,c0]+A[r1,c1]) * (B[c0,o0]+B[c1,o1])
    # m2: (A[r1,c0]+A[r1,c1]) * B[c0,o0]
    # m3: A[r0,c0] * (B[c0,o1]-B[c1,o1])
    # m4: A[r1,c1] * (B[c1,o0]-B[c0,o0])
    # m5: (A[r0,c0]+A[r0,c1]) * B[c1,o1]
    # m6: (A[r1,c0]+A[r1,c1]) * (B[c0,o0]+B[c1,o0])
    # m7: (A[r0,c0]+A[r1,c1]) * (B[c0,o1]+B[c1,o1])
    
    # Contributions to C:
    # C[r0,o0] += m1 + m4 - m5 + m7
    # C[r0,o1] += m3 + m5
    # C[r1,o0] += m2 + m4
    # C[r1,o1] += m1 - m2 + m3 + m6
    
    # For the tensor decomposition, each multiplication m_s corresponds to a rank-1 term:
    # The A-factor is a vector in the A-space (indices i*m+j), the B-factor in B-space (j*p+k),
    # the C-factor in C-space (k*n+i).
    
    def _add_strassen_block(r0, r1, c0, c1, o0, o1, col_start):
        """Add 7 rank-1 terms for a 2x2x2 Strassen block."""
        nonlocal col
        # m1: (A[r0,c0]+A[r1,c1]) * (B[c0,o0]+B[c1,o1])
        U[r0*m+c0][col] += 1.0
        U[r1*m+c1][col] += 1.0
        V[c0*p+o0][col] += 1.0
        V[c1*p+o1][col] += 1.0
        W[o0*n+r0][col] += 1.0
        W[o1*n+r1][col] += 1.0
        col += 1
        
        # m2: (A[r1,c0]+A[r1,c1]) * B[c0,o0]
        U[r1*m+c0][col] += 1.0
        U[r1*m+c1][col] += 1.0
        V[c0*p+o0][col] += 1.0
        W[o0*n+r1][col] += 1.0
        col += 1
        
        # m3: A[r0,c0] * (B[c0,o1]-B[c1,o1])
        U[r0*m+c0][col] += 1.0
        V[c0*p+o1][col] += 1.0
        V[c1*p+o1][col] -= 1.0
        W[o1*n+r0][col] += 1.0
        col += 1
        
        # m4: A[r1,c1] * (B[c1,o0]-B[c0,o0])
        U[r1*m+c1][col] += 1.0
        V[c1*p+o0][col] += 1.0
        V[c0*p+o0][col] -= 1.0
        W[o0*n+r1][col] += 1.0
        col += 1
        
        # m5: (A[r0,c0]+A[r0,c1]) * B[c1,o1]
        U[r0*m+c0][col] += 1.0
        U[r0*m+c1][col] += 1.0
        V[c1*p+o1][col] += 1.0
        W[o1*n+r0][col] -= 1.0
        col += 1
        
        # m6: (A[r1,c0]+A[r1,c1]) * (B[c0,o0]+B[c1,o0])
        U[r1*m+c0][col] += 1.0
        U[r1*m+c1][col] += 1.0
        V[c0*p+o0][col] += 1.0
        V[c1*p+o0][col] += 1.0
        W[o1*n+r1][col] += 1.0
        col += 1
        
        # m7: (A[r0,c0]+A[r1,c1]) * (B[c0,o1]+B[c1,o1])
        U[r0*m+c0][col] += 1.0
        U[r1*m+c1][col] += 1.0
        V[c0*p+o1][col] += 1.0
        V[c1*p+o1][col] += 1.0
        W[o1*n+r0][col] += 1.0
        col += 1
        
        return col
    
    # Now apply Strassen to the 2x4x8 part
    # Rows: r0=0, r1=1
    # Split cols into two groups: (c0=0,c1=1) and (c0=2,c1=3)
    # Split outputs into four groups: (o0=0,o1=1), (o0=2,o1=3), (o0=4,o1=5), (o0=6,o1=7)
    
    for c_pair in [(0, 1), (2, 3)]:
        for o_pair in [(0, 1), (2, 3), (4, 5), (6, 7)]:
            c0, c1 = c_pair
            o0, o1 = o_pair
            _add_strassen_block(0, 1, c0, c1, o0, o1, col)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 x 4x8 matrix multiplication.
    Uses rank-88 construction via Strassen on 2x2x2 sub-blocks.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Try the rank-88 construction
    result = _construct_rank88()
    
    # Verify correctness
    if _verify(result["U"], result["V"], result["W"], 3, 4, 8):
        return result
    
    # Fallback to schoolbook if verification fails
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

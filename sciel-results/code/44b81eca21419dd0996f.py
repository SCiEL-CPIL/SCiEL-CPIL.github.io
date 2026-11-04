"""Rewritten program for rank-minimal exact tensor decomposition for 3x4 x 4x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Rank 66 decomposition
    # Split j into two blocks of 2: J1=[0,1], J2=[2,3]
    # Split k into three blocks of 2: K1=[0,1], K2=[2,3], K3=[4,5]
    # For each k-block, we compute A1*B1 + A2*B2
    # A1 is 3x2, B1 is 2x2. A2 is 3x2, B2 is 2x2.
    
    # Sub-decomposition for 3x2 x 2x2:
    # Split i into i0=0 and i1=[1,2]
    # i0: 1x2 x 2x2 -> Rank 4 (Trivial)
    # i1: 2x2 x 2x2 -> Rank 7 (Strassen)
    # Total 11 per 3x2 x 2x2 block.
    # Two such blocks (A1, A2) per k-block -> 22.
    # Three k-blocks -> 66.

    R = 66
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    # Indices for U: i*m + j
    # Indices for V: j*p + k
    # Indices for W: k*n + i

    def add_trivial_1x2_2x2(offset, i_row, j_block, k_block):
        """
        Adds trivial decomposition for 1x2 x 2x2.
        i_row is the specific i index (0).
        j_block is [j0, j1] (indices in original m).
        k_block is [k0, k1] (indices in original p).
        """
        j0, j1 = j_block
        k0, k1 = k_block
        # Terms:
        # 1: U[i,j0]=1, V[j0,k0]=1, W[k0,i]=1
        # 2: U[i,j0]=1, V[j0,k1]=1, W[k1,i]=1
        # 3: U[i,j1]=1, V[j1,k0]=1, W[k0,i]=1
        # 4: U[i,j1]=1, V[j1,k1]=1, W[k1,i]=1
        
        U[i_row * m + j0][offset] = 1
        V[j0 * p + k0][offset] = 1
        W[k0 * n + i_row][offset] = 1
        offset += 1

        U[i_row * m + j0][offset] = 1
        V[j0 * p + k1][offset] = 1
        W[k1 * n + i_row][offset] = 1
        offset += 1

        U[i_row * m + j1][offset] = 1
        V[j1 * p + k0][offset] = 1
        W[k0 * n + i_row][offset] = 1
        offset += 1

        U[i_row * m + j1][offset] = 1
        V[j1 * p + k1][offset] = 1
        W[k1 * n + i_row][offset] = 1
        offset += 1

        return offset

    def add_strassen_2x2_2x2(offset, i_block, j_block, k_block):
        """
        Adds Strassen decomposition for 2x2 x 2x2.
        i_block is [i0, i1] (indices in original n).
        j_block is [j0, j1] (indices in original m).
        k_block is [k0, k1] (indices in original p).
        """
        i0, i1 = i_block
        j0, j1 = j_block
        k0, k1 = k_block

        # P1 = (A00 + A11)(B00 + B11)
        # U: A00=1, A11=1. V: B00=1, B11=1. W: R00=1, R11=1.
        # This is not rank 1. We need to decompose the product of sums.
        # Strassen uses 7 multiplications.
        # We map the 7 terms to the 7 rank-1 tensors.
        
        # Term 1: P1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1. V: B00=1, B11=1.
        # This contributes to R00, R01, R10, R11?
        # No, the multiplication is scalar.
        # The tensor entry is U[i,j] V[j,k] W[k,i].
        # We need to form the linear combinations in W.
        
        # Let's define the 7 intermediate values P1..P7.
        # Each Pc corresponds to a rank-1 tensor.
        # U and V are fixed for the Strassen steps.
        # W combines them.
        
        # Step 1: P1
        # U[i0,j0]=1, U[i1,j1]=1
        # V[j0,k0]=1, V[j1,k1]=1
        # This product is (A00+A11)(B00+B11).
        # We assign this to offset 0.
        U[i0 * m + j0][offset] = 1
        U[i1 * m + j1][offset] = 1
        V[j0 * p + k0][offset] = 1
        V[j1 * p + k1][offset] = 1
        # W contributions:
        # P1 adds to R00, R11 (and subtracts from others in final sum)
        # R00 = P1 + P4 - P5 + P7
        # R11 = P1 - P2 + P3 + P6
        # We will accumulate W values.
        W[k0 * n + i0][offset] = 1  # R00
        W[k1 * n + i1][offset] = 1  # R11
        offset += 1

        # Step 2: P2 = (A10+A11)B00
        # U[i1,j0]=1, U[i1,j1]=1
        # V[j0,k0]=1
        U[i1 * m + j0][offset] = 1
        U[i1 * m + j1][offset] = 1
        V[j0 * p + k0][offset] = 1
        # P2 subtracts from R11, adds to R10
        # R10 = P2 + P4
        # R11 = P1 - P2 + P3 + P6
        W[k0 * n + i1][offset] = 1  # R10
        W[k1 * n + i1][offset] = -1 # R11
        offset += 1

        # Step 3: P3 = A00(B01-B11)
        # U[i0,j0]=1
        # V[j0,k1]=1, V[j1,k1]=-1
        U[i0 * m + j0][offset] = 1
        V[j0 * p + k1][offset] = 1
        V[j1 * p + k1][offset] = -1
        # P3 adds to R01, R11
        # R01 = P3 + P5
        # R11 = P1 - P2 + P3 + P6
        W[k1 * n + i0][offset] = 1  # R01
        W[k1 * n + i1][offset] = 1  # R11
        offset += 1

        # Step 4: P4 = A11(B10-B00)
        # U[i1,j1]=1
        # V[j1,k0]=1, V[j0,k0]=-1
        U[i1 * m + j1][offset] = 1
        V[j1 * p + k0][offset] = 1
        V[j0 * p + k0][offset] = -1
        # P4 adds to R00, R10
        # R00 = P1 + P4 - P5 + P7
        # R10 = P2 + P4
        W[k0 * n + i0][offset] = 1  # R00
        W[k0 * n + i1][offset] = 1  # R10
        offset += 1

        # Step 5: P5 = (A00+A01)B11
        # U[i0,j0]=1, U[i0,j1]=1
        # V[j1,k1]=1
        U[i0 * m + j0][offset] = 1
        U[i0 * m + j1][offset] = 1
        V[j1 * p + k1][offset] = 1
        # P5 subtracts from R00, adds to R01
        # R00 = P1 + P4 - P5 + P7
        # R01 = P3 + P5
        W[k0 * n + i0][offset] = -1 # R00
        W[k1 * n + i0][offset] = 1  # R01
        offset += 1

        # Step 6: P6 = (A10-A00)(B00+B01)
        # U[i1,j0]=1, U[i0,j0]=-1
        # V[j0,k0]=1, V[j0,k1]=1
        U[i1 * m + j0][offset] = 1
        U[i0 * m + j0][offset] = -1
        V[j0 * p + k0][offset] = 1
        V[j0 * p + k1][offset] = 1
        # P6 adds to R11
        # R11 = P1 - P2 + P3 + P6
        W[k1 * n + i1][offset] = 1  # R11
        offset += 1

        # Step 7: P7 = (A01-A11)(B10+B11)
        # U[i0,j1]=1, U[i1,j1]=-1
        # V[j1,k0]=1, V[j1,k1]=1
        U[i0 * m + j1][offset] = 1
        U[i1 * m + j1][offset] = -1
        V[j1 * p + k0][offset] = 1
        V[j1 * p + k1][offset] = 1
        # P7 adds to R00
        # R00 = P1 + P4 - P5 + P7
        W[k0 * n + i0][offset] = 1  # R00
        offset += 1

        return offset

    # Main loop
    offset = 0
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    for jb in j_blocks:
        for kb in k_blocks:
            # Process A1 (i=0) trivial
            offset = add_trivial_1x2_2x2(offset, 0, jb, kb)
            
            # Process A1 (i=1,2) Strassen
            offset = add_strassen_2x2_2x2(offset, (1, 2), jb, kb)
            
            # Process A2 (i=0) trivial
            offset = add_trivial_1x2_2x2(offset, 0, jb, kb) # Wait, A2 is also 3x2.
            # The j_block for A2 is the same jb.
            # But A2 corresponds to j indices 2,3.
            # My jb is (0,1) or (2,3).
            # So if jb is (0,1), it's A1. If jb is (2,3), it's A2.
            # So I should not loop over jb and then process A1 and A2.
            # I should process A1 for jb=(0,1) and A2 for jb=(2,3).
            
    # Let's restart the construction logic correctly.
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    offset = 0

    # A1 uses j in {0,1}. A2 uses j in {2,3}.
    # For each k_block:
    for kb in k_blocks:
        # A1 part (j=0,1)
        offset = add_trivial_1x2_2x2(offset, 0, (0, 1), kb)
        offset = add_strassen_2x2_2x2(offset, (1, 2), (0, 1), kb)
        
        # A2 part (j=2,3)
        offset = add_trivial_1x2_2x2(offset, 0, (2, 3), kb)
        offset = add_strassen_2x2_2x2(offset, (1, 2), (2, 3), kb)

    if offset != R:
        # Fallback to trivial if construction failed
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build a decomposition of rank 78 by splitting the 4x4 block into two 2x2 blocks.
    # The matrix multiplication is A (3x4) * B (4x7) = C (3x7).
    # Let A = [A1, A2] where A1 is 3x2 and A2 is 3x2.
    # Let B = [B1; B2] where B1 is 2x7 and B2 is 2x7.
    # C = A1*B1 + A2*B2.
    
    # For each term A_i * B_i, we use a rank-14 decomposition for the 2x2x7 part? 
    # Wait, A_i is 3x2, B_i is 2x7.
    # Let's split the 3x2 matrix A_i into 1x2 + 2x2? No, let's just use the known rank for 3x2x7.
    # R(3,2,7) <= R(1,2,7) + R(2,2,7) = 14 + 25 = 39.
    # So total rank 39 + 39 = 78.
    
    # We need to construct the factors for R(3,2,7) = 39.
    # R(1,2,7) = 14. This is trivial: 14 rank-1 terms.
    # R(2,2,7) = 25.
    # R(2,2,7) can be split into R(2,2,4) + R(2,2,3).
    # R(2,2,4) = 14 (two 2x2x2 Strassen blocks).
    # R(2,2,3) = 11 (one 2x2x2 Strassen block + one 2x2x1 block? 7+4=11).
    
    # Let's build the factors for the sub-problems.
    
    # Helper to build rank-1 term
    def add_term(U, V, W, i, j, k, col):
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        U[i * m + j, col] = 1.0
        V[j * p + k, col] = 1.0
        W[k * n + i, col] = 1.0

    # Helper to build Strassen 2x2x2 decomposition
    # Inputs: i0, j0, k0 are the base indices for the 2x2x2 block.
    # U indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
    # V indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
    # W indices: (k0, i0), (k0+1, i0), (k0, i0+1), (k0+1, i0+1)
    # Note: W index is k*n + i.
    def add_strassen(U, V, W, i0, j0, k0, base_col):
        # The 7 Strassen terms for 2x2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B12-B22)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to map these to linear combinations of the basis vectors in U, V, W.
        # Since we are building explicit factors, we can just add the rank-1 terms corresponding to the Strassen algorithm.
        # However, the Strassen algorithm uses sums and differences. 
        # The tensor decomposition is a sum of rank-1 tensors.
        # The rank-1 tensors for Strassen are:
        # T1 = (e11+e22) ot (e11+e22) ot (e11+e22) ? No.
        # The decomposition is:
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # This means the tensor is a linear combination of 7 rank-1 tensors?
        # No, the rank is 7. The 7 rank-1 tensors are:
        # U1 = e11+e22, V1 = e11+e22, W1 = e11+e22 ? No.
        # Let's look at the standard Strassen decomposition as a tensor.
        # The 7 terms are:
        # 1. U: e11+e22, V: e11+e22, W: e11+e22
        # 2. U: e21+e22, V: e11, W: e21
        # 3. U: e11, V: e21-e11, W: e12
        # 4. U: e22, V: e12-e22, W: e21
        # 5. U: e11+e12, V: e22, W: e11
        # 6. U: e21-e11, V: e11+e12, W: e11
        # 7. U: e12-e22, V: e21+e22, W: e22
        
        # Let's verify term 1:
        # (e11+e22) ot (e11+e22) ot (e11+e22)
        # Coeff of C11: 1*1*1 = 1.
        # Coeff of C12: 1*1*0 = 0.
        # Coeff of C21: 1*0*1 = 0.
        # Coeff of C22: 1*1*1 = 1.
        # This gives C11 + C22.
        # But Strassen M1 contributes to C11 and C22.
        # C11 = M1 + ...
        # C22 = M1 - ...
        # So M1 = (A11+A22)(B11+B22).
        # This corresponds to U: e11+e22, V: e11+e22, W: e11+e22?
        # If W is e11+e22, then it adds to C11 and C22.
        # Yes.
        
        # Term 2: M2 = (A21+A22)B11.
        # U: e21+e22, V: e11, W: e21.
        # Contributes to C21.
        # Yes.
        
        # Term 3: M3 = A11(B21-B11).
        # U: e11, V: e21-e11, W: e12.
        # Contributes to C12.
        # Yes.
        
        # Term 4: M4 = A22(B12-B22).
        # U: e22, V: e12-e22, W: e21.
        # Contributes to C21.
        # Yes.
        
        # Term 5: M5 = (A11+A12)B22.
        # U: e11+e12, V: e22, W: e11.
        # Contributes to C11.
        # Yes.
        
        # Term 6: M6 = (A21-A11)(B11+B12).
        # U: e21-e11, V: e11+e12, W: e11.
        # Contributes to C11.
        # Yes.
        
        # Term 7: M7 = (A12-A22)(B21+B22).
        # U: e12-e22, V: e21+e22, W: e22.
        # Contributes to C22.
        # Yes.
        
        # So we just add these 7 rank-1 tensors with coefficients 1.
        
        # Define the vectors
        # U basis: 0:e11, 1:e12, 2:e21, 3:e22 (relative to i0, j0)
        # V basis: 0:e11, 1:e12, 2:e21, 3:e22 (relative to j0, k0)
        # W basis: 0:e11, 1:e12, 2:e21, 3:e22 (relative to k0, i0)
        
        # Note: The indices in the global arrays are:
        # U[i, j] -> i*m + j
        # V[j, k] -> j*p + k
        # W[k, i] -> k*n + i
        
        # We need to add the linear combinations to the columns.
        
        # Term 1
        col = base_col
        U[i0, j0, col] += 1.0
        U[i0+1, j0+1, col] += 1.0
        V[j0, k0, col] += 1.0
        V[j0+1, k0+1, col] += 1.0
        W[k0, i0, col] += 1.0
        W[k0+1, i0+1, col] += 1.0
        
        # Term 2
        col = base_col + 1
        U[i0+1, j0, col] += 1.0
        U[i0+1, j0+1, col] += 1.0
        V[j0, k0, col] += 1.0
        W[k0, i0+1, col] += 1.0
        
        # Term 3
        col = base_col + 2
        U[i0, j0, col] += 1.0
        V[j0+1, k0, col] += 1.0
        V[j0, k0, col] -= 1.0
        W[k0+1, i0, col] += 1.0
        
        # Term 4
        col = base_col + 3
        U[i0+1, j0+1, col] += 1.0
        V[j0, k0+1, col] += 1.0
        V[j0+1, k0+1, col] -= 1.0
        W[k0, i0+1, col] += 1.0
        
        # Term 5
        col = base_col + 4
        U[i0, j0, col] += 1.0
        U[i0, j0+1, col] += 1.0
        V[j0+1, k0+1, col] += 1.0
        W[k0, i0, col] += 1.0
        
        # Term 6
        col = base_col + 5
        U[i0+1, j0, col] += 1.0
        U[i0, j0, col] -= 1.0
        V[j0, k0, col] += 1.0
        V[j0, k0+1, col] += 1.0
        W[k0, i0, col] += 1.0
        
        # Term 7
        col = base_col + 6
        U[i0, j0+1, col] += 1.0
        U[i0+1, j0+1, col] -= 1.0
        V[j0+1, k0, col] += 1.0
        V[j0+1, k0+1, col] += 1.0
        W[k0+1, i0+1, col] += 1.0

    # Helper to build 2x2x1 decomposition (rank 4)
    def add_2x2x1(U, V, W, i0, j0, k0, base_col):
        # 4 rank-1 terms
        for di in range(2):
            for dj in range(2):
                col = base_col + di*2 + dj
                U[i0+di, j0+dj, col] = 1.0
                V[j0+dj, k0, col] = 1.0
                W[k0, i0+di, col] = 1.0

    # Now, assemble the full 3x4x7 decomposition.
    # Total rank 78.
    # We split the 4 columns of A into two groups of 2.
    # Group 1: j=0,1. Group 2: j=2,3.
    # For each group, we compute A_i (3x2) * B_i (2x7).
    # We use the R(3,2,7) = 39 decomposition.
    # R(3,2,7) is split into R(1,2,7) + R(2,2,7).
    # R(1,2,7) = 14.
    # R(2,2,7) = 25.
    # R(2,2,7) is split into R(2,2,4) + R(2,2,3).
    # R(2,2,4) = 14 (two Strassen 2x2x2).
    # R(2,2,3) = 11 (one Strassen 2x2x2 + one 2x2x1).
    
    U = np.zeros((n * m, 78), dtype=float)
    V = np.zeros((m * p, 78), dtype=float)
    W = np.zeros((p * n, 78), dtype=float)
    
    col_offset = 0
    
    for block in range(2):
        # j0 is the start of the 2x2 block in the middle dimension
        j0 = block * 2
        
        # Part 1: R(1,2,7) for i=0 (first row of A)
        # A is 3x2, B is 2x7.
        # We take the first row of A (i=0) and multiply by B (2x7).
        # This is 1x2x7. Rank 14.
        # We can just use 14 rank-1 terms.
        # For k in 0..6:
        #   For j in 0..1:
        #     U[0*m + j0+j, col] = 1
        #     V[(j0+j)*p + k, col] = 1
        #     W[k*n + 0, col] = 1
        for k in range(p):
            for j in range(2):
                col = col_offset
                U[0 * m + (j0 + j), col] = 1.0
                V[(j0 + j) * p + k, col] = 1.0
                W[k * n + 0, col] = 1.0
                col_offset += 1
        # col_offset increased by 14.
        
        # Part 2: R(2,2,7) for i=1,2 (last two rows of A)
        # This is 2x2x7. Rank 25.
        # Split into R(2,2,4) + R(2,2,3).
        
        # R(2,2,4): k=0,1,2,3.
        # Two Strassen blocks: k=0,1 and k=2,3.
        # Block 1: k0=0, k1=1.
        add_strassen(U, V, W, 1, j0, 0, col_offset)
        col_offset += 7
        # Block 2: k0=2, k1=3.
        add_strassen(U, V, W, 1, j0, 2, col_offset)
        col_offset += 7
        # col_offset increased by 14.
        
        # R(2,2,3): k=4,5,6.
        # One Strassen block for k=4,5.
        add_strassen(U, V, W, 1, j0, 4, col_offset)
        col_offset += 7
        # One 2x2x1 block for k=6.
        add_2x2x1(U, V, W, 1, j0, 6, col_offset)
        col_offset += 4
        # col_offset increased by 11.
        
        # Total for this block: 14 + 14 + 11 = 39.
        # col_offset increased by 39.
    
    # Total col_offset should be 78.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

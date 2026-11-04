"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # We use Strassen's algorithm for 4x4x4 matrix multiplication.
    # Partition 4x4 into 2x2 blocks, use Strassen for the block multiplication (7 products),
    # and use Strassen again for each 2x2x2 product (7 scalar multiplications each).
    # Total rank for 4x4x4: 7 * 7 = 49
    
    # For 4x4x9: 9 = 4 + 4 + 1
    # First 4 columns: rank 49
    # Next 4 columns: rank 49  
    # Last 1 column: schoolbook rank 16
    # Total rank: 49 + 49 + 16 = 114
    
    rank = 114
    U = np.zeros((n * m, rank), dtype=int)  # 16 x 114
    V = np.zeros((m * p, rank), dtype=int)  # 36 x 114
    W = np.zeros((p * n, rank), dtype=int)  # 36 x 114
    
    col = 0
    
    def add_strassen_2x2(a_offset, b_offset, c_offset, col_start):
        """
        Add Strassen 2x2x2 decomposition.
        Indices: for 2x2 matrix, element (i,j) is at offset + i*2 + j
        So: (0,0)->+0, (0,1)->+1, (1,0)->+2, (1,1)->+3
        """
        c = col_start
        # M1 = (A11+A22)(B11+B22)
        U[a_offset + 0][c] = 1
        U[a_offset + 3][c] = 1
        V[b_offset + 0][c] = 1
        V[b_offset + 3][c] = 1
        W[c_offset + 0][c] = 1   # C11
        W[c_offset + 3][c] = 1   # C22
        c += 1
        
        # M2 = (A21+A22)B11
        U[a_offset + 2][c] = 1
        U[a_offset + 3][c] = 1
        V[b_offset + 0][c] = 1
        W[c_offset + 2][c] = 1   # C21
        W[c_offset + 3][c] = -1  # C22
        c += 1
        
        # M3 = A11(B21-B22)
        U[a_offset + 0][c] = 1
        V[b_offset + 2][c] = 1
        V[b_offset + 3][c] = -1
        W[c_offset + 1][c] = 1   # C12
        W[c_offset + 3][c] = 1   # C22
        c += 1
        
        # M4 = A22(B12-B11)
        U[a_offset + 3][c] = 1
        V[b_offset + 1][c] = 1
        V[b_offset + 0][c] = -1
        W[c_offset + 0][c] = 1   # C11
        W[c_offset + 2][c] = 1   # C21
        c += 1
        
        # M5 = (A11+A12)B22
        U[a_offset + 0][c] = 1
        U[a_offset + 1][c] = 1
        V[b_offset + 3][c] = 1
        W[c_offset + 0][c] = -1  # C11
        W[c_offset + 1][c] = 1   # C12
        c += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[a_offset + 2][c] = 1
        U[a_offset + 0][c] = -1
        V[b_offset + 0][c] = 1
        V[b_offset + 1][c] = 1
        W[c_offset + 3][c] = 1   # C22
        c += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a_offset + 1][c] = 1
        U[a_offset + 3][c] = -1
        V[b_offset + 2][c] = 1
        V[b_offset + 3][c] = 1
        W[c_offset + 0][c] = 1   # C11
        c += 1
        
        return c
    
    def add_4x4x4_strassen(a_offset, b_offset, c_offset, col_start):
        """
        Add 49-rank Strassen-based decomposition for 4x4x4.
        Uses Strassen for 2x2 block multiplication, with Strassen for each 2x2x2 product.
        
        For 4x4 matrix stored with row index = i*4 + j:
        Block A11 (rows 0-1, cols 0-1): elements at a_offset + 0,1,4,5
        Block A12 (rows 0-1, cols 2-3): elements at a_offset + 2,3,6,7
        Block A21 (rows 2-3, cols 0-1): elements at a_offset + 8,9,12,13
        Block A22 (rows 2-3, cols 2-3): elements at a_offset + 10,11,14,15
        
        We need to create virtual 2x2 matrices for each Strassen product.
        The approach: for each of the 7 Strassen products M1-M7, we create a virtual 2x2 
        matrix from linear combinations of A blocks, multiply by virtual 2x2 from B blocks,
        and distribute to C blocks.
        
        This requires careful bookkeeping. Let me use a different approach:
        I'll directly compute the tensor decomposition by iterating over all possible 
        linear combinations.
        """
        c = col_start
        
        # For 4x4x4 with Strassen:
        # We partition into 2x2 blocks. The 7 Strassen products are:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each M_i is a 2x2 matrix product. We compute each with Strassen (7 scalar mults).
        # For each M_i, we need to:
        # 1. Create virtual 2x2 A matrix from linear combo of A blocks
        # 2. Create virtual 2x2 B matrix from linear combo of B blocks  
        # 3. Compute 2x2x2 with Strassen
        # 4. Distribute result to appropriate C blocks with appropriate coefficients
        
        # Let me define the block element mappings:
        # For a 4x4 block, element at block position (bi, bj) and local position (li, lj)
        # is at global index: a_offset + bi*2*2 + li*2 + bj*2 + lj = a_offset + bi*4 + li*2 + bj*2 + lj
        
        # Block A11: bi=0, bj=0 -> base = a_offset + 0
        # Block A12: bi=0, bj=1 -> base = a_offset + 2
        # Block A21: bi=1, bj=0 -> base = a_offset + 8
        # Block A22: bi=1, bj=1 -> base = a_offset + 10
        
        # For Strassen on 2x2x2, we need 4 elements per 2x2 matrix.
        # The virtual 2x2 matrices are formed from linear combinations of the 4x4 blocks.
        
        # This is getting complex. Let me use a simpler approach:
        # Just use schoolbook for the inner 2x2x2 multiplications.
        # 7 outer Strassen products * 4 schoolbook inner products = 28 rank for 4x4x4.
        
        c = col_start
        
        # For each of the 7 Strassen products, compute the 2x2x2 with schoolbook (4 mults)
        
        # M1 = (A11+A22)(B11+B22)
        # Virtual A: A11+A22, Virtual B: B11+B22
        # A11 elements: a_offset+0, a_offset+1, a_offset+4, a_offset+5
        # A22 elements: a_offset+10, a_offset+11, a_offset+14, a_offset+15
        # B11 elements: b_offset+0, b_offset+1, b_offset+4, b_offset+5
        # B22 elements: b_offset+10, b_offset+11, b_offset+14, b_offset+15
        
        # Schoolbook for 2x2x2: C[i,j] = sum_k A[i,k]*B[k,j]
        # For M1, the result contributes to C11 and C22 with coefficient +1
        
        # C11[i,j] += M1[i,j] for i,j in {0,1}
        # C22[i,j] += M1[i,j] for i,j in {0,1}
        
        # M1[0,0] = (A11[0,0]+A22[0,0])*(B11[0,0]+B22[0,0]) + (A11[0,1]+A22[0,1])*(B11[1,0]+B22[1,0])
        # This expands to 4 terms, but we can do it with 4 scalar multiplications
        # if we precompute the sums. However, in tensor decomposition, each scalar 
        # multiplication is one rank-1 term.
        
        # Actually, for schoolbook 2x2x2:
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]  -> 2 scalar mults
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]  -> 2 scalar mults
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]  -> 2 scalar mults
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]  -> 2 scalar mults
        # Total: 8 scalar multiplications for 2x2x2 schoolbook!
        
        # Wait, that's not right. Schoolbook for 2x2x2 is 4*4=16? No.
        # C[i,j] = sum_k A[i,k] * B[k,j]
        # For 2x2: each C[i,j] requires 2 multiplications, 4 C entries, total 8 multiplications.
        # But that's not minimal. The minimal rank for 2x2x2 is 7 (Strassen).
        
        # Let me just use Strassen for the inner 2x2x2 as well.
        # 7 outer * 7 inner = 49 rank for 4x4x4.
        
        # I'll implement this properly now.
        
        # For each Strassen product M_i, I need to:
        # 1. Define the virtual 2x2 A matrix as a linear combination of 4x4 A blocks
        # 2. Define the virtual 2x2 B matrix as a linear combination of 4x4 B blocks
        # 3. Apply Strassen 2x2x2 to these virtual matrices
        # 4. Add the results to the appropriate C blocks with the appropriate signs
        
        # Let me define helper functions to add contributions.
        
        # For M1 = (A11+A22)(B11+B22):
        # Virtual A[0,0] = A11[0,0] + A22[0,0] = A[0,0] + A[2,2]
        # Virtual A[0,1] = A11[0,1] + A22[0,1] = A[0,1] + A[2,3]
        # Virtual A[1,0] = A11[1,0] + A22[1,0] = A[1,0] + A[3,2]
        # Virtual A[1,1] = A11[1,1] + A22[1,1] = A[1,1] + A[3,3]
        
        # Virtual B[0,0] = B11[0,0] + B22[0,0] = B[0,0] + B[2,2]
        # Virtual B[0,1] = B11[0,1] + B22[0,1] = B[0,1] + B[2,3]
        # Virtual B[1,0] = B11[1,0] + B22[1,0] = B[1,0] + B[3,2]
        # Virtual B[1,1] = B11[1,1] + B22[1,1] = B[1,1] + B[3,3]
        
        # Strassen products for M1:
        # S1 = (VA[0,0]+VA[1,1])*(VB[0,0]+VB[1,1])
        # S2 = (VA[1,0]+VA[1,1])*VB[0,0]
        # S3 = VA[0,0]*(VB[1,0]-VB[1,1])
        # S4 = VA[1,1]*(VB[0,1]-VB[0,0])
        # S5 = (VA[0,0]+VA[0,1])*VB[1,1]
        # S6 = (VA[1,0]-VA[0,0])*(VB[0,0]+VB[0,1])
        # S7 = (VA[0,1]-VA[1,1])*(VB[1,0]+VB[1,1])
        
        # VM[0,0] = S1 + S4 - S5 + S7
        # VM[0,1] = S3 + S5
        # VM[1,0] = S2 + S4
        # VM[1,1] = S1 - S2 + S3 + S6
        
        # C11[0,0] += VM[0,0], C11[0,1] += VM[0,1], C11[1,0] += VM[1,0], C11[1,1] += VM[1,1]
        # C22[0,0] += VM[0,0], C22[0,1] += VM[0,1], C22[1,0] += VM[1,0], C22[1,1] += VM[1,1]
        
        # This is extremely tedious to do by hand. Let me use a computational approach.
        
        # I'll build the decomposition programmatically.
        
        # Define the 4x4x4 tensor and decompose it.
        
        # Actually, let me just use a known rank-49 decomposition from the literature
        # or compute it algorithmically.
        
        # For now, let me fall back to schoolbook for the entire 4x4x9 to ensure correctness.
        # I can optimize later.
        
        return c
    
    # For safety and correctness, use schoolbook for the entire computation.
    # Rank = 4 * 4 * 9 = 144
    
    # But wait, the current program already does this with rank 144.
    # Let me at least try to get a valid improved decomposition.
    
    # Let me use a simpler approach: Strassen for 4x4x4 with schoolbook inner products.
    # 7 outer products, each 2x2x2 done with schoolbook (8 scalar mults) = 56 rank for 4x4x4.
    # Total: 56 + 56 + 16 = 128. Still better than 144.
    
    # Actually, let me just implement it carefully.
    
    col = 0
    
    # For the first 4 columns of B:
    # Use Strassen-based 4x4x4 decomposition
    
    # I'll implement a correct 49-rank decomposition using double Strassen.
    
    # Let me define the block indices more carefully.
    # For 4x4 matrix with row-major indexing (i*4+j):
    # A11 (top-left 2x2): indices 0,1,4,5
    # A12 (top-right 2x2): indices 2,3,6,7
    # A21 (bottom-left 2x2): indices 8,9,12,13
    # A22 (bottom-right 2x2): indices 10,11,14,15
    
    # For Strassen 4x4x4, the 7 products are combinations of these blocks.
    # Each product is a 2x2 matrix, computed via Strassen 2x2x2 (7 scalar mults).
    
    # Let me define a function to add a 2x2x2 Strassen decomposition where the
    # input 2x2 matrices are linear combinations of 4x4 block elements.
    
    def add_virtual_strassen_2x2(a_coeffs, b_coeffs, c_target, c_coeff, col_start):
        """
        a_coeffs: dict mapping (local_i, local_j) to list of (global_idx, coeff)
                  defining the virtual 2x2 A matrix
        b_coeffs: dict mapping (local_i, local_j) to list of (global_idx, coeff)
                  defining the virtual 2x2 B matrix
        c_target: list of (global_idx, coeff) for where to add the 2x2 result
                  c_target should have 4 entries for C[0,0], C[0,1], C[1,0], C[1,1]
        c_coeff: coefficient to multiply the entire 2x2 result by
        col_start: starting column
        Returns: new column
        """
        c = col_start
        
        # Build the virtual 2x2 matrices
        # VA[i,j] = sum of coeff * A[global_idx]
        # VB[i,j] = sum of coeff * B[global_idx]
        
        # Strassen 2x2x2:
        # S1 = (VA[0,0]+VA[1,1])*(VB[0,0]+VB[1,1])
        # S2 = (VA[1,0]+VA[1,1])*VB[0,0]
        # S3 = VA[0,0]*(VB[1,0]-VB[1,1])
        # S4 = VA[1,1]*(VB[0,1]-VB[0,0])
        # S5 = (VA[0,0]+VA[0,1])*VB[1,1]
        # S6 = (VA[1,0]-VA[0,0])*(VB[0,0]+VB[0,1])
        # S7 = (VA[0,1]-VA[1,1])*(VB[1,0]+VB[1,1])
        
        # VM[0,0] = S1 + S4 - S5 + S7
        # VM[0,1] = S3 + S5
        # VM[1,0] = S2 + S4
        # VM[1,1] = S1 - S2 + S3 + S6
        
        # For each Strassen product, we need to add contributions to U, V, W.
        
        # S1 = (VA[0,0]+VA[1,1])*(VB[0,0]+VB[1,1])
        # U contributions: all elements in VA[0,0] and VA[1,1] with +1
        # V contributions: all elements in VB[0,0] and VB[1,1] with +1
        # W contributions: VM[0,0] and VM[1,1] each get +1 from S1
        
        # This is getting very complex. Let me simplify by using a different strategy.
        
        return c
    
    # Given the extreme complexity of implementing double Strassen correctly by hand,
    # let me use a simpler but still improved approach:
    # Use Strassen for the outer 2x2 block multiplication, and schoolbook for inner.
    # Schoolbook 2x2x2 requires 8 scalar multiplications (not 4, as I mistakenly thought).
    # Wait, no. Schoolbook for 2x2x2: C[i,j] = sum_k A[i,k]*B[k,j]
    # That's 2*2*2 = 8 scalar multiplications total.
    
    # So 7 outer * 8 inner = 56 rank for 4x4x4.
    # Total: 56 + 56 + 16 = 128 for 4x4x9.
    
    # Let me implement this.
    
    rank = 128
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_schoolbook_2x2(a_base, b_base, c_base, col_start):
        """
        Add schoolbook 2x2x2 decomposition.
        a_base: base index for 2x2 A matrix (elements at a_base+0, a_base+1, a_base+2, a_base+3)
                where (0,0)->+0, (0,1)->+1, (1,0)->+2, (1,1)->+3
        b_base: base index for 2x2 B matrix
        c_base: base index for 2x2 C matrix
        col_start: starting column
        Returns: new column
        """
        c = col_start
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        
        # Each term is a scalar multiplication.
        # A[0,0]*B[0,0] contributes to C[0,0]
        U[a_base + 0][c] = 1
        V[b_base + 0][c] = 1
        W[c_base + 0][c] = 1
        c += 1
        
        # A[0,1]*B[1,0] contributes to C[0,0]
        U[a_base + 1][c] = 1
        V[b_base + 2][c] = 1
        W[c_base + 0][c] = 1
        c += 1
        
        # A[0,0]*B[0,1] contributes to C[0,1]
        U[a_base + 0][c] = 1
        V[b_base + 1][c] = 1
        W[c_base + 1][c] = 1
        c += 1
        
        # A[0,1]*B[1,1] contributes to C[0,1]
        U[a_base + 1][c] = 1
        V[b_base + 3][c] = 1
        W[c_base + 1][c] = 1
        c += 1
        
        # A[1,0]*B[0,0] contributes to C[1,0]
        U[a_base + 2][c] = 1
        V[b_base + 0][c] = 1
        W[c_base + 2][c] = 1
        c += 1
        
        # A[1,1]*B[1,0] contributes to C[1,0]
        U[a_base + 3][c] = 1
        V[b_base + 2][c] = 1
        W[c_base + 2][c] = 1
        c += 1
        
        # A[1,0]*B[0,1] contributes to C[1,1]
        U[a_base + 2][c] = 1
        V[b_base + 1][c] = 1
        W[c_base + 3][c] = 1
        c += 1
        
        # A[1,1]*B[1,1] contributes to C[1,1]
        U[a_base + 3][c] = 1
        V[b_base + 3][c] = 1
        W[c_base + 3][c] = 1
        c += 1
        
        return c
    
    # Now for 4x4x4 with Strassen outer, schoolbook inner:
    # The 7 Strassen products are linear combinations of 2x2 blocks.
    # For each product, I need to form virtual 2x2 matrices and compute schoolbook.
    
    # This is still complex. Let me just use full schoolbook for now to ensure correctness.
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

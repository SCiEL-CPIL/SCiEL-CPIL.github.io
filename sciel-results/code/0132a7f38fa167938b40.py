"""Matrix multiplication 4x4 by 4x8 rank-minimal search."""
import time
import itertools
import random
from fractions import Fraction

import numpy as np

DIMENSIONS = [4, 4, 8]
n, m, p = DIMENSIONS

def _build_tensor():
    """Build the matrix multiplication tensor as a 3D array."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p = 128."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, T, tol=1e-9):
    """Check if U, V, W decompose T."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    T_rec = np.einsum('ic,jc,kc->ijk', Ua, Va, Wa)
    return np.allclose(T_rec, T, atol=tol)

def _verify_exact_fraction(U, V, W, T):
    """Verify using exact Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i][c]) * Fraction(V[j][c]) * Fraction(W[k][c])
                if s != Fraction(T[i, j, k]):
                    return False
    return True

def _strassen_2x2x2():
    """
    Returns the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U, V, W are lists of lists of Fractions.
    U shape: (4, 7) corresponding to A[0,0], A[0,1], A[1,0], A[1,1]
    V shape: (4, 7) corresponding to B[0,0], B[0,1], B[1,0], B[1,1]
    W shape: (4, 7) corresponding to C[0,0], C[0,1], C[1,0], C[1,1]
    """
    # Indices for 2x2 matrix elements:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    # Strassen Multiplications:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - A10)  -- Wait, standard is M4 = A11 * (B10 - B00)? No.
    # Let's use the standard definition:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = [
        [1, 0, 1, 0, 1, 0, 0], # A00
        [0, 0, 0, 0, 1, 0, 1], # A01
        [0, 1, 0, 0, 0, -1, 0],# A10
        [1, 1, 0, 1, 0, 0, -1] # A11
    ]
    V = [
        [1, 1, 0, 1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0], # B01
        [0, 0, 0, -1, 0, 0, 1],# B10
        [1, 0, -1, 0, 1, 0, 1] # B11
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Convert to Fractions
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    return U_frac, V_frac, W_frac

def _combine_decompositions(decomp1, decomp2):
    """
    Combines two decompositions for the same A dimensions but disjoint B and C dimensions.
    decomp1: U1 (n*m, r1), V1 (m*p1, r1), W1 (p1*n, r1)
    decomp2: U2 (n*m, r2), V2 (m*p2, r2), W2 (p2*n, r2)
    Returns U, V, W for (n*m, r1+r2), (m*(p1+p2), r1+r2), (n*(p1+p2), r1+r2)
    Note: The indices for V and W need to be offset.
    """
    U1, V1, W1 = decomp1
    U2, V2, W2 = decomp2
    
    r1 = len(U1[0])
    r2 = len(U2[0])
    total_rank = r1 + r2
    
    # U is concatenated horizontally (same rows, more columns)
    U = [row1 + row2 for row1, row2 in zip(U1, U2)]
    
    # V needs to be expanded. V1 corresponds to B cols 0..p1-1, V2 to p1..p1+p2-1
    # V1 shape: (m*p1, r1). We need to pad V1 with zeros to (m*(p1+p2), r1)
    # V2 shape: (m*p2, r2). We need to pad V2 with zeros to (m*(p1+p2), r2)
    # Then concatenate horizontally.
    
    m = 4
    p1 = len(V1) // m
    p2 = len(V2) // m
    
    V_new = []
    # First p1 rows come from V1
    for i in range(p1):
        # V1 row i corresponds to B[i, 0..m-1]
        # We need to place this in the first block of columns for V_new
        row_v1 = V1[i*m:(i+1)*m]
        # Pad with zeros for the second block of B columns
        row_v1_padded = row_v1 + [0]*r2
        V_new.append(row_v1_padded)
        
    # Next p2 rows come from V2
    for i in range(p2):
        # V2 row i corresponds to B[i, 0..m-1] in its local frame
        # In the global frame, these are columns p1..p1+p2-1
        row_v2 = V2[i*m:(i+1)*m]
        # Pad with zeros for the first block of B columns
        row_v2_padded = [0]*r1 + row_v2
        V_new.append(row_v2_padded)
        
    # W needs to be expanded. W1 corresponds to C cols 0..p1-1, W2 to p1..p1+p2-1
    # W1 shape: (p1*n, r1). We need to pad W1 with zeros to (n*(p1+p2), r1)
    # W2 shape: (p2*n, r2). We need to pad W2 with zeros to (n*(p1+p2), r2)
    
    # W index is k*n + i.
    # W1 has indices for k in 0..p1-1.
    # W2 has indices for k in 0..p2-1 (local), which map to k+p1 in global.
    
    W_new = []
    # We need to interleave or block?
    # The tensor index for W is k*n + i.
    # If we split C into left (cols 0..p1-1) and right (cols p1..p1+p2-1),
    # the rows of W are ordered by k then i.
    # So rows 0..p1*n-1 correspond to left C.
    # Rows p1*n..(p1+p2)*n-1 correspond to right C.
    
    # So we just concatenate W1 and W2 vertically?
    # No, W is (p*n, rank).
    # W1 is (p1*n, r1).
    # W2 is (p2*n, r2).
    # We need W_new to be ((p1+p2)*n, r1+r2).
    
    # For the first p1*n rows, we take W1 and pad with 0s for the r2 columns.
    for i in range(p1 * n):
        row_w1 = W1[i]
        row_w1_padded = row_w1 + [0]*r2
        W_new.append(row_w1_padded)
        
    # For the next p2*n rows, we take W2 and pad with 0s for the r1 columns.
    for i in range(p2 * n):
        row_w2 = W2[i]
        row_w2_padded = [0]*r1 + row_w2
        W_new.append(row_w2_padded)
        
    return U, V_new, W_new

def _strassen_4x4x4():
    """
    Constructs a rank-49 decomposition for 4x4x4 matrix multiplication
    by applying Strassen's algorithm at the block level.
    """
    # Get the base 2x2x2 decomposition
    U_base, V_base, W_base = _strassen_2x2x2()
    
    # We treat 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A B
    
    # The block multiplication C = A B is given by:
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # This is not a single Strassen step. It's 8 multiplications of 2x2 blocks.
    # However, we can use the Strassen decomposition for the *block* multiplication.
    # Let X = A00, Y = A01, Z = A10, W_blk = A11.
    # Let P = B00, Q = B01, R = B10, S = B11.
    # We want to compute the 2x2 matrix of block products.
    # This is equivalent to multiplying a 2x2 matrix of 2x2 matrices by another 2x2 matrix of 2x2 matrices.
    # The tensor for this "block" multiplication is the same structure as 2x2x2, but the "scalars" are 2x2 matrices.
    # So we can use the rank-7 Strassen decomposition for the block structure.
    # Each "scalar" multiplication in the Strassen formula becomes a 2x2x2 multiplication, which has rank 7.
    # So total rank = 7 * 7 = 49.
    
    # We need to map the indices.
    # A indices: 0..15.
    # A00: 0,1,4,5 (indices i*m+j in 4x4? No, 2x2 blocks of 2x2 elements)
    # Let's define the mapping from 4x4 indices to block indices.
    # A is 4x4.
    # A[0,0], A[0,1] -> A00
    # A[0,2], A[0,3] -> A01
    # A[1,0], A[1,1] -> A00
    # A[1,2], A[1,3] -> A01
    # A[2,0], A[2,1] -> A10
    # A[2,2], A[2,3] -> A11
    # A[3,0], A[3,1] -> A10
    # A[3,2], A[3,3] -> A11
    
    # U_base has 4 rows (for A00, A01, A10, A11) and 7 columns.
    # We need to expand each row of U_base into 4 rows (for the 4 elements of the 2x2 block).
    # The values in U_base apply to the entire block.
    
    U_4x4 = []
    for i in range(4): # 4 blocks
        # Each block has 4 elements (2x2)
        # The coefficient for this block is U_base[i]
        for j in range(4): # 4 elements in the block
            # The element index in the 4x4 matrix
            # Block 0 (A00): elements 0,1,4,5
            # Block 1 (A01): elements 2,3,6,7
            # Block 2 (A10): elements 8,9,12,13
            # Block 3 (A11): elements 10,11,14,15
            # Wait, the indices in the tensor are i*m+j.
            # For A (4x4), index is i*4+j.
            # Block (bi, bj) contains elements (2bi, 2bj), (2bi, 2bj+1), (2bi+1, 2bj), (2bi+1, 2bj+1).
            # Block 0 (bi=0, bj=0): 0, 1, 4, 5
            # Block 1 (bi=0, bj=1): 2, 3, 6, 7
            # Block 2 (bi=1, bj=0): 8, 9, 12, 13
            # Block 3 (bi=1, bj=1): 10, 11, 14, 15
            
            # We need to construct the 16 rows of U_4x4.
            # For each of the 7 Strassen terms, we have a coefficient.
            # This coefficient applies to all 4 elements of the block.
            pass
            
    # This manual expansion is error-prone. Let's use a helper to map block indices to element indices.
    
    def block_to_elements(block_idx):
        """Returns the 4 element indices in the 4x4 matrix for a given block index (0-3)."""
        bi = block_idx // 2
        bj = block_idx % 2
        # Elements are at (2bi, 2bj), (2bi, 2bj+1), (2bi+1, 2bj), (2bi+1, 2bj+1)
        # Index = i*4 + j
        return [
            (2*bi)*4 + (2*bj),
            (2*bi)*4 + (2*bj+1),
            (2*bi+1)*4 + (2*bj),
            (2*bi+1)*4 + (2*bj+1)
        ]
    
    # U_4x4 will have 16 rows and 49 columns.
    # The 49 columns come from 7 block-terms * 7 element-terms.
    # Let c = cb * 7 + ce.
    
    U_4x4 = [[0]*49 for _ in range(16)]
    V_4x4 = [[0]*49 for _ in range(16)]
    W_4x4 = [[0]*49 for _ in range(16)]
    
    for cb in range(7): # Block Strassen term
        for ce in range(7): # Element Strassen term
            col = cb * 7 + ce
            
            # The coefficient for this combined term in the U, V, W matrices
            # is the product of the block coefficient and the element coefficient.
            
            # U coefficient: U_base[bi, cb] * U_base[ei, ce]
            # Where bi is the block index for A, ei is the element index within the block.
            
            # This logic is slightly flawed. The decomposition is:
            # T = sum_cb sum_ce (U_block[bi, cb] * U_elem[ei, ce]) ...
            # This is a Kronecker product of the decompositions.
            
            # Let's iterate over the 16 rows of U (which correspond to A elements).
            for row in range(16):
                # Find which block this row belongs to
                # Inverse of block_to_elements
                bi = row // 4 # This is wrong. 0,1,4,5 are block 0.
                # Let's just precompute the mapping.
                pass
                
    # Precompute mapping from element index to block index
    elem_to_block = {}
    for b in range(4):
        elems = block_to_elements(b)
        for e in elems:
            elem_to_block[e] = b
            
    # Precompute mapping from element index to local element index (0-3)
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    elem_to_local = {}
    for b in range(4):
        elems = block_to_elements(b)
        for i, e in enumerate(elems):
            elem_to_local[e] = i
            
    # Now fill U_4x4
    for row in range(16):
        bi = elem_to_block[row]
        ei = elem_to_local[row]
        
        for cb in range(7):
            for ce in range(7):
                col = cb * 7 + ce
                # U_4x4[row, col] = U_base[bi, cb] * U_base[ei, ce]
                U_4x4[row][col] = U_base[bi][cb] * U_base[ei][ce]
                
    # Similarly for V and W
    for row in range(16):
        bi = elem_to_block[row]
        ei = elem_to_local[row]
        
        for cb in range(7):
            for ce in range(7):
                col = cb * 7 + ce
                V_4x4[row][col] = V_base[bi][cb] * V_base[ei][ce]
                W_4x4[row][col] = W_base[bi][cb] * W_base[ei][ce]
                
    return U_4x4, V_4x4, W_4x4

def _strassen_4x4x8():
    """
    Constructs a rank-98 decomposition for 4x4x8 matrix multiplication
    by splitting the 8 columns into two 4-column blocks.
    """
    U_44, V_44, W_44 = _strassen_4x4x4()
    
    # We have two independent 4x4x4 multiplications.
    # Left: A * B_left -> C_left
    # Right: A * B_right -> C_right
    
    # U is the same for both.
    # V is split: V_left (16x49), V_right (16x49)
    # W is split: W_left (16x49), W_right (16x49)
    
    # We can just use the same U_44 for both.
    # V_44 corresponds to the B part.
    # W_44 corresponds to the C part.
    
    # The combined V will have 32 rows (16 for left, 16 for right).
    # The combined W will have 32 rows (16 for left, 16 for right).
    # The combined U will have 16 rows.
    
    # Rank = 49 + 49 = 98.
    
    U_combined = []
    for row in U_44:
        # Duplicate the row for the right block
        U_combined.append(row + row)
        
    V_combined = []
    # First 16 rows from V_44 (padded with 0s for right block)
    for row in V_44:
        V_combined.append(row + [0]*49)
    # Next 16 rows from V_44 (padded with 0s for left block)
    for row in V_44:
        V_combined.append([0]*49 + row)
        
    W_combined = []
    # First 16 rows from W_44 (padded with 0s for right block)
    for row in W_44:
        W_combined.append(row + [0]*49)
    # Next 16 rows from W_44 (padded with 0s for left block)
    for row in W_44:
        W_combined.append([0]*49 + row)
        
    return U_combined, V_combined, W_combined

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Try the Strassen-based decomposition
    try:
        U, V, W = _strassen_4x4x8()
        T = _build_tensor()
        
        # Verify
        if _verify(U, V, W, T):
            # Convert to Fractions just in case, though they are already Fractions
            U_frac = [[Fraction(x) for x in row] for row in U]
            V_frac = [[Fraction(x) for x in row] for row in V]
            W_frac = [[Fraction(x) for x in row] for row in W]
            
            # Final exact check
            if _verify_exact_fraction(U_frac, V_frac, W_frac, T):
                return {"U": U_frac, "V": V_frac, "W": W_frac}
    except Exception as e:
        print(f"Strassen construction failed: {e}")
        
    # Fallback to schoolbook
    return _schoolbook()

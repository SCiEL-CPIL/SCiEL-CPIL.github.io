import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 4x4 matrix multiplication using the rank-49 Strassen algorithm.
    The decomposition is the tensor product of two 2x2 Strassen decompositions.
    """
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2 matrix multiplication
    # U, V, W are 4x7 matrices (flattened 2x2)
    # Indices: 0:00, 1:01, 2:10, 3:11
    
    # U factors for A
    # m1: A00+A11 -> [1,0,0,1]
    # m2: A10+A11 -> [0,0,1,1]
    # m3: A00 -> [1,0,0,0]
    # m4: A11 -> [0,0,0,1]
    # m5: A00+A01 -> [1,1,0,0]
    # m6: A10-A00 -> [-1,0,1,0]
    # m7: A01-A11 -> [0,1,-1,0] -- Wait, A01 is index 1, A11 is index 3.
    # Let's re-verify Strassen formulas.
    
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Flattened indices for 2x2:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    U_2x2 = np.array([
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ], dtype=float)
    
    V_2x2 = np.array([
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 1, 0, -1], # M3: B01 - B11
        [0, 0, 1, -1], # M4: B10 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B01
        [0, 0, 1, 1]   # M7: B10 + B11
    ], dtype=float)
    
    W_2x2 = np.array([
        [1, 0, 0, 1],  # M1: C00 += 1, C11 += 1
        [0, 0, 1, -1], # M2: C10 += 1, C11 += -1
        [0, 1, 0, 1],  # M3: C01 += 1, C11 += 1
        [1, 0, 1, 0],  # M4: C00 += 1, C10 += 1
        [-1, 1, 0, 0], # M5: C00 += -1, C01 += 1
        [0, 0, 0, 1],  # M6: C11 += 1
        [1, 0, 0, 0]   # M7: C00 += 1
    ], dtype=float)
    
    # For 4x4, we treat A, B, C as 2x2 blocks of 2x2 matrices.
    # The decomposition is the Kronecker product of the 2x2 Strassen factors.
    # Rank = 7 * 7 = 49.
    
    # The Kronecker product of U_2x2 (4x7) with itself (4x7) gives a (16x49) matrix.
    # However, we need to be careful with the ordering of indices.
    # The block structure:
    # A = [[A00, A01], [A10, A11]]
    # Flattened A: [A00_flat, A01_flat, A10_flat, A11_flat]
    # Where Aij_flat is the 4 entries of the 2x2 block.
    
    # The Kronecker product U_2x2 \otimes U_2x2 will map the 16 entries of A to the 49 terms.
    # Let's verify the index mapping.
    # If we use np.kron(U_2x2, U_2x2), the row index is 4*i + j, where i is the block row/col index in the outer 2x2, and j is the inner 2x2 index.
    # This matches the flattening order if we flatten block-major: [Block00, Block01, Block10, Block11].
    
    U = np.kron(U_2x2, U_2x2)
    V = np.kron(V_2x2, V_2x2)
    W = np.kron(W_2x2, W_2x2)
    
    # The problem requires W[k*n+i, r] convention.
    # np.kron(W_2x2, W_2x2) produces rows indexed by 4*outer_idx + inner_idx.
    # outer_idx corresponds to the block (br, bc) flattened as 2*br + bc?
    # In 2x2, flatten order is 00, 01, 10, 11 -> indices 0, 1, 2, 3.
    # So outer_idx is 2*br + bc.
    # inner_idx is 2*lr + lc.
    # Total index = 4*(2*br + bc) + (2*lr + lc) = 8*br + 4*bc + 2*lr + lc.
    # This is the row-major flattening of the 4x4 matrix: row = 2*br + lr, col = 2*bc + lc.
    # Row index in 4x4: r = 2*br + lr. Col index: c = 2*bc + lc.
    # Row-major index: r*4 + c = (2*br + lr)*4 + (2*bc + lc) = 8*br + 4*lr + 2*bc + lc.
    # This is DIFFERENT from 8*br + 4*bc + 2*lr + lc.
    # np.kron uses the order: (row1, col1) then (row2, col2).
    # So np.kron(A, B) has index (i1, j1, i2, j2) mapped to (4*i1 + i2)*4 + (4*j1 + j2)? No.
    # np.kron(A, B) where A is (r1, c1) and B is (r2, c2).
    # Result is (r1*r2, c1*c2).
    # Element (i, j) of result is A[i//r2, j//c2] * B[i%r2, j%c2].
    # So row index i = i1*r2 + i2. Col index j = j1*c2 + j2.
    # Flattened row-major index: i*ncols + j = (i1*r2 + i2)*(c1*c2) + (j1*c2 + j2).
    # For 2x2, r2=2, c2=2, c1=2.
    # Index = (2*i1 + i2)*4 + (2*j1 + j2) = 8*i1 + 4*i2 + 4*j1 + j2.
    
    # We need W[k*n+i, r] which is Col-Major flattening?
    # "W[k*n+i,r]" where k is col, i is row?
    # If k=col, i=row, index = col*4 + row.
    # In our np.kron result, the flattened index is Row-Major: row*4 + col.
    # row = 2*i1 + i2. col = 2*j1 + j2.
    # Row-Major Index = (2*i1 + i2)*4 + (2*j1 + j2) = 8*i1 + 4*i2 + 4*j1 + j2.
    # Col-Major Index = (2*j1 + j2)*4 + (2*i1 + i2) = 8*j1 + 4*j2 + 4*i1 + i2.
    
    # The prompt says "W[k*n+i,r]".
    # Usually k is the column index in the flattened vector if it's column-major?
    # Or is it just the index in the vector of length p*n?
    # If the vector is flattened column-major, index = col*n + row.
    # If the vector is flattened row-major, index = row*p + col.
    # The prompt says "W[k*n+i,r]". This looks like index = k*n + i.
    # If n=4, index = 4*k + i.
    # If k is col and i is row, this is Column-Major.
    # If k is row and i is col, this is Row-Major.
    # Standard trace convention for matrix multiplication tensor often uses $C_{ij} = \sum A_{ik} B_{kj}$.
    # The tensor index is often $(i, j, k)$ or similar.
    # Let's assume the evaluator expects the vector to be ordered such that index = col*4 + row (Column-Major) or row*4 + col (Row-Major).
    # The previous code used `W[gc * n + gr]`.
    # `gc` is col, `gr` is row.
    # So index = col*4 + row. This is Column-Major.
    
    # My np.kron produces Row-Major flattening of the 4x4 matrix.
    # I need to permute the rows of W to match Column-Major order.
    
    # Let's create the permutation.
    # Current row index in W_kron: idx_rm = 8*i1 + 4*i2 + 4*j1 + j2.
    # Here i1, i2 are row components, j1, j2 are col components.
    # Row = 2*i1 + i2. Col = 2*j1 + j2.
    # Target index (Col-Major): idx_cm = 4*Col + Row = 4*(2*j1 + j2) + (2*i1 + i2) = 8*j1 + 4*j2 + 4*i1 + i2.
    
    # We need to reorder the rows of W_kron.
    # For each row r in 0..15, we calculate its (i1, i2, j1, j2) components from the Row-Major index.
    # Then we calculate the Target Index.
    # Then we place the row at the Target Index.
    
    perm = np.zeros(16, dtype=int)
    for idx_rm in range(16):
        # Decompose idx_rm into i1, i2, j1, j2
        # idx_rm = (2*i1 + i2)*4 + (2*j1 + j2)
        row_part = idx_rm // 4  # 2*i1 + i2
        col_part = idx_rm % 4   # 2*j1 + j2
        
        i1 = row_part // 2
        i2 = row_part % 2
        j1 = col_part // 2
        j2 = col_part % 2
        
        # Calculate Target Index (Col-Major)
        target_idx = 4 * (2*j1 + j2) + (2*i1 + i2)
        
        # The row at idx_rm in W_kron should go to target_idx in W_final
        # So W_final[target_idx] = W_kron[idx_rm]
        # This means perm[target_idx] = idx_rm
        perm[target_idx] = idx_rm
        
    W = W[perm, :]
    
    # Convert to lists of lists as required by the interface
    # The interface expects {'U': U, 'V': V, 'W': W}
    # U, V, W are lists of lists? Or NumPy arrays?
    # "factor matrices of shapes (n*m, rank)..."
    # The previous code returned lists. I will return lists to be safe.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

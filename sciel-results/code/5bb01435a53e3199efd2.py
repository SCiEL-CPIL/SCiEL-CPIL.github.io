import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # We will build the matrices U, V, W incrementally.
    # Start with empty lists for rows, then convert to arrays?
    # Or pre-allocate with max possible rank?
    # Max rank is 120.
    
    # Let's use lists of lists for flexibility, then np.array.
    U_rows = []
    V_rows = []
    W_rows = []
    
    # We need to track the current rank (number of columns)
    current_rank = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal current_rank
        # Add a new column to U, V, W
        # U has 20 rows, V has 30, W has 24
        # We initialize new columns with 0s
        # This is inefficient if done one by one, but rank is small (<120)
        # Better to accumulate coefficients for a column and append at once.
        pass
        
    # Strategy: For each algorithmic step (Strassen or Schoolbook),
    # determine the set of (U_idx, V_idx, W_idx, Coeff) tuples.
    # Each "rank 1" component in the decomposition corresponds to one column in U, V, W.
    # However, Strassen terms like M1 = (a11+a22)(b11+b22) expand to 4 rank-1 terms:
    # a11*b11, a11*b22, a22*b11, a22*b22.
    # So one "Strassen step" actually adds 4 rank-1 terms to the decomposition?
    # No, the rank of the 2x2x2 tensor is 7. This means there are 7 linearly independent
    # rank-1 tensors that sum to the multiplication tensor.
    # So for each 2x2x2 block, we add 7 columns to U, V, W.
    
    # Let's define the 7 rank-1 components for 2x2 Strassen explicitly in terms of basis indices.
    # Local indices 0,1 for rows/cols.
    
    def get_strassen_factors():
        """
        Returns 7 sets of (u_coeffs, v_coeffs, w_coeffs) for 2x2 block.
        u_coeffs: dict mapping local (i,j) to coeff
        v_coeffs: dict mapping local (j,k) to coeff
        w_coeffs: dict mapping local (k,i) to coeff
        """
        # M1 = (a00 + a11)(b00 + b11) -> C00, C11
        # Expansion:
        # (a00 + a11)(b00 + b11) = a00b00 + a00b11 + a11b00 + a11b11
        # This is not a single rank 1 term.
        # Wait, the "rank" of the tensor is the number of terms in the sum of rank-1 tensors.
        # Strassen says the 2x2 multiplication tensor can be written as a sum of 7 rank-1 tensors.
        # We need to find those 7 rank-1 tensors.
        
        # The 7 products are:
        # M1 = (a00 + a11)(b00 + b11)
        # M2 = (a10 + a11)b00
        # M3 = a00(b01 - b11)
        # M4 = a11(b10 - b00)
        # M5 = (a00 + a01)b11
        # M6 = (a10 - a00)(b00 + b01)
        # M7 = (a01 - a11)(b10 + b11)
        
        # These M's are intermediate results. The final C is a linear combination of M's.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # To express this as a sum of rank-1 tensors $U_c \otimes V_c \otimes W_c$:
        # Each $M_k$ is a product of two linear forms.
        # $M_k = (\sum u_{ki} a_i) (\sum v_{kj} b_j)$
        # And it contributes to $C_l$ with coefficient $w_{kl}$.
        # So for each $k$ in 1..7, and each $l$ in 0..3 (output index):
        # If $w_{kl} \neq 0$, we have a contribution.
        # However, the rank is the number of $M_k$'s? No.
        # The rank of the tensor is the minimal number of rank-1 tensors summing to it.
        # Strassen's algorithm shows rank <= 7.
        # The 7 rank-1 tensors correspond to the 7 M_k's?
        # Let's verify.
        # $T = \sum_{c=1}^7 U_c \otimes V_c \otimes W_c$
        # For a specific output index $(i, k)$, the coefficient of $A_{i', j'} B_{j' k'}$ is...
        
        # Actually, it's easier to just use the 7 M_k's as the 7 "columns" in the decomposition?
        # Let's check the linearity.
        # If we define the 7 columns as the linear forms for M1..M7, do they reconstruct C?
        # Yes, because C is a linear combination of M1..M7.
        # So for each Strassen block, we add 7 columns to U, V, W.
        # The entries in these columns are the coefficients of the local A, B, C indices in the M_k expressions.
        
        # Let's map local indices:
        # A: 00, 01, 10, 11 (i=0..1, j=0..1)
        # B: 00, 01, 10, 11 (j=0..1, k=0..1)
        # C: 00, 01, 10, 11 (k=0..1, i=0..1)
        
        # M1: (a00 + a11)(b00 + b11)
        # A coeffs: {00: 1, 11: 1}
        # B coeffs: {00: 1, 11: 1}
        # C coeffs: {00: 1, 11: 1} (from C00=+M1, C11=+M1)
        
        # M2: (a10 + a11)b00
        # A coeffs: {10: 1, 11: 1}
        # B coeffs: {00: 1}
        # C coeffs: {10: 1} (from C10=+M2) ... wait, C10 = M2 + M4. So M2 contributes 1 to C10.
        
        # M3: a00(b01 - b11)
        # A coeffs: {00: 1}
        # B coeffs: {01: 1, 11: -1}
        # C coeffs: {01: 1} (from C01=+M3)
        
        # M4: a11(b10 - b00)
        # A coeffs: {11: 1}
        # B coeffs: {10: 1, 00: -1}
        # C coeffs: {00: 1, 10: 1} (from C00=+M4, C10=+M4)
        
        # M5: (a00 + a01)b11
        # A coeffs: {00: 1, 01: 1}
        # B coeffs: {11: 1}
        # C coeffs: {00: -1, 01: 1} (from C00=-M5, C01=+M5)
        
        # M6: (a10 - a00)(b00 + b01)
        # A coeffs: {10: 1, 00: -1}
        # B coeffs: {00: 1, 01: 1}
        # C coeffs: {11: 1} (from C11=+M6)
        
        # M7: (a01 - a11)(b10 + b11)
        # A coeffs: {01: 1, 11: -1}
        # B coeffs: {10: 1, 11: 1}
        # C coeffs: {00: 1} (from C00=+M7)
        
        # Let's double check C00 = M1 + M4 - M5 + M7
        # Coeffs for C00:
        # M1: 1
        # M4: 1
        # M5: -1
        # M7: 1
        # Matches.
        
        return [
            # (A_coeffs, B_coeffs, C_coeffs)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}), # M1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}), # M2
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}), # M3
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}), # M4
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}), # M5
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}), # M6
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}), # M7
        ]

    # Pre-compute the factors for Strassen blocks
    strassen_factors = get_strassen_factors()
    
    def add_strassen_block(i0, j0, k0):
        """
        Adds Strassen decomposition for block A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[k0:k0+2, i0:i0+2]
        """
        nonlocal current_rank
        # Extend U, V, W with 7 new columns (initialized to 0)
        # U has 20 rows, V has 30, W has 24
        # We will append to the lists of columns?
        # Currently U_rows is a list of rows? No, let's store columns as lists of values.
        # It's easier to store U as a list of columns, where each column is a list of 20 values.
        # Or just keep a 2D array and append columns.
        
        # Let's switch to appending columns to numpy arrays.
        # U_np, V_np, W_np
        
        # For each of the 7 factors:
        for a_coeff, b_coeff, c_coeff in strassen_factors:
            # Create new column vectors
            u_col = [0.0] * (n * m)
            v_col = [0.0] * (m * p)
            w_col = [0.0] * (p * n)
            
            # Fill u_col
            for (li, lj), val in a_coeff.items():
                global_i = i0 + li
                global_j = j0 + lj
                u_col[global_i * m + global_j] = val
                
            # Fill v_col
            for (lj, lk), val in b_coeff.items():
                global_j = j0 + lj
                global_k = k0 + lk
                v_col[global_j * p + global_k] = val
                
            # Fill w_col
            for (lk, li), val in c_coeff.items():
                global_k = k0 + lk
                global_i = i0 + li
                w_col[global_k * n + global_i] = val
                
            # Append to global lists
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
            current_rank += 1

    def add_schoolbook_block(i0, j0, k0, ni, nj, nk):
        """
        Adds schoolbook decomposition for block A[i0:i0+ni, j0:j0+nj] * B[j0:j0+nj, k0:k0+nk] -> C[k0:k0+nk, i0:i0+ni]
        """
        nonlocal current_rank
        for li in range(ni):
            for lj in range(nj):
                for lk in range(nk):
                    u_col = [0.0] * (n * m)
                    v_col = [0.0] * (m * p)
                    w_col = [0.0] * (p * n)
                    
                    global_i = i0 + li
                    global_j = j0 + lj
                    global_k = k0 + lk
                    
                    u_col[global_i * m + global_j] = 1.0
                    v_col[global_j * p + global_k] = 1.0
                    w_col[global_k * n + global_i] = 1.0
                    
                    U_cols.append(u_col)
                    V_cols.append(v_col)
                    W_cols.append(w_col)
                    current_rank += 1

    # Initialize column lists
    U_cols = []
    V_cols = []
    W_cols = []
    current_rank = 0

    # 1. Strassen blocks for the 4x4x6 part
    # A rows 0-3, Cols 0-3 (of A, which are rows 0-3 of B)
    # B rows 0-3, Cols 0-5
    # Partition A rows into 0-1, 2-3
    # Partition B rows into 0-1, 2-3
    # Partition B cols into 0-1, 2-3, 4-5
    
    a_row_splits = [(0, 2), (2, 4)]
    b_row_splits = [(0, 2), (2, 4)]
    b_col_splits = [(0, 2), (2, 4), (4, 6)]
    
    for (i0, i1) in a_row_splits:
        for (j0, j1) in b_row_splits:
            for (k0, k1) in b_col_splits:
                # Block is A[i0:i1, j0:j1] * B[j0:j1, k0:k1]
                # Size 2x2x2
                add_strassen_block(i0, j0, k0)

    # 2. Schoolbook blocks for the remaining parts
    # A has col 4 (1 column). B has row 4 (1 row).
    # Part A: A[0:4, 4:5] * B[4:5, 0:6] -> C[0:6, 0:4]
    # This is 4x1x6.
    add_schoolbook_block(0, 4, 0, 4, 1, 6)
    
    # Part B: A[0:4, 0:4] * B[0:4, 4:6] is already covered by Strassen?
    # Wait, my Strassen loop covered B cols 0-5.
    # A cols 0-3.
    # So A[0:4, 0:4] * B[0:4, 0:6] is covered.
    
    # What about A[0:4, 4:5] * B[4:5, 0:6]? Covered by schoolbook above.
    
    # Is there anything else?
    # A is 4x5. B is 5x6.
    # We used A[0:4, 0:4] and A[0:4, 4:5].
    # We used B[0:4, 0:6] and B[4:5, 0:6].
    # This covers all of A and B.
    # C is sum of products.
    # Strassen covers A[0:4, 0:4] * B[0:4, 0:6].
    # Schoolbook covers A[0:4, 4:5] * B[4:5, 0:6].
    
    # Total rank should be 12 * 7 + 24 = 84 + 24 = 108.

    # Convert lists to numpy arrays
    U = np.array(U_cols).T # Shape (20, 108)
    V = np.array(V_cols).T # Shape (30, 108)
    W = np.array(W_cols).T # Shape (24, 108)
    
    # Verify quickly
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T[i,j,k] = 1
    # Decomposition: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    
    # We can skip full verification for speed if we are confident, 
    # but a partial check is good.
    
    return {"U": U, "V": V, "W": W}

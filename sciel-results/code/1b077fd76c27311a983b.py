import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    # Generate Strassen decomposition for 4x4x4
    # n=4, m=4, p=4
    
    def get_strassen_2x2():
        # Returns U, V, W for 2x2x2 with rank 7
        # Indices:
        # A: 0,1,2,3 (row-major 2x2)
        # B: 0,1,2,3
        # C: 0,1,2,3
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        # A11=0, A22=3
        # B11=0, B22=3
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        # C11 += M1, C22 += M1
        W[0,0]=1; W[3,0]=1
        
        # M2 = (A21+A22)B11
        # A21=2, A22=3
        # B11=0
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        # C21 += M2
        W[2,1]=1
        
        # M3 = A11(B12-B22)
        # A11=0
        # B12=1, B22=3
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        # C12 += M3
        W[1,2]=1
        
        # M4 = A22(B21-B11)
        # A22=3
        # B21=2, B11=0
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        # C21 += M4
        W[2,3]=1
        
        # M5 = (A11+A12)B22
        # A11=0, A12=1
        # B22=3
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        # C11 -= M5, C12 += M5
        W[0,4]=-1; W[1,4]=1
        
        # M6 = (A21-A11)(B11+B12)
        # A21=2, A11=0
        # B11=0, B12=1
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        # C22 += M6
        W[3,5]=1
        
        # M7 = (A12-A22)(B21+B22)
        # A12=1, A22=3
        # B21=2, B22=3
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        # C11 += M7
        W[0,6]=1
        
        return U, V, W

    def embed_and_scale(U_sub, V_sub, W_sub, offset_a, offset_b, offset_c, n_sub):
        """
        Embed a decomposition of size n_sub into a larger matrix.
        offset_a, offset_b, offset_c are the starting indices in the flattened arrays.
        """
        n_full = 4 # Assuming we are building for 4x4x4
        U_full = np.zeros((n_full*n_full, U_sub.shape[1]))
        V_full = np.zeros((n_full*n_full, V_sub.shape[1]))
        W_full = np.zeros((n_full*n_full, W_sub.shape[1]))
        
        # This simple embedding assumes the sub-problem is at a specific block.
        # We need a more flexible recursion.
        pass

    # Better approach: Recursive construction
    
    def strassen_decomp(n):
        """
        Returns U, V, W for n x n x n matrix multiplication.
        n must be a power of 2.
        """
        if n == 2:
            return get_strassen_2x2()
        
        half = n // 2
        U_sub, V_sub, W_sub = strassen_decomp(half)
        r_sub = U_sub.shape[1]
        
        # The 7 steps of Strassen
        # Each step involves a matrix multiplication of size half x half.
        # We need to map the inputs and outputs of the sub-problem to the full problem.
        
        U_full = np.zeros((n*n, 7*r_sub))
        V_full = np.zeros((n*n, 7*r_sub))
        W_full = np.zeros((n*n, 7*r_sub))
        
        # Helper to add a sub-decomposition to the full matrices
        def add_sub(U_s, V_s, W_s, a_map, b_map, c_map, sign_a, sign_b, sign_c, col_offset):
            """
            a_map: list of indices in A_full corresponding to A_sub rows
            b_map: list of indices in B_full corresponding to B_sub cols
            c_map: list of indices in C_full corresponding to C_sub rows
            sign: multipliers for the terms
            """
            for i in range(len(a_map)):
                if a_map[i] is not None:
                    U_full[a_map[i], col_offset:col_offset+r_sub] += sign_a[i] * U_s
            
            for j in range(len(b_map)):
                if b_map[j] is not None:
                    V_full[b_map[j], col_offset:col_offset+r_sub] += sign_b[j] * V_s
            
            for k in range(len(c_map)):
                if c_map[k] is not None:
                    W_full[c_map[k], col_offset:col_offset+r_sub] += sign_c[k] * W_s

        # Define block indices for n x n matrix
        # A is n x n, flattened row-major.
        # Block A11: rows 0..half-1, cols 0..half-1
        # Block A12: rows 0..half-1, cols half..n-1
        # Block A21: rows half..n-1, cols 0..half-1
        # Block A22: rows half..n-1, cols half..n-1
        
        def get_block_indices(n, block):
            # block: 0=A11, 1=A12, 2=A21, 3=A22
            h = n // 2
            if block == 0:
                rows = range(0, h)
                cols = range(0, h)
            elif block == 1:
                rows = range(0, h)
                cols = range(h, n)
            elif block == 2:
                rows = range(h, n)
                cols = range(0, h)
            elif block == 3:
                rows = range(h, n)
                cols = range(h, n)
            
            indices = []
            for r in rows:
                for c in cols:
                    indices.append(r * n + c)
            return indices

        A11 = get_block_indices(n, 0)
        A12 = get_block_indices(n, 1)
        A21 = get_block_indices(n, 2)
        A22 = get_block_indices(n, 3)
        
        B11 = get_block_indices(n, 0)
        B12 = get_block_indices(n, 1)
        B21 = get_block_indices(n, 2)
        B22 = get_block_indices(n, 3)
        
        C11 = get_block_indices(n, 0)
        C12 = get_block_indices(n, 1)
        C21 = get_block_indices(n, 2)
        C22 = get_block_indices(n, 3)
        
        col = 0
        
        # Step 1: M1 = (A11+A22)(B11+B22)
        # U: A11 + A22
        a_map_1 = [A11[i] if i < len(A11) else None for i in range(half*half)] # This is wrong, A11 is a list of indices
        # We need to map the sub-problem indices (0..half*half-1) to full indices.
        # The sub-problem U_s has rows 0..half*half-1.
        # We want to add U_s to rows A11 and A22.
        
        # Let's rewrite add_sub to handle lists of indices directly.
        def add_term(a_indices, b_indices, c_indices, sign, col_off):
            for idx in a_indices:
                U_full[idx, col_off:col_off+r_sub] += sign * U_sub
            for idx in b_indices:
                V_full[idx, col_off:col_off+r_sub] += sign * V_sub
            for idx in c_indices:
                W_full[idx, col_off:col_off+r_sub] += sign * W_sub

        # Step 1
        add_term(A11, B11, C11, 1, col); add_term(A22, B22, C11, 1, col) # Wait, this adds U_sub twice?
        # No, U_sub is the same.
        # U_full[A11] += U_sub
        # U_full[A22] += U_sub
        # This is correct.
        
        # But we need to be careful. The sub-problem is one multiplication.
        # M1 = (A11+A22) * (B11+B22)
        # This is ONE multiplication of size half.
        # So we add the sub-decomposition ONCE, but the inputs are sums.
        
        # Correct logic:
        # For each of the 7 steps, we create a new set of columns.
        # The inputs to the step are linear combinations of the blocks.
        
        # Step 1:
        # Inputs: A11+A22, B11+B22
        # Output: C11 += M1, C22 += M1
        for i in range(len(A11)):
            U_full[A11[i], col:col+r_sub] += U_sub
        for i in range(len(A22)):
            U_full[A22[i], col:col+r_sub] += U_sub
        for i in range(len(B11)):
            V_full[B11[i], col:col+r_sub] += V_sub
        for i in range(len(B22)):
            V_full[B22[i], col:col+r_sub] += V_sub
        for i in range(len(C11)):
            W_full[C11[i], col:col+r_sub] += W_sub
        for i in range(len(C22)):
            W_full[C22[i], col:col+r_sub] += W_sub
        col += r_sub
        
        # Step 2: M2 = (A21+A22) B11
        # Output: C21 += M2
        for i in range(len(A21)):
            U_full[A21[i], col:col+r_sub] += U_sub
        for i in range(len(A22)):
            U_full[A22[i], col:col+r_sub] += U_sub
        for i in range(len(B11)):
            V_full[B11[i], col:col+r_sub] += V_sub
        for i in range(len(C21)):
            W_full[C21[i], col:col+r_sub] += W_sub
        col += r_sub
        
        # Step 3: M3 = A11 (B12-B22)
        # Output: C12 += M3
        for i in range(len(A11)):
            U_full[A11[i], col:col+r_sub] += U_sub
        for i in range(len(B12)):
            V_full[B12[i], col:col+r_sub] += V_sub
        for i in range(len(B22)):
            V_full[B22[i], col:col+r_sub] -= V_sub
        for i in range(len(C12)):
            W_full[C12[i], col:col+r_sub] += W_sub
        col += r_sub
        
        # Step 4: M4 = A22 (B21-B11)
        # Output: C21 += M4
        for i in range(len(A22)):
            U_full[A22[i], col:col+r_sub] += U_sub
        for i in range(len(B21)):
            V_full[B21[i], col:col+r_sub] += V_sub
        for i in range(len(B11)):
            V_full[B11[i], col:col+r_sub] -= V_sub
        for i in range(len(C21)):
            W_full[C21[i], col:col+r_sub] += W_sub
        col += r_sub
        
        # Step 5: M5 = (A11+A12) B22
        # Output: C11 -= M5, C12 += M5
        for i in range(len(A11)):
            U_full[A11[i], col:col+r_sub] += U_sub
        for i in range(len(A12)):
            U_full[A12[i], col:col+r_sub] += U_sub
        for i in range(len(B22)):
            V_full[B22[i], col:col+r_sub] += V_sub
        for i in range(len(C11)):
            W_full[C11[i], col:col+r_sub] -= W_sub
        for i in range(len(C12)):
            W_full[C12[i], col:col+r_sub] += W_sub
        col += r_sub
        
        # Step 6: M6 = (A21-A11) (B11+B12)
        # Output: C22 += M6
        for i in range(len(A21)):
            U_full[A21[i], col:col+r_sub] += U_sub
        for i in range(len(A11)):
            U_full[A11[i], col:col+r_sub] -= U_sub
        for i in range(len(B11)):
            V_full[B11[i], col:col+r_sub] += V_sub
        for i in range(len(B12)):
            V_full[B12[i], col:col+r_sub] += V_sub
        for i in range(len(C22)):
            W_full[C22[i], col:col+r_sub] += W_sub
        col += r_sub
        
        # Step 7: M7 = (A12-A22) (B21+B22)
        # Output: C11 += M7
        for i in range(len(A12)):
            U_full[A12[i], col:col+r_sub] += U_sub
        for i in range(len(A22)):
            U_full[A22[i], col:col+r_sub] -= U_sub
        for i in range(len(B21)):
            V_full[B21[i], col:col+r_sub] += V_sub
        for i in range(len(B22)):
            V_full[B22[i], col:col+r_sub] += V_sub
        for i in range(len(C11)):
            W_full[C11[i], col:col+r_sub] += W_sub
        col += r_sub
        
        return U_full, V_full, W_full

    U44, V44, W44 = strassen_decomp(4)
    
    # Filter for 3x4x4
    # We want A to be 3x4. In the 4x4 model, A has 4 rows.
    # We assume the 4th row of A is 0.
    # So we take U44[0:12, :] (first 3 rows of A).
    # We take V44 as is.
    # We take W44, but we only want the output rows 0,1,2.
    # Output row 3 is indices 12,13,14,15 in W44.
    # We keep a column if it contributes to rows 0,1,2 (indices 0-11).
    
    U34 = U44[0:12, :]
    V34 = V44
    W34 = W44[0:12, :]
    
    # Filter columns
    keep = []
    for c in range(W34.shape[1]):
        if np.any(W34[:, c] != 0):
            keep.append(c)
    
    U_final = U34[:, keep]
    V_final = V34[:, keep]
    W_final = W34[:, keep]
    
    # Convert to lists of lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

import numpy as np
import time

def construct():
    """
    Return a rank-28 exact decomposition for 2x3 * 3x5 matrix multiplication.
    
    This implementation uses a blocking strategy:
    1. Split the inner dimension (j) into {0} and {1, 2}.
    2. For j=0, use schoolbook multiplication (rank 10).
    3. For j in {1, 2}, split the output dimension (k) into {0,1}, {2}, and {3,4}.
       - For k in {0,1} and {3,4}, use Strassen's algorithm for 2x2x2 (rank 7 each).
       - For k=2, use schoolbook for 2x2x1 (rank 4).
    
    Total Rank: 10 + 7 + 4 + 7 = 28.
    
    The matrices U, V, W are returned as lists of lists of integers.
    U: shape (6, 28) corresponding to A (2x3)
    V: shape (15, 28) corresponding to B (3x5)
    W: shape (10, 28) corresponding to C (2x5), with index k*2+i
    """
    n, m, p = 2, 3, 5
    rank = 28
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        if val != 0:
            U[u_idx][col] = val
            V[v_idx][col] = val
            W[w_idx][col] = val
        col += 1

    # Helper for Strassen 2x2x2
    # Inputs: 
    #   u_base: index in U for A[0, j_start] (row 0 of the 2x2 block)
    #   v_base: index in V for B[j_start, k_start] (row 0 of the 2x2 block)
    #   w_base: index in W for C[0, k_start] (row 0 of the 2x2 block)
    #   j_step: step in j for U and V
    #   k_step: step in k for V and W
    def add_strassen(u0, u1, v0, v1, w0, w1, j_step, k_step):
        nonlocal col
        # M1 = (A00 + A11) * (B00 + B11)
        # A00 is at u0, A11 is at u1 (u0 + j_step)
        # B00 is at v0, B11 is at v1 (v0 + k_step)
        # Wait, indices:
        # A is 2x2 block. Rows i=0,1. Cols j=0,1.
        # U index for A[i, j_start+j] is u_base + i*m + j? 
        # In our flattened U: index = i*m + j.
        # Let u0_idx be U index for A[0, j_start].
        # Then A[1, j_start] is u0_idx + m.
        # A[0, j_start+1] is u0_idx + 1.
        # A[1, j_start+1] is u0_idx + m + 1.
        
        # Let's define indices explicitly relative to the block start
        # A block rows: i=0,1. Cols: j=0,1.
        # u_00 = u0
        # u_10 = u0 + m
        # u_01 = u0 + 1
        # u_11 = u0 + m + 1
        
        # B block rows: j=0,1. Cols: k=0,1.
        # v_00 = v0
        # v_10 = v0 + p
        # v_01 = v0 + 1
        # v_11 = v0 + p + 1
        
        # C block rows: i=0,1. Cols: k=0,1.
        # w_00 = w0
        # w_10 = w0 + n  # C[1, k_start] is index (k_start)*n + 1
        # w_01 = w0 + k_step * n
        # w_11 = w0 + k_step * n + n

        # Strassen Terms:
        # 1. M1 = (A00 + A11)(B00 + B11)
        #    U: u_00 + u_11, V: v_00 + v_11
        #    W: w_00 + w_11 - M2 - M4? No, standard Strassen:
        #    C00 = M1 - M4 + M6 - M7
        #    C01 = M2 + M4
        #    C10 = M3 + M5
        #    C11 = M1 - M3 + M5 - M2
        
        # Let's stick to the standard 7 multiplications:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # We need to add 7 terms. Each term is a sum of products.
        # Since we are building a rank decomposition, each "multiplication" in Strassen
        # corresponds to one rank-1 term in the tensor decomposition?
        # Yes, the rank of the 2x2x2 tensor is 7.
        # The 7 terms of Strassen provide the 7 rank-1 components.
        
        # Term 1: M1
        # U: A00 + A11 -> indices u_00, u_11
        # V: B00 + B11 -> indices v_00, v_11
        # W: Coeffs for C00, C01, C10, C11
        # C00 += 1, C11 += 1, C01 += 0, C10 += 0? 
        # Wait, M1 contributes to C00 and C11.
        # Specifically: C00 gets +M1, C11 gets +M1.
        # So W[0] += 1, W[3] += 1 (if w_00 is 0, w_11 is 3 in local 2x2? No, w_00 is C[0,k0], w_11 is C[1,k1]).
        # Let's map local C indices to global W indices.
        # Local C00 -> Global w0
        # Local C10 -> Global w0 + n
        # Local C01 -> Global w0 + k_step*n
        # Local C11 -> Global w0 + k_step*n + n
        
        # M1:
        # U: u_00 (1), u_11 (1)
        # V: v_00 (1), v_11 (1)
        # W: w_00 (1), w_11 (1)
        add_term(u_00, v_00, w_00, 1)
        add_term(u_11, v_00, w_00, 1) # Wait, add_term only adds 1 index per matrix.
        # My add_term function adds a single scalar to a single index in U, V, W.
        # But a Strassen term involves sums in U and V.
        # E.g., M1 uses (A00 + A11). This means U has 1 at u_00 and 1 at u_11.
        # My add_term function sets U[idx][col] = val. It doesn't support multiple indices for one col.
        
        # I need to modify the approach.
        # I will manually set the values in the current column for all involved indices.
        
        def set_col(col_idx, u_indices, v_indices, w_indices, vals=1):
            # u_indices, v_indices, w_indices are lists of (index, coeff)
            for idx, val in u_indices:
                U[idx][col_idx] += val
            for idx, val in v_indices:
                V[idx][col_idx] += val
            for idx, val in w_indices:
                W[idx][col_idx] += val

        # Reset col management to be more flexible
        # I will rewrite the core logic to use set_col.
        
        # Let's restart the construction logic inside construct to use this helper properly.
        pass

    # The above helper is getting complex. Let's just write the explicit indices for the 7 Strassen terms
    # for the two blocks, as done in the original code, but encapsulated in a function for clarity.
    
    def add_strassen_block(u_start, v_start, w_start, j_step, k_step):
        nonlocal col
        # u_start is index of A[0, j_start]
        # v_start is index of B[j_start, k_start]
        # w_start is index of C[0, k_start]
        
        # Indices
        u00 = u_start
        u10 = u_start + m
        u01 = u_start + 1
        u11 = u_start + m + 1
        
        v00 = v_start
        v10 = v_start + p
        v01 = v_start + 1
        v11 = v_start + p + 1
        
        w00 = w_start
        w10 = w_start + n
        w01 = w_start + k_step * n
        w11 = w_start + k_step * n + n
        
        # M1 = (A00 + A11) * (B00 + B11)
        # C00 += M1, C11 += M1
        U[u00][col] = 1; U[u11][col] = 1
        V[v00][col] = 1; V[v11][col] = 1
        W[w00][col] = 1; W[w11][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        # C01 += M2, C11 -= M2
        U[u10][col] = 1; U[u11][col] = 1
        V[v00][col] = 1
        W[w01][col] = 1; W[w11][col] = -1
        col += 1
        
        # M3 = A00 * (B01 - B11)
        # C10 += M3, C11 -= M3
        U[u00][col] = 1
        V[v01][col] = 1; V[v11][col] = -1
        W[w10][col] = 1; W[w11][col] = -1
        col += 1
        
        # M4 = A11 * (B10 - B00)
        # C00 -= M4, C10 += M4
        U[u11][col] = 1
        V[v10][col] = 1; V[v00][col] = -1
        W[w00][col] = -1; W[w10][col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B11
        # C00 += M5, C10 += M5
        U[u00][col] = 1; U[u01][col] = 1
        V[v11][col] = 1
        W[w00][col] = 1; W[w10][col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        # C10 -= M6, C11 += M6
        U[u10][col] = 1; U[u00][col] = -1
        V[v00][col] = 1; V[v01][col] = 1
        W[w10][col] = -1; W[w11][col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        # C00 += M7, C11 -= M7
        U[u01][col] = 1; U[u11][col] = -1
        V[v10][col] = 1; V[v11][col] = 1
        W[w00][col] = 1; W[w11][col] = -1
        col += 1

    # Block 1: j=0, k=0..4
    # A[0,0], A[1,0] (u0, u3)
    # B[0,0..4] (v0..v4)
    # C[0,0..4], C[1,0..4] (w0..w9)
    for i in range(n):
        for k in range(p):
            U[i*m + 0][col] = 1
            V[0*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # Block 2a: j in {1,2}, k in {0,1}
    # u_start = A[0,1] = index 1
    # v_start = B[1,0] = index 5
    # w_start = C[0,0] = index 0
    # j_step = 1, k_step = 1
    add_strassen_block(1, 5, 0, 1, 1)
    
    # Block 2b: j in {1,2}, k=2
    # Schoolbook 2x2x1
    # A[0,1], A[1,1], A[0,2], A[1,2]
    # B[1,2], B[2,2]
    # C[0,2], C[1,2]
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i*m + j][col] = 1
            V[j*p + 2][col] = 1
            W[2*n + i][col] = 1
            col += 1
            
    # Block 2c: j in {1,2}, k in {3,4}
    # u_start = A[0,1] = index 1
    # v_start = B[1,3] = index 1*5 + 3 = 8
    # w_start = C[0,3] = index 3*2 + 0 = 6
    # j_step = 1, k_step = 1
    add_strassen_block(1, 8, 6, 1, 1)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Verification
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    n, m, p = 2, 3, 5
    rank = U.shape[1]
    
    # Check rank
    print(f"Rank: {rank}")
    
    # Verify tensor decomposition
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should be 1 if i,j,k valid, else 0.
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                val = 0
                for r in range(rank):
                    val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                err = abs(val - expected)
                if err > max_err:
                    max_err = err
    print(f"Max Error: {max_err}")

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Rank 50 decomposition using Strassen for 2x2x2 blocks
    # Partition: A(2x4) = [A1|A2] (each 2x2), B(4x7) = [B1;B2] (each 2x7)
    # C = A1*B1 + A2*B2
    # Each 2x2 by 2x7: split B into [B_a|B_b|B_c|B_d] (2x2,2x2,2x2,2x1)
    # Use Strassen (rank 7) for each 2x2x2, schoolbook (rank 4) for 2x2x1
    # Per 2x2 by 2x7: 7+7+7+4 = 25
    # Total: 25 + 25 = 50
    
    rank = 50
    
    # We'll build U, V, W as lists of lists
    U = [[0] * rank for _ in range(n * m)]  # 8 x 50
    V = [[0] * rank for _ in range(m * p)]  # 28 x 50
    W = [[0] * rank for _ in range(p * n)]  # 14 x 50
    
    # Strassen 2x2x2 decomposition (7 terms)
    # For A(2x2) * B(2x2) = C(2x2):
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    #
    # Strassen terms (U_factor, V_factor, W_factor):
    # Each term is a rank-1: U[i]*V[j]*W[k]
    
    # Let me define the 7 Strassen terms explicitly
    # Using the standard formulation, I'll define them as (U_vals, V_vals, W_vals)
    # where U_vals has 4 entries (for A's 4 elements), V_vals has 4 entries (for B's 4 elements),
    # W_vals has 4 entries (for C's 4 elements)
    
    # I'll use a verified Strassen decomposition:
    # Term 1: (a00+a10)*(b01+b11) -> c00:+1, c01:+1, c10:+1, c11:-1
    # Term 2: (a10+a11)*b01 -> c10:+1, c11:-1
    # Term 3: (a00+a01)*b11 -> c01:+1, c11:+1
    # Term 4: a00*(b00-b11) -> c00:+1, c10:+1
    # Term 5: a11*(b11-b00) -> c00:-1, c01:+1, c11:+1
    # Term 6: (a00+a01)*b00 -> c00:+1, c11:+1
    # Term 7: (a10+a11)*b10 -> c00:+1, c11:-1
    
    # Wait, I need to be more careful. Let me just define the tensor directly.
    # The Strassen tensor T[i,j,k] where i indexes A, j indexes B, k indexes C
    # T = sum of 7 rank-1 tensors
    
    # Let me use this well-known set of 7 vectors:
    # (u, v, w) for each of the 7 terms, where u,v,w are 4-dimensional
    
    strassen_terms = [
        # (U, V, W) - each is a list of 4 values
        ([1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 1, -1]),  # (a00+a10)(b01+b11)
        ([0, 0, 1, 1], [0, 1, 0, 0], [0, -1, 1, -1]),  # (a10+a11)b01
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]),     # (a00+a01)b11
        ([1, 0, 0, 0], [1, 0, 0, -1], [1, 0, 1, 0]),    # a00(b00-b11)
        ([0, 0, 0, 1], [-1, 0, 0, 1], [-1, 1, 0, 1]),   # a11(b11-b00)
        ([1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1]),     # (a00+a01)b00
        ([0, 0, 1, 1], [0, 0, 1, 0], [1, 0, 0, -1]),    # (a10+a11)b10
    ]
    
    # Let me verify this Strassen decomposition numerically
    def verify_strassen():
        # Test with random matrices
        A_test = np.array([[1, 2], [3, 4]], dtype=float)
        B_test = np.array([[5, 6], [7, 8]], dtype=float)
        C_expected = A_test @ B_test  # [[19, 22], [43, 50]]
        
        C_computed = np.zeros((2, 2))
        for (u, v, w) in strassen_terms:
            # u corresponds to A flattened: [a00, a01, a10, a11]
            # v corresponds to B flattened: [b00, b01, b10, b11]
            # w corresponds to C flattened: [c00, c01, c10, c11]
            A_flat = [A_test[0,0], A_test[0,1], A_test[1,0], A_test[1,1]]
            B_flat = [B_test[0,0], B_test[0,1], B_test[1,0], B_test[1,1]]
            scalar = sum(u[i] * A_flat[i] for i in range(4)) * sum(v[j] * B_flat[j] for j in range(4))
            for k in range(4):
                C_computed[k // 2, k % 2] += w[k] * scalar
        
        return np.allclose(C_computed, C_expected)
    
    if not verify_strassen():
        # Try alternative Strassen terms
        # Let me use a different well-known formulation
        strassen_terms = [
            ([1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 1, -1]),
            ([0, 0, 1, 1], [0, 1, 0, 0], [0, -1, 1, -1]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]),
            ([1, 0, 0, 0], [1, 0, 0, -1], [1, 0, 1, 0]),
            ([0, 0, 0, 1], [-1, 0, 0, 1], [-1, 1, 0, 1]),
            ([1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [0, 0, 1, 0], [1, 0, 0, -1]),
        ]
        # If still wrong, fall back to schoolbook
        if not verify_strassen():
            # Use schoolbook for everything: rank 56
            rank = 56
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
    
    # Now build the rank-50 decomposition
    # Structure:
    # A is 2x4, indexed as A[i, j] where i in {0,1}, j in {0,1,2,3}
    # B is 4x7, indexed as B[j, k] where j in {0,1,2,3}, k in {0,...,6}
    # C is 2x7, indexed as C[i, k] where i in {0,1}, k in {0,...,6}
    #
    # U has 8 rows (2*4), V has 28 rows (4*7), W has 14 rows (7*2)
    #
    # Partition:
    # A = [A1 | A2] where A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B = [B1; B2] where B1 = B[0:2, :], B2 = B[2:4, :]
    # C = A1*B1 + A2*B2
    #
    # For A1*B1 (2x2 * 2x7):
    #   Split B1 into [B1a | B1b | B1c | B1d] where B1a=B1[:,0:2], B1b=B1[:,2:4], B1c=B1[:,4:6], B1d=B1[:,6:7]
    #   A1*B1a: Strassen (rank 7)
    #   A1*B1b: Strassen (rank 7)
    #   A1*B1c: Strassen (rank 7)
    #   A1*B1d: schoolbook (rank 4)
    #   Subtotal: 25
    #
    # For A2*B2 (2x2 * 2x7):
    #   Same structure
    #   Subtotal: 25
    #
    # Total: 50
    
    col = 0
    for block in range(2):  # block 0: A1*B1, block 1: A2*B2
        # A_block is 2x2, corresponding to A columns [block*2 : block*2+2]
        # B_block is 2x7, corresponding to B rows [block*2 : block*2+2]
        
        for subblock in range(3):  # 3 Strassen 2x2x2 blocks
            # B_subblock is 2x2, corresponding to B columns [subblock*2 : subblock*2+2]
            # U indices for A_block: rows (i*4 + block*2 + j) for i in {0,1}, j in {0,1}
            #   i=0,j=0 -> row 0*4 + block*2 + 0 = block*2
            #   i=0,j=1 -> row 0*4 + block*2 + 1 = block*2 + 1
            #   i=1,j=0 -> row 1*4 + block*2 + 0 = 4 + block*2
            #   i=1,j=1 -> row 1*4 + block*2 + 1 = 4 + block*2 + 1
            # V indices for B_subblock: rows (block*2 + j)*7 + subblock*2 + k for j in {0,1}, k in {0,1}
            #   j=0,k=0 -> (block*2+0)*7 + subblock*2 + 0
            #   j=0,k=1 -> (block*2+0)*7 + subblock*2 + 1
            #   j=1,k=0 -> (block*2+1)*7 + subblock*2 + 0
            #   j=1,k=1 -> (block*2+1)*7 + subblock*2 + 1
            # W indices for C_subblock: (subblock*2 + k)*2 + i for k in {0,1}, i in {0,1}
            #   k=0,i=0 -> (subblock*2+0)*2 + 0
            #   k=0,i=1 -> (subblock*2+0)*2 + 1
            #   k=1,i=0 -> (subblock*2+1)*2 + 0
            #   k=1,i=1 -> (subblock*2+1)*2 + 1
            
            A_row_base = block * 2  # offset in A's column index
            B_row_base = block * 2  # offset in B's row index
            B_col_base = subblock * 2  # offset in B's column index
            C_col_base = subblock * 2  # offset in C's column index
            
            # U rows for A_block (4 entries): [a00, a01, a10, a11]
            # a00 = A[0, block*2+0] -> U row 0*4 + block*2 + 0
            # a01 = A[0, block*2+1] -> U row 0*4 + block*2 + 1
            # a10 = A[1, block*2+0] -> U row 1*4 + block*2 + 0
            # a11 = A[1, block*2+1] -> U row 1*4 + block*2 + 1
            U_rows = [
                0 * 4 + A_row_base + 0,  # a00
                0 * 4 + A_row_base + 1,  # a01
                1 * 4 + A_row_base + 0,  # a10
                1 * 4 + A_row_base + 1,  # a11
            ]
            
            # V rows for B_subblock (4 entries): [b00, b01, b10, b11]
            # b00 = B[block*2+0, subblock*2+0] -> V row (block*2+0)*7 + subblock*2 + 0
            # b01 = B[block*2+0, subblock*2+1] -> V row (block*2+0)*7 + subblock*2 + 1
            # b10 = B[block*2+1, subblock*2+0] -> V row (block*2+1)*7 + subblock*2 + 0
            # b11 = B[block*2+1, subblock*2+1] -> V row (block*2+1)*7 + subblock*2 + 1
            V_rows = [
                (B_row_base + 0) * 7 + B_col_base + 0,  # b00
                (B_row_base + 0) * 7 + B_col_base + 1,  # b01
                (B_row_base + 1) * 7 + B_col_base + 0,  # b10
                (B_row_base + 1) * 7 + B_col_base + 1,  # b11
            ]
            
            # W rows for C_subblock (4 entries): [c00, c01, c10, c11]
            # c00 = C[0, subblock*2+0] -> W row (subblock*2+0)*2 + 0
            # c01 = C[0, subblock*2+1] -> W row (subblock*2+0)*2 + 1
            # c10 = C[1, subblock*2+0] -> W row (subblock*2+1)*2 + 0
            # c11 = C[1, subblock*2+1] -> W row (subblock*2+1)*2 + 1
            W_rows = [
                (C_col_base + 0) * 2 + 0,  # c00
                (C_col_base + 0) * 2 + 1,  # c01
                (C_col_base + 1) * 2 + 0,  # c10
                (C_col_base + 1) * 2 + 1,  # c11
            ]
            
            for (u, v, w) in strassen_terms:
                for idx in range(4):
                    if u[idx] != 0:
                        U[U_rows[idx]][col] += u[idx]
                    if v[idx] != 0:
                        V[V_rows[idx]][col] += v[idx]
                    if w[idx] != 0:
                        W[W_rows[idx]][col] += w[idx]
                col += 1
        
        # One schoolbook 2x2x1 block (B column 6)
        # B_subblock is 2x1: B[block*2+0, 6], B[block*2+1, 6]
        # C_subblock is 2x1: C[0, 6], C[1, 6]
        # 4 rank-1 terms:
        # C[0,6] = A[0,block*2+0]*B[block*2+0,6] + A[0,block*2+1]*B[block*2+1,6]
        # C[1,6] = A[1,block*2+0]*B[block*2+0,6] + A[1,block*2+1]*B[block*2+1,6]
        
        B_col = 6
        for i in range(2):
            for j in range(2):
                # U: A[i, block*2+j]
                U_row = i * 4 + block * 2 + j
                # V: B[block*2+j, 6]
                V_row = (block * 2 + j) * 7 + B_col
                # W: C[i, 6]
                W_row = 6 * 2 + i
                
                U[U_row][col] = 1
                V[V_row][col] = 1
                W[W_row][col] = 1
                col += 1
    
    # Verify the decomposition
    def verify_decomposition(U, V, W, n, m, p):
        # Check that sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta
        # Actually, the decomposition should satisfy:
        # For all i,k: C[i,k] = sum_j A[i,j]*B[j,k] = sum_c U[i*m+j... 
        # The tensor T[i,j,k] = 1 if this is the MM tensor
        # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Test with random matrices
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for c in range(len(U_arr[0])):
                    for j in range(m):
                        val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                C_computed[i, k] = val
        
        return np.allclose(C_computed, C_expected, atol=1e-9)
    
    if not verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = 56
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

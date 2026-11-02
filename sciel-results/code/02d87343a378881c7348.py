"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit
    
    n, m, p = 6, 6, 6
    
    # Schoolbook baseline: rank 216
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    # Try Strassen-based decomposition: 2x2 block with 3x3 schoolbook sub-blocks
    # This gives rank = 7 * (3*3*3) = 7 * 27 = 189
    # Actually, for 6x6x6 using 2x2 blocks of 3x3 matrices:
    # Strassen: 7 multiplications of 3x3 matrices
    # Each 3x3 multiplication via schoolbook: 27 terms
    # Total rank: 7 * 27 = 189
    
    # Build the Strassen decomposition
    # A = [[A11, A12], [A21, A22]] where each Ai is 3x3
    # B = [[B11, B12], [B21, B22]] where each Bj is 3x3
    # C = [[C11, C12], [C21, C22]]
    
    # Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen multiplication Mk, we decompose it into schoolbook 3x3 multiplications
    # A 3x3 matrix has 9 entries, indexed by (i,j) with i,j in {0,1,2}
    # The full 6x6 matrix A has entries A[i,j] for i,j in {0,...,5}
    # Block mapping: A11[i,j] = A[2i, 2j], A12[i,j] = A[2i, 2j+1], etc.
    
    # Let's build the full decomposition
    # Each term in the schoolbook 3x3 multiplication of matrices X and Y:
    # X[i,j] * Y[j,k] contributes to Z[i,k]
    
    # For the full 6x6 matrices, we need to map block indices to full indices
    
    def block_to_full(block_idx, i, j):
        """Map block index and within-block indices to full matrix indices."""
        # block_idx: 0=A11, 1=A12, 2=A21, 3=A22 (row block, col block)
        row_block, col_block = divmod(block_idx, 2)
        return 3 * row_block + i, 3 * col_block + j
    
    # Strassen terms: each is a product of (sum/diff of A blocks) * (sum/diff of B blocks)
    # We need to expand each into individual schoolbook terms
    
    # Let's define the Strassen decomposition more carefully
    # M1 = (A11 + A22)(B11 + B22)
    # This expands to: A11*B11 + A11*B22 + A22*B11 + A22*B22
    # Each of these is a 3x3 matrix multiplication
    
    # For efficiency, let's directly construct the rank-189 decomposition
    
    rank_strassen = 7 * 27  # 189
    U_strassen = np.zeros((n * m, rank_strassen), dtype=float)
    V_strassen = np.zeros((m * p, rank_strassen), dtype=float)
    W_strassen = np.zeros((p * n, rank_strassen), dtype=float)
    
    # Define the 7 Strassen products and their contributions to C blocks
    # Each Strassen product Mk is a 3x3 matrix = sum of schoolbook terms
    # For each schoolbook term in Mk: X[i,j]*Y[j,k] -> Z[i,k]
    
    # Strassen product definitions: (A_part, B_part, sign, C_block_contribution)
    # A_part: list of (block_idx, sign) for A
    # B_part: list of (block_idx, sign) for B
    
    strassen_products = [
        # M1 = (A11 + A22)(B11 + B22)
        {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1), (2, -1), (1, 1)]},
        # M2 = (A21 + A22)B11
        {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)]},
        # M3 = A11(B12 - B22)
        {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},
        # M4 = A22(B21 - B11)
        {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
        # M5 = (A11 + A12)B22
        {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},
        # M6 = (A21 - A11)(B11 + B12)
        {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},
        # M7 = (A12 - A22)(B21 + B22)
        {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)]},
    ]
    
    col = 0
    for prod_idx, prod in enumerate(strassen_products):
        # Expand this Strassen product into schoolbook terms
        for a_block, a_sign in prod['A']:
            for b_block, b_sign in prod['B']:
                # This is a 3x3 * 3x3 multiplication
                # For each schoolbook term: i,j,k in {0,1,2}
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            # A entry: block a_block, position (i,j)
                            # B entry: block b_block, position (j,k)
                            # C contribution: for each (c_block, c_sign) in prod['C'], position (i,k)
                            
                            a_row, a_col = block_to_full(a_block, i, j)
                            b_row, b_col = block_to_full(b_block, j, k)
                            
                            U_strassen[a_row * m + a_col, col] = a_sign
                            V_strassen[b_row * p + b_col, col] = b_sign
                            
                            for c_block, c_sign in prod['C']:
                                c_row, c_col = block_to_full(c_block, i, k)
                                W_strassen[c_row * n + c_col, col] = c_sign
                            
                            col += 1
    
    # Verify the decomposition
    # Check that sum_c U[:,c] * V[:,c] * W[:,c] equals the matrix multiplication tensor
    # For 6x6x6, this is a 216x216x216 tensor (too large to verify directly)
    # Instead, verify on random inputs
    
    def verify_decomposition(U, V, W, n, m, p, num_trials=10):
        np.random.seed(42)
        for _ in range(num_trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Compute via decomposition
            C_computed = np.zeros((n, p))
            for c in range(U.shape[1]):
                u_vec = U[:, c].reshape(n, m)
                v_vec = V[:, c].reshape(m, p)
                w_vec = W[:, c].reshape(p, n)
                # The decomposition should give: C = sum_c (u_vec * v_vec) * w_vec (element-wise in some sense)
                # Actually, for tensor decomposition: C[i,k] = sum_j sum_c A[i,j]*B[j,k]*W[k*n+i,c]*U[i*m+j,c]*V[j*p+k,c]
                # Let's compute it properly
                pass
            
            # Proper verification:
            # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all j
            # Actually: C[i,k] = sum_j sum_c U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k] * W[k*n+i, c]
            # Wait, that's not right either. Let me think...
            
            # The tensor decomposition means:
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # where T[i,j,k] = 1 if this is a valid (i,j,k) triple for matrix mult, i.e., always 1 for 6x6x6
            # Actually no, T[i,j,k] represents the coefficient in C = A*B:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # So T[i,j,k] = 1 for all i,j,k (since all products contribute)
            
            # Verification: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1 for all i,j,k
            # And the computed C[i,k] = sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] * A[i,j] * B[j,k]
            
            C_computed = np.zeros((n, p))
            for c in range(U.shape[1]):
                Uc = U[:, c]  # shape (n*m,)
                Vc = V[:, c]  # shape (m*p,)
                Wc = W[:, c]  # shape (p*n,)
                
                # Reshape
                Uc_mat = Uc.reshape(n, m)
                Vc_mat = Vc.reshape(m, p)
                Wc_mat = Wc.reshape(p, n)
                
                # Contribution to C: sum_j A[i,j]*B[j,k]*... 
                # C[i,k] += sum_j Uc_mat[i,j] * A[i,j] * Vc_mat[j,k] * B[j,k] * Wc_mat[k,i]
                # This doesn't look right. Let me reconsider.
                
                # Actually, the standard tensor decomposition for matrix multiplication:
                # C = sum_c (A * Uc^T) * (Vc * B^T) * Wc^T ... no
                # 
                # The correct interpretation:
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # = sum_j sum_c delta(i,j,k,c) * A[i,j] * B[j,k]
                # where delta is the tensor T[i,j,k] decomposed as sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                # 
                # So: C[i,k] = sum_c sum_j U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] * A[i,j] * B[j,k]
                # = sum_c W[k*n+i,c] * (sum_j A[i,j]*U[i*m+j,c]) * (sum_j V[j*p+k,c]*B[j,k])
                # Hmm, that's not quite separable in the simple way.
                
                # Let me just verify the tensor identity directly:
                pass
            
            # Direct tensor verification
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] should equal 1
                        tensor_val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        max_err = max(max_err, abs(tensor_val - 1.0))
            
            if max_err > 1e-10:
                return False
        return True
    
    # Verify Strassen decomposition
    if verify_decomposition(U_strassen, V_strassen, W_strassen, n, m, p, num_trials=1):
        # Convert to lists for the interface
        U_list = U_strassen.tolist()
        V_list = V_strassen.tolist()
        W_list = W_strassen.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    U_list = U_school.tolist()
    V_list = V_school.tolist()
    W_list = W_school.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

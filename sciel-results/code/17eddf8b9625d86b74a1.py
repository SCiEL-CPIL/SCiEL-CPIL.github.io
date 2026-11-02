import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a bilinear decomposition for 4x5 * 5x5 matrix multiplication.
    Uses Strassen's algorithm on the 4x4 x 4x4 core and schoolbook for the rest.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vals, v_vals, w_vals):
        """Add a single rank-1 term to the decomposition."""
        U_list.append(u_vals)
        V_list.append(v_vals)
        W_list.append(w_vals)
    
    # Strategy: Partition A (4x5) and B (5x5) to use Strassen on a 4x4 x 4x4 subproblem.
    # Split the inner dimension (size 5) into 4 + 1.
    # A = [A1 | A2] where A1 is 4x4, A2 is 4x1
    # B = [B1; B2] where B1 is 4x5, B2 is 1x5
    # C = A1 @ B1 + A2 @ B2
    #
    # A1 @ B1 is (4x4) x (4x5) = 4x5.
    # Split B1 into B11 (4x4) and B12 (4x1).
    # C1 = A1 @ B11 (4x4) + A1 @ B12 (4x1)
    #
    # A2 @ B2 is (4x1) x (1x5) = 4x5.
    #
    # So:
    # C = A1 @ B11 + A1 @ B12 + A2 @ B2
    #
    # A1 @ B11: 4x4 x 4x4. Use Strassen (rank 49).
    # A1 @ B12: 4x4 x 1. Schoolbook (rank 16).
    # A2 @ B2: 4x1 x 5. Schoolbook (rank 20).
    #
    # Total rank: 49 + 16 + 20 = 85.
    
    # Strassen for 4x4 x 4x4:
    # Split A1 into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # Split B11 into 2x2 blocks: B111, B112, B211, B212 (each 2x2)
    #
    # M1 = (A11+A22)(B111+B212)
    # M2 = (A21+A22)B111
    # M3 = A11(B112-B212)
    # M4 = A22(B211-B111)
    # M5 = (A11+A12)B212
    # M6 = (A21-A11)(B111+B112)
    # M7 = (A12-A22)(B211+B212)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2 x 2x2 product. Use Strassen for 2x2 x 2x2 (rank 7).
    # So 7 * 7 = 49 mults for the 4x4 x 4x4 core.
    
    # Strassen for 2x2 x 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Strassen 2x2 decomposition as (u, v, w) tuples where u, v, w are 4-element vectors
    strassen_2x2 = [
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]),  # m1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # m2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # m3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # m4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # m5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),  # m6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # m7
    ]
    
    # 4x4 Strassen: For each Mi, define which 2x2 blocks of A1 and B11 are used
    # with what coefficients.
    # A1 blocks: A11 (top-left), A12 (top-right), A21 (bottom-left), A22 (bottom-right)
    # B11 blocks: B111 (top-left), B112 (top-right), B211 (bottom-left), B212 (bottom-right)
    
    # For each Mi, specify:
    # - Which A blocks are summed (with coefficients) to form A_Mi
    # - Which B blocks are summed (with coefficients) to form B_Mi
    # - How the 2x2 output C_Mi maps to the 4x4 output C11
    
    # A11 is at positions (0,0), (0,1), (1,0), (1,1) in 4x4
    # A12 is at positions (0,2), (0,3), (1,2), (1,3)
    # A21 is at positions (2,0), (2,1), (3,0), (3,1)
    # A22 is at positions (2,2), (2,3), (3,2), (3,3)
    
    # B111 is at positions (0,0), (0,1), (1,0), (1,1) in B11 (4x4)
    # B112 is at positions (0,2), (0,3), (1,2), (1,3)
    # B211 is at positions (2,0), (2,1), (3,0), (3,1)
    # B212 is at positions (2,2), (2,3), (3,2), (3,3)
    
    # Define block offsets for A1 (4x4) and B11 (4x4)
    # Each 2x2 block has 4 elements in row-major order
    # A11: rows 0-1, cols 0-1 -> indices 0-3
    # A12: rows 0-1, cols 2-3 -> indices 4-7
    # A21: rows 2-3, cols 0-1 -> indices 8-11
    # A22: rows 2-3, cols 2-3 -> indices 12-15
    
    # B111: rows 0-1, cols 0-1 -> indices 0-3
    # B112: rows 0-1, cols 2-3 -> indices 4-7
    # B211: rows 2-3, cols 0-1 -> indices 8-11
    # B212: rows 2-3, cols 2-3 -> indices 12-15
    
    # For each Strassen product Mi, define the linear combinations:
    # A_Mi = sum of alpha_i * A_block_i
    # B_Mi = sum of beta_i * B_block_i
    
    # M1 = (A11+A22)(B111+B212)
    # A_M1 = A11 + A22
    # B_M1 = B111 + B212
    # C_M1 contributes to C11 (top-left 2x2 of C)
    
    # M2 = (A21+A22)B111
    # A_M2 = A21 + A22
    # B_M2 = B111
    # C_M2 contributes to C21 (bottom-left 2x2 of C)
    
    # M3 = A11(B112-B212)
    # A_M3 = A11
    # B_M3 = B112 - B212
    # C_M3 contributes to C12 (top-right 2x2 of C)
    
    # M4 = A22(B211-B111)
    # A_M4 = A22
    # B_M4 = B211 - B111
    # C_M4 contributes to C11 and C21
    
    # M5 = (A11+A12)B212
    # A_M5 = A11 + A12
    # B_M5 = B212
    # C_M5 contributes to C11 and C12
    
    # M6 = (A21-A11)(B111+B112)
    # A_M6 = A21 - A11
    # B_M6 = B111 + B112
    # C_M6 contributes to C22 (bottom-right 2x2 of C)
    
    # M7 = (A12-A22)(B211+B212)
    # A_M7 = A12 - A22
    # B_M7 = B211 + B212
    # C_M7 contributes to C11
    
    # For each Mi, we need to:
    # 1. Form A_Mi and B_Mi as linear combinations of the 4 blocks
    # 2. Apply the 2x2 Strassen decomposition (7 rank-1 terms)
    # 3. Map the 2x2 output C_Mi to the appropriate positions in the 4x4 output
    
    # Let's build the 49 rank-1 terms for the 4x4 x 4x4 part.
    
    # For each of the 7 Strassen products, we'll generate 7 rank-1 terms.
    # Total: 7 * 7 = 49 terms.
    
    # Block indices in the flat 16-element vector for A1 and B11:
    # A11: 0-3, A12: 4-7, A21: 8-11, A22: 12-15
    # B111: 0-3, B112: 4-7, B211: 8-11, B212: 12-15
    
    # For each Mi, define:
    # - A_comb: dict from block index to coefficient
    # - B_comb: dict from block index to coefficient
    # - C_target: which 2x2 block of the 4x4 output this contributes to
    
    strassen_4x4_specs = [
        # M1: (A11+A22)(B111+B212) -> C11
        ({0: 1, 12: 1}, {0: 1, 12: 1}, 'C11'),
        # M2: (A21+A22)B111 -> C21
        ({8: 1, 12: 1}, {0: 1}, 'C21'),
        # M3: A11(B112-B212) -> C12
        ({0: 1}, {4: 1, 12: -1}, 'C12'),
        # M4: A22(B211-B111) -> C11 and C21
        ({12: 1}, {8: 1, 0: -1}, 'C11_C21'),
        # M5: (A11+A12)B212 -> C11 and C12
        ({0: 1, 4: 1}, {12: 1}, 'C11_C12'),
        # M6: (A21-A11)(B111+B112) -> C22
        ({8: 1, 0: -1}, {0: 1, 4: 1}, 'C22'),
        # M7: (A12-A22)(B211+B212) -> C11
        ({4: 1, 12: -1}, {8: 1, 12: 1}, 'C11'),
    ]
    
    # For each Mi, apply the 2x2 Strassen and map to 4x4 output.
    
    for mi_idx, (A_comb, B_comb, c_target) in enumerate(strassen_4x4_specs):
        for strassen_term in strassen_2x2:
            u_2x2, v_2x2, w_2x2 = strassen_term
            
            # u_2x2 is a 4-element vector for the 2x2 input A_Mi
            # v_2x2 is a 4-element vector for the 2x2 input B_Mi
            # w_2x2 is a 4-element vector for the 2x2 output C_Mi
            
            # We need to form the full 16-element U and V vectors for the 4x4 x 4x4 part.
            
            # U vector: 16 elements for A1
            U_vec = [0] * 16
            for block_idx, coeff in A_comb.items():
                for k in range(4):
                    U_vec[block_idx + k] += coeff * u_2x2[k]
            
            # V vector: 16 elements for B11
            V_vec = [0] * 16
            for block_idx, coeff in B_comb.items():
                for k in range(4):
                    V_vec[block_idx + k] += coeff * v_2x2[k]
            
            # W vector: 16 elements for the 4x4 output C
            W_vec = [0] * 16
            # Map the 2x2 output C_Mi to the appropriate positions in the 4x4 output
            if c_target == 'C11':
                # C11 is top-left 2x2: positions 0-3
                for k in range(4):
                    W_vec[k] = w_2x2[k]
            elif c_target == 'C12':
                # C12 is top-right 2x2: positions 4-7
                for k in range(4):
                    W_vec[4 + k] = w_2x2[k]
            elif c_target == 'C21':
                # C21 is bottom-left 2x2: positions 8-11
                for k in range(4):
                    W_vec[8 + k] = w_2x2[k]
            elif c_target == 'C22':
                # C22 is bottom-right 2x2: positions 12-15
                for k in range(4):
                    W_vec[12 + k] = w_2x2[k]
            elif c_target == 'C11_C21':
                # Contributes to both C11 and C21
                for k in range(4):
                    W_vec[k] = w_2x2[k]  # C11
                    W_vec[8 + k] = w_2x2[k]  # C21
            elif c_target == 'C11_C12':
                # Contributes to both C11 and C12
                for k in range(4):
                    W_vec[k] = w_2x2[k]  # C11
                    W_vec[4 + k] = w_2x2[k]  # C12
            
            # Now, we need to extend U, V, W to the full 4x5, 5x5, 5x4 dimensions.
            # U is 4x5 = 20 elements, V is 5x5 = 25 elements, W is 5x4 = 20 elements.
            
            # A1 is the first 4 columns of A (4x5), so U has 20 elements.
            # A1 occupies columns 0-3, so U_vec (16 elements) maps to U[0:16]
            # A2 is the 5th column, which is not used in this part.
            
            U_full = [0] * 20
            for k in range(16):
                U_full[k] = U_vec[k]
            
            # B11 is the top-left 4x4 of B (5x5), so V has 25 elements.
            # B11 occupies rows 0-3, cols 0-3, so V_vec (16 elements) maps to V[0:16]
            # B12 is the 5th column of B1, B2 is the 5th row.
            
            V_full = [0] * 25
            for k in range(16):
                V_full[k] = V_vec[k]
            
            # C is 5x4 = 20 elements. W_vec (16 elements) maps to W[0:16]
            # The bottom row and rightmost column of C are not affected by A1@B11.
            
            W_full = [0] * 20
            for k in range(16):
                W_full[k] = W_vec[k]
            
            U_list.append(U_full)
            V_list.append(V_full)
            W_list.append(W_full)
    
    # Now add the schoolbook terms for A1 @ B12 (4x4 x 1, rank 16)
    # A1 is 4x4, B12 is 4x1, result is 4x1
    # This contributes to columns 0-3 of C (the first 4 columns)
    
    for j in range(4):  # rows of A1
        for k in range(4):  # cols of A1 = rows of B12
            for i in range(4):  # this is actually the output row
                # Wait, A1 @ B12: A1 is 4x4, B12 is 4x1, result is 4x1
                # So the output is 4x1, which contributes to the first 4 rows of C, column 4 (the 5th column)
                
                # Actually, let me reconsider the dimensions.
                # A is 4x5, B is 5x5, C is 4x5
                # A = [A1 | A2] where A1 is 4x4, A2 is 4x1
                # B = [B1; B2] where B1 is 4x5, B2 is 1x5
                # C = A1 @ B1 + A2 @ B2
                #
                # B1 is 4x5. Split B1 = [B11 | B12] where B11 is 4x4, B12 is 4x1
                # C = A1 @ B11 + A1 @ B12 + A2 @ B2
                #
                # A1 @ B11: 4x4 x 4x4 = 4x4 (contributes to top-left 4x4 of C)
                # A1 @ B12: 4x4 x 4x1 = 4x1 (contributes to top-right 4x1 of C, i.e., rows 0-3, col 4)
                # A2 @ B2: 4x1 x 1x5 = 4x5 (contributes to bottom row of C)
                
                # So A1 @ B12 contributes to C[0:4, 4]
                
                pass
    
    # Let's redo the schoolbook terms properly.
    
    # A1 @ B12: A1 is 4x4, B12 is 4x1, result is 4x1
    # C[i, 4] += sum_j A1[i, j] * B12[j, 0] for i in 0..3, j in 0..3
    
    for i in range(4):  # output row (0-3)
        for j in range(4):  # A1 column / B12 row
            U_term = [0] * 20
            V_term = [0] * 25
            W_term = [0] * 20
            
            # A1[i, j] is at position i*5 + j in U (since A is 4x5)
            U_term[i * 5 + j] = 1
            
            # B12[j, 0] is at position j*5 + 4 in V (since B is 5x5, B12 is col 4, rows 0-3)
            V_term[j * 5 + 4] = 1
            
            # C[i, 4] is at position 4*4 + i = 16 + i in W (since C is 4x5, row-major)
            W_term[4 * 4 + i] = 1  # Wait, C is 4x5, so C[i, 4] is at position i*5 + 4
            
            W_term[i * 5 + 4] = 1
            
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
    
    # A2 @ B2: A2 is 4x1, B2 is 1x5, result is 4x5
    # C[i, k] += A2[i, 0] * B2[0, k] for i in 0..3, k in 0..4
    
    for i in range(4):  # output row
        for k in range(5):  # output column
            U_term = [0] * 20
            V_term = [0] * 25
            W_term = [0] * 20
            
            # A2[i, 0] is at position i*5 + 4 in U (col 4 of A)
            U_term[i * 5 + 4] = 1
            
            # B2[0, k] is at position 4*5 + k in V (row 4 of B)
            V_term[4 * 5 + k] = 1
            
            # C[i, k] is at position i*5 + k in W
            W_term[i * 5 + k] = 1
            
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
    
    # Convert to numpy arrays
    U = np.array(U_list, dtype=float).T
    V = np.array(V_list, dtype=float).T
    W = np.array(W_list, dtype=float).T
    
    # Verify the decomposition
    # The tensor T_{i,j,k} = delta_{i,j} * delta_{j,k} for matrix multiplication
    # Actually, for A (4x5) @ B (5x5) = C (4x5):
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # So T_{i, j, k} = 1 if i, j, k form a valid product term, else 0
    # More precisely, T is a 4x5x5 tensor where T[i, j, k] = 1 if we're computing C[i, k] = A[i, j] * B[j, k]
    
    # Let's verify by checking that sum_k U[:, k] * V[:, k] * W[:, k] = T
    # where T[i, j, k] is the matrix multiplication tensor
    
    # Actually, the decomposition is:
    # C = sum_k (U_k * A^T) * (V_k * B^T) * W_k^T
    # Or more precisely:
    # C[i, k] = sum_l U[i, l] * V[j, l] * W[k, l]  (this is not quite right)
    
    # The correct form is:
    # T_{i, j, k} = sum_l U_{i, l} * V_{j, l} * W_{k, l}
    # where T is the matrix multiplication tensor.
    
    # For 4x5 * 5x5:
    # U is 20x rank, V is 25x rank, W is 20x rank
    # T is a 4x5x5 tensor, but we're using flat indices:
    # U index: i*5 + j (i in 0..3, j in 0..4)
    # V index: j*5 + k (j in 0..4, k in 0..4)
    # W index: i*5 + k (i in 0..3, k in 0..4)
    
    # Let's verify a few entries.
    
    # T[i*5+j, j*5+k, i*5+k] = 1 for valid i, j, k
    # Actually, the tensor T has dimensions (n*m) x (m*p) x (p*n) = 20 x 25 x 20
    # T[a, b, c] = 1 if a = i*m + j, b = j*p + k, c = k*n + i for some i, j, k
    # Otherwise T[a, b, c] = 0
    
    # Let's construct T and verify.
    
    T = np.zeros((20, 25, 20))
    for i in range(4):
        for j in range(5):
            for k in range(5):
                T[i * 5 + j, j * 5 + k, k * 4 + i] = 1
    
    # Compute U @ T @ V^T @ W or something similar
    # Actually, the decomposition is:
    # T = sum_l U_l * V_l^T * W_l^T
    # Or T[a, b, c] = sum_l U[a, l] * V[b, l] * W[c, l]
    
    T_reconstructed = np.einsum('al,bl,cl->abc', U, V, W)
    
    # Check if they match
    max_diff = np.max(np.abs(T - T_reconstructed))
    
    if max_diff > 1e-9:
        # Decomposition is not exact, fall back to schoolbook
        rank = n * m * p
        U_school = np.zeros((n * m, rank), dtype=int)
        V_school = np.zeros((m * p, rank), dtype=int)
        W_school = np.zeros((p * n, rank), dtype=int)
        
        col = 0
        for k in range(p):
            for j in range(m):
                for i in range(n):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
                    
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
    
    # Convert to integer/half-integer if possible
    U_out = U.astype(int) if np.allclose(U, np.round(U)) else U * 2
    V_out = V.astype(int) if np.allclose(V, np.round(V)) else V * 2
    W_out = W.astype(int) if np.allclose(W, np.round(W)) else W * 2
    
    # Make sure they're actually integers or half-integers
    if not np.allclose(U_out, np.round(U_out)):
        U_out = np.round(U_out).astype(int)
    if not np.allclose(V_out, np.round(V_out)):
        V_out = np.round(V_out).astype(int)
    if not np.allclose(W_out, np.round(W_out)):
        W_out = np.round(W_out).astype(int)
    
    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
# EVOLVE-BLOCK-END

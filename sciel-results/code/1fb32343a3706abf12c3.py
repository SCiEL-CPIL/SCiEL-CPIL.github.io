import numpy as np

# Task dimensions: A (n x m) @ B (m x p) = C (n x p)
# n=4, m=4, p=8
N, M, P = 4, 4, 8

def construct():
    """
    Construct a rank-minimal (or low-rank) exact tensor decomposition for 4x4x8 matrix multiplication.
    Uses recursive Strassen algorithm for the two 4x4x4 blocks.
    Rank = 49 (for 4x4x4) + 49 (for 4x4x4) = 98.
    """
    
    def strassen_4x4_forms():
        """
        Recursively generate linear forms for A, B, C for 4x4x4 matrix multiplication using Strassen.
        Returns three lists of linear forms (represented as dicts mapping (i,j) -> coeff) for A, B, C.
        Actually, for the tensor decomposition, we need to build the U, V, W matrices directly.
        Each 'rank term' corresponds to a scalar multiplication $u_r \cdot v_r \cdot w_r$.
        $u_r$ is a linear form on A, $v_r$ on B, $w_r$ on C.
        
        We will return U, V, W as numpy arrays of shape (N*M, rank), (M*P, rank), (N*P, rank)
        for the 4x4x4 case.
        """
        # Base case: 1x1x1
        # A[0,0] * B[0,0] = C[0,0]
        # U[0,0] = 1, V[0,0] = 1, W[0,0] = 1
        # Indices: A: 0, B: 0, C: 0
        
        def generate(n, m, p):
            """
            Generates U, V, W for n x m x p multiplication using Strassen if possible.
            Assumes n=m=p for simplicity in this specific implementation, 
            but the logic holds for square matrices.
            Returns U (n*m, rank), V (m*p, rank), W (p*n, rank) -- wait, W index convention.
            Task says W[k*n+i, r]. So W is (p*n, rank).
            A index: i*m + j ? No, standard A is (n,m). Row i, Col j. Index i*m+j.
            B index: j*p + k ? No, B is (m,p). Row j, Col k. Index j*p+k.
            C index: k*n + i ? Task says W[k*n+i, r]. This corresponds to C[i, k] where 
            C is (n, p). Wait.
            Usually C[i,k] = sum_j A[i,j] B[j,k].
            Tensor index: (i, j, k).
            U row index: i*m + j.
            V row index: j*p + k.
            W row index: k*n + i.
            Let's verify W index. If i=0, k=0, idx=0. i=1, k=0, idx=1. i=0, k=1, idx=4 (since n=4).
            This is C[i, k] flattened in column-major order of C? No.
            C is n x p.
            k is column index (0..p-1). i is row index (0..n-1).
            Index = k*n + i. This is column-major storage of C.
            Okay, I will stick to this.
            """
            if n == 1 and m == 1 and p == 1:
                U = np.array([[1.0]])
                V = np.array([[1.0]])
                W = np.array([[1.0]])
                return U, V, W
            
            # Split into halves
            n1, n2 = n // 2, n - n // 2
            m1, m2 = m // 2, m - m // 2
            p1, p2 = p // 2, p - p // 2
            
            # Strassen assumes square matrices for the standard 7-product formulation.
            # Here n=m=p=4 -> 2x2 blocks.
            # We split A into A11, A12, A21, A22 (each n1 x m1)
            # B into B11, B12, B21, B22 (each m1 x p1)
            # C into C11, C12, C21, C22 (each n1 x p1)
            
            # Indices in local coordinates (i, j, k) range [0, n1-1], [0, m1-1], [0, p1-1]
            # We need to map global indices to local indices and vice versa.
            
            # The 7 products:
            # P1 = (A11 + A22) * (B11 + B22)
            # P2 = (A21 + A22) * B11
            # P3 = A11 * (B12 - B22)
            # P4 = A22 * (B21 - B11)
            # P5 = (A11 + A12) * B22
            # P6 = (A21 - A11) * (B11 + B12)
            # P7 = (A12 - A22) * (B21 + B22)
            
            # Recursively get decompositions for the sub-products.
            # Each sub-product is a matrix multiplication of size (n1, m1) x (m1, p1).
            # U1, V1, W1 = generate(n1, m1, p1)
            
            U_sub, V_sub, W_sub = generate(n1, m1, p1)
            rank_sub = U_sub.shape[1]
            
            # Total rank is 7 * rank_sub
            total_rank = 7 * rank_sub
            
            U = np.zeros((n * m, total_rank))
            V = np.zeros((m * p, total_rank))
            W = np.zeros((p * n, total_rank))
            
            # Define helper to extract sub-matrix coefficients from U_sub/V_sub/W_sub
            # and place them into the larger U/V/W.
            
            # The linear form for A in P1 is (A11 + A22).
            # This means for each rank term r in P1, the coefficient for A[i,j] is 
            # coeff_A11[i,j] + coeff_A22[i,j] where i,j are local indices.
            # We need to add the rows of U_sub corresponding to A11 indices to the rows
            # of U corresponding to A11 global indices, and similarly for A22.
            
            def add_block(U_target, start_i, start_j, n_block, m_block, U_source, scale=1.0):
                """
                Adds U_source (n_block*m_block x rank) to U_target at offset (start_i, start_j).
                U_source row index corresponds to local (i, j) -> i*m_block + j.
                U_target row index corresponds to global (start_i + i, start_j + j) -> (start_i + i)*m + (start_j + j).
                """
                # Determine the row indices in U_target
                # i ranges 0..n_block-1, j ranges 0..m_block-1
                # target_row = (start_i + i) * m + (start_j + j)
                # source_row = i * m_block + j
                
                # Iterate over source rows and add to target rows
                for i in range(n_block):
                    for j in range(m_block):
                        src_idx = i * m_block + j
                        tgt_idx = (start_i + i) * m + (start_j + j)
                        U_target[tgt_idx, :] += U_source[src_idx, :] * scale
            
            def add_block_V(V_target, start_j, start_k, m_block, p_block, V_source, scale=1.0):
                """
                Adds V_source to V_target at offset (start_j, start_k).
                V_source row index: j*p_block + k (local)
                V_target row index: (start_j + j)*p + (start_k + k) (global)
                Note: m_block is the row dim of B block, p_block is col dim.
                """
                for j in range(m_block):
                    for k in range(p_block):
                        src_idx = j * p_block + k
                        tgt_idx = (start_j + j) * p + (start_k + k)
                        V_target[tgt_idx, :] += V_source[src_idx, :] * scale
                        
            def add_block_W(W_target, start_i, start_k, n_block, p_block, W_source, scale=1.0):
                """
                Adds W_source to W_target at offset (start_i, start_k).
                W_source row index: k*n_block + i (local, col-major)
                W_target row index: (start_k + k)*n + (start_i + i) (global)
                """
                for i in range(n_block):
                    for k in range(p_block):
                        src_idx = k * n_block + i
                        tgt_idx = (start_k + k) * n + (start_i + i)
                        W_target[tgt_idx, :] += W_source[src_idx, :] * scale

            # Now populate U, V, W for the 7 Strassen products
            
            # 1. P1 = (A11 + A22) * (B11 + B22) -> C11, C12, C21, C22 contributions
            # A part: A11 + A22
            add_block(U, 0, 0, n1, m1, U_sub, 1.0)   # A11
            add_block(U, n1, m1, n2, m2, U_sub, 1.0) # A22
            # B part: B11 + B22
            add_block_V(V, 0, 0, m1, p1, V_sub, 1.0) # B11
            add_block_V(V, m1, p1, m2, p2, V_sub, 1.0) # B22
            # C part: C11 += P1, C12 += P1, C21 += P1, C22 -= P1 (Standard Strassen reconstruction: C11 = P1 + P4 - P5 - P7)
            # Wait, let's check the standard Strassen formulas again.
            # C11 = P1 + P4 - P5 - P7
            # C12 = P3 + P7
            # C21 = P2 + P5
            # C22 = P1 - P2 + P3 + P6
            
            # So for P1, it contributes +1 to C11, +1 to C12, +1 to C21, -1 to C22.
            add_block_W(W, 0, 0, n1, p1, W_sub, 1.0)   # C11
            add_block_W(W, 0, p1, n1, p2, W_sub, 1.0)  # C12
            add_block_W(W, n1, 0, n2, p1, W_sub, 1.0)  # C21
            add_block_W(W, n1, p1, n2, p2, W_sub, -1.0) # C22
            
            # Copy U, V, W for P1 to columns 0..rank_sub-1
            # Actually, I've been adding to U, V, W which are zeros. 
            # The `add_block` functions modify U, V, W in place.
            # So for P1, the contributions are in columns 0..rank_sub-1.
            
            # 2. P2 = (A21 + A22) * B11
            # A part: A21 + A22
            # We need to shift the source U_sub to correspond to A21 and A22.
            # But U_sub is the same linear form structure. 
            # The linear form for P2's A-part is (A21 + A22).
            # We add U_sub to A21 position and A22 position.
            # We must use a different set of columns in U, V, W for P2.
            
            # Let's create a helper to "write" a term into a specific slice of columns.
            
            def write_term(U_target, V_target, W_target, col_offset, 
                           A_offsets, B_offsets, C_coeffs, 
                           U_sub, V_sub, W_sub):
                """
                Writes a Strassen term into U, V, W at col_offset.
                A_offsets: list of (i_start, j_start, n_block, m_block, sign)
                B_offsets: list of (j_start, k_start, m_block, p_block, sign)
                C_coeffs: list of (i_start, k_start, n_block, p_block, sign)
                """
                cols = slice(col_offset, col_offset + rank_sub)
                
                # A
                for i_s, j_s, n_b, m_b, sign in A_offsets:
                    for i in range(n_b):
                        for j in range(m_b):
                            src_idx = i * m_b + j
                            tgt_idx = (i_s + i) * m + (j_s + j)
                            U_target[tgt_idx, cols] += U_sub[src_idx, :] * sign
                            
                # B
                for j_s, k_s, m_b, p_b, sign in B_offsets:
                    for j in range(m_b):
                        for k in range(p_b):
                            src_idx = j * p_b + k
                            tgt_idx = (j_s + j) * p + (k_s + k)
                            V_target[tgt_idx, cols] += V_sub[src_idx, :] * sign
                            
                # C
                for i_s, k_s, n_b, p_b, sign in C_coeffs:
                    for i in range(n_b):
                        for k in range(p_b):
                            src_idx = k * n_b + i
                            tgt_idx = (k_s + k) * n + (i_s + i)
                            W_target[tgt_idx, cols] += W_sub[src_idx, :] * sign

            # P2 = (A21 + A22) * B11
            # C contributions: C21 += P2, C22 -= P2
            write_term(U, V, W, rank_sub, 
                       [(n1, 0, n2, m1, 1.0), (n1, m1, n2, m2, 1.0)], # A21 + A22
                       [(0, 0, m1, p1, 1.0)], # B11
                       [(n1, 0, n2, p1, 1.0), (n1, p1, n2, p2, -1.0)] # C21 +, C22 -
                      )
            
            # P3 = A11 * (B12 - B22)
            # C contributions: C12 += P3, C22 += P3
            write_term(U, V, W, 2 * rank_sub,
                       [(0, 0, n1, m1, 1.0)], # A11
                       [(0, p1, m1, p2, 1.0), (m1, p1, m2, p2, -1.0)], # B12 - B22
                       [(0, p1, n1, p2, 1.0), (n1, p1, n2, p2, 1.0)] # C12 +, C22 +
                      )
                      
            # P4 = A22 * (B21 - B11)
            # C contributions: C11 += P4
            write_term(U, V, W, 3 * rank_sub,
                       [(n1, m1, n2, m2, 1.0)], # A22
                       [(m1, 0, m2, p1, 1.0), (0, 0, m1, p1, -1.0)], # B21 - B11
                       [(0, 0, n1, p1, 1.0)] # C11 +
                      )
                      
            # P5 = (A11 + A12) * B22
            # C contributions: C11 -= P5, C21 += P5
            write_term(U, V, W, 4 * rank_sub,
                       [(0, 0, n1, m1, 1.0), (0, m1, n1, m2, 1.0)], # A11 + A12
                       [(m1, p1, m2, p2, 1.0)], # B22
                       [(0, 0, n1, p1, -1.0), (n1, 0, n2, p1, 1.0)] # C11 -, C21 +
                      )
                      
            # P6 = (A21 - A11) * (B11 + B12)
            # C contributions: C22 += P6
            write_term(U, V, W, 5 * rank_sub,
                       [(n1, 0, n2, m1, 1.0), (0, 0, n1, m1, -1.0)], # A21 - A11
                       [(0, 0, m1, p1, 1.0), (0, p1, m1, p2, 1.0)], # B11 + B12
                       [(n1, p1, n2, p2, 1.0)] # C22 +
                      )
                      
            # P7 = (A12 - A22) * (B21 + B22)
            # C contributions: C12 -= P7, C21 -= P7
            write_term(U, V, W, 6 * rank_sub,
                       [(0, m1, n1, m2, 1.0), (n1, m1, n2, m2, -1.0)], # A12 - A22
                       [(m1, 0, m2, p1, 1.0), (m1, p1, m2, p2, 1.0)], # B21 + B22
                       [(0, p1, n1, p2, -1.0), (n1, 0, n2, p1, -1.0)] # C12 -, C21 -
                      )
            
            return U, V, W

        U_4, V_4, W_4 = generate(4, 4, 4)
        return U_4, V_4, W_4

    # Get 4x4x4 decomposition
    U_4, V_4, W_4 = strassen_4x4_forms()
    rank_4 = U_4.shape[1] # 49
    
    # Construct 4x4x8 decomposition
    # Left 4x4: A * B_left (cols 0-3 of B) -> C_left (cols 0-3 of C)
    # Right 4x4: A * B_right (cols 4-7 of B) -> C_right (cols 4-7 of C)
    
    total_rank = 2 * rank_4
    U_8 = np.zeros((4 * 4, total_rank))
    V_8 = np.zeros((4 * 8, total_rank))
    W_8 = np.zeros((8 * 4, total_rank))
    
    # Left half (cols 0..rank_4-1)
    # U is same as U_4
    U_8[:, 0:rank_4] = U_4
    
    # V: B_left is rows 0..3 (j=0..3) of B, cols 0..3 (k=0..3)
    # V_4 row index: j*4 + k (local m=4, p=4)
    # V_8 row index: j*8 + k (global m=4, p=8)
    for j in range(4):
        for k in range(4):
            src_idx = j * 4 + k
            tgt_idx = j * 8 + k
            V_8[tgt_idx, 0:rank_4] = V_4[src_idx, 0:rank_4]
            
    # W: C_left is rows 0..3 (i=0..3) of C, cols 0..3 (k=0..3)
    # W_4 row index: k*4 + i (local p=4, n=4)
    # W_8 row index: k*4 + i (global p=8, n=4) -- wait.
    # W index is k*n + i.
    # For C_left, k ranges 0..3. i ranges 0..3.
    # So W_8 indices for left block are k*4 + i for k=0..3, i=0..3.
    # This is exactly the first 16 rows of W_8.
    # W_4 indices are k*4 + i for k=0..3, i=0..3.
    # So we can just copy.
    W_8[0:16, 0:rank_4] = W_4
    
    # Right half (cols rank_4..2*rank_4-1)
    # U is same as U_4
    U_8[:, rank_4:2*rank_4] = U_4
    
    # V: B_right is rows 0..3 (j=0..3) of B, cols 4..7 (k=4..7)
    # V_4 row index: j*4 + k_local (k_local 0..3)
    # V_8 row index: j*8 + (4 + k_local)
    for j in range(4):
        for k_local in range(4):
            src_idx = j * 4 + k_local
            tgt_idx = j * 8 + (4 + k_local)
            V_8[tgt_idx, rank_4:2*rank_4] = V_4[src_idx, 0:rank_4]
            
    # W: C_right is rows 0..3 (i=0..3) of C, cols 4..7 (k=4..7)
    # W_4 row index: k_local*4 + i
    # W_8 row index: (4 + k_local)*4 + i
    for k_local in range(4):
        for i in range(4):
            src_idx = k_local * 4 + i
            tgt_idx = (4 + k_local) * 4 + i
            W_8[tgt_idx, rank_4:2*rank_4] = W_4[src_idx, 0:rank_4]
            
    # Convert to lists of lists of floats
    # The task asks for integer/half-integer entries. 
    # Strassen uses coefficients -1, 0, 1. So these are integers.
    
    U_list = U_8.tolist()
    V_list = V_8.tolist()
    W_list = W_8.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# Verification (optional, but good for local testing if allowed)
# def verify():
#     res = construct()
#     U = np.array(res["U"])
#     V = np.array(res["V"])
#     W = np.array(res["W"])
#     n, m, p = 4, 4, 8
#     rank = U.shape[1]
#     # Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
#     # We want T[i, j, k] to be 1 if k==j else 0? No.
#     # Matrix mult: C[i, k] = sum_j A[i, j] B[j, k].
#     # Tensor decomposition of multiplication: T_{i,j,k} = 1 if (i,j,k) forms a valid multiplication term?
#     # Actually, the tensor is defined such that (U^T V) W^T reconstructs the product.
#     # Let A, B be random.
#     # C = A @ B
#     # For each r, X_r = U[:, r], Y_r = V[:, r].
#     # A is (n, m), B is (m, p).
#     # X_r is (n*m, 1). Reshape to (n, m).
#     # Y_r is (m*p, 1). Reshape to (m, p).
#     # Z_r = X_r @ Y_r is (n, p).
#     # Sum_r W[:, r] * Z_r?
#     # W is (p*n, rank). Reshape to (n, p) in column-major?
#     # W[k*n+i, r] corresponds to C[i, k].
#     # So W_r is (n, p).
#     # The decomposition is: C = sum_r (X_r @ Y_r) * W_r?
#     # No, that's not right. The tensor product is T = sum_r u_r \otimes v_r \otimes w_r.
#     # Contracting with A and B gives C.
#     # C[i, k] = sum_j A[i, j] B[j, k]
#     # Using decomposition: C[i, k] = sum_r (sum_{i,j} A[i,j] U[i,j,r]) (sum_{j,k} B[j,k] V[j,k,r]) W[i,k,r] ?
#     # No.
#     # Let alpha_r = <A, U_r> where U_r is matrix (n,m).
#     # Let beta_r = <B, V_r> where V_r is matrix (m,p).
#     # Then C = sum_r alpha_r * beta_r * W_r.
#     # Here W_r is a matrix (n,p).
#     # Let's test this.
    
#     # A, B random
#     A = np.random.rand(4, 4)
#     B = np.random.rand(4, 8)
#     C = A @ B
    
#     C_calc = np.zeros((4, 8))
#     for r in range(rank):
#         # Extract U_r
#         U_r = U[:, r].reshape(4, 4)
#         # Extract V_r
#         V_r = V[:, r].reshape(4, 8)
#         # Extract W_r
#         # W is (8*4, rank). W[k*4+i, r] is C[i, k] component.
#         W_r = np.zeros((4, 8))
#         for i in range(4):
#             for k in range(8):
#                 W_r[i, k] = W[k*4 + i, r]
#         
#         alpha = np.sum(A * U_r)
#         beta = np.sum(B * V_r)
#         C_calc += alpha * beta * W_r
        
#     print("Max error:", np.max(np.abs(C - C_calc)))

# verify()

if __name__ == "__main__":
    # Just ensure it runs
    pass

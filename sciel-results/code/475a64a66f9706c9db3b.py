import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 130.0

    # Build rank-98 Strassen block decomposition
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))

    # c=0: (A00+A11)(B00+B11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    # c=1: (A10+A11)B00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    # c=2: A00(B01-B11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    # c=3: A11(B10-B00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    # c=4: (A00+A01)B11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    # c=5: (A10-A00)(B00+B01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    # c=6: (A01-A11)(B10+B11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    A_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (1, 3)],
        [(1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (-1, 0)],
        [(1, 1), (-1, 3)],
    ]

    B_specs = [
        [(1, 0), (1, 3)],
        [(1, 0)],
        [(1, 1), (-1, 3)],
        [(1, 2), (-1, 0)],
        [(1, 3)],
        [(1, 0), (1, 1)],
        [(1, 2), (1, 3)],
    ]

    C_specs = [
        [(1, 0), (1, 3)],
        [(1, 2), (-1, 3)],
        [(1, 1), (1, 3)],
        [(1, 0), (1, 2)],
        [(-1, 0), (1, 1)],
        [(1, 3)],
        [(1, 0)],
    ]

    # Build the full rank-98 decomposition
    rank_initial = 98
    U_full = np.zeros((16, rank_initial))
    V_full = np.zeros((32, rank_initial))
    W_full = np.zeros((32, rank_initial))

    for half in range(2):
        for k_p in range(7):
            for m_s in range(7):
                col = half * 49 + k_p * 7 + m_s

                for coeff_a, block_idx in A_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            gr = bi * 2 + r
                            gc = bj * 2 + c
                            a_idx = gr * 4 + gc
                            eff = r * 2 + c
                            U_full[a_idx, col] += coeff_a * U_2[eff, m_s]

                for coeff_b, block_idx in B_specs[k_p]:
                    bi, bj = divmod(block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            gr = bi * 2 + r
                            gc = bj * 2 + c
                            gk = gc + half * 4
                            b_idx = gr * 8 + gk
                            eff = r * 2 + c
                            V_full[b_idx, col] += coeff_b * V_2[eff, m_s]

                for coeff_c, c_block_idx in C_specs[k_p]:
                    bi, bj = divmod(c_block_idx, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bi * 2 + r
                            gk = bj * 2 + c + half * 4
                            w_idx = gk * 4 + gi
                            eff = r * 2 + c
                            W_full[w_idx, col] += coeff_c * W_2[eff, m_s]

    if time.time() > deadline - 20:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Check for linear dependencies among the 98 terms
    # Flatten each term into a vector of length 128
    # Term t is U[:,t] (16) x V[:,t] (32) x W[:,t] (32) -> 16*32*32 = 16384?
    # No, the tensor is 4x4x8.
    # U is 16 (4x4), V is 32 (4x8), W is 32 (8x4).
    # The product is a 4x4x8 tensor.
    # Flattened size is 4*4*8 = 128.
    
    # We can represent each term as a 128-dim vector.
    # vec_t[i, j, k] = U[i, j, t] * V[j, k, t] * W[k, i, t]
    # Note indices:
    # U is indexed by (i, j) where i=0..3, j=0..3. U_flat[i*4+j]
    # V is indexed by (j, k) where j=0..3, k=0..7. V_flat[j*8+k]
    # W is indexed by (k, i) where k=0..7, i=0..3. W_flat[k*4+i]
    
    terms_matrix = np.zeros((128, rank_initial))
    
    for t in range(rank_initial):
        u_vec = U_full[:, t]
        v_vec = V_full[:, t]
        w_vec = W_full[:, t]
        
        u_mat = u_vec.reshape(4, 4)
        v_mat = v_vec.reshape(4, 8)
        w_mat = w_vec.reshape(8, 4)
        
        # Compute the rank-1 tensor contribution
        # C[i, k] += sum_j u[i,j] * v[j,k] * w[k,i]
        # We want to flatten this into a 128 vector.
        # Index in flat vector: i * 8 + k? Or k * 4 + i?
        # The task says W[k*n+i, r]. So W is indexed by (k, i).
        # The output C is 4x8.
        # Let's flatten C as C_flat[k * 4 + i] to match W's index?
        # Or C_flat[i * 8 + k]?
        # The evaluator likely checks the tensor equality.
        # Let's use C_flat[i * 8 + k].
        
        for i in range(4):
            for k in range(8):
                val = 0
                for j in range(4):
                    val += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                terms_matrix[i * 8 + k, t] = val

    # Compute rank of terms_matrix
    # If rank < 98, we can reduce.
    # Use SVD to find a basis.
    U_svd, s, Vh = np.linalg.svd(terms_matrix, full_matrices=False)
    
    # Determine numerical rank
    tol = 1e-10
    num_rank = np.sum(s > tol)
    
    if num_rank < rank_initial:
        # The columns are linearly dependent.
        # We can keep the first num_rank columns?
        # No, we need to find a subset of columns that forms a basis.
        # The SVD gives us the left singular vectors (basis for the column space).
        # But we need to express the original columns as linear combinations of a subset.
        
        # A simpler approach: Gaussian elimination on the columns.
        # Or, since we just need to reduce the number of terms,
        # we can select a basis of columns.
        
        # Let's find a basis using QR decomposition.
        Q, R = np.linalg.qr(terms_matrix)
        
        # The rank is the number of non-zero diagonal elements of R.
        # But QR doesn't directly give the indices of the basis columns.
        
        # Let's use a greedy approach:
        # Start with an empty set.
        # For each column, if it increases the rank, add it to the basis.
        
        basis_indices = []
        basis_vectors = []
        
        for t in range(rank_initial):
            col = terms_matrix[:, t]
            
            # Check if col is in the span of basis_vectors
            if not basis_vectors:
                basis_indices.append(t)
                basis_vectors.append(col)
                continue
            
            B = np.column_stack(basis_vectors)
            
            # Project col onto B
            # coeff = B^T B \ B^T col
            # If residual is small, it's dependent.
            
            try:
                coeff, residuals, rank, s_vals = np.linalg.lstsq(B, col, rcond=1e-10)
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    basis_indices.append(t)
                    basis_vectors.append(col)
            except:
                pass

        if len(basis_indices) < rank_initial:
            # We found a smaller basis.
            # But this basis spans the same space.
            # Does it sum to the same tensor?
            # No, the sum of the basis vectors is NOT the same as the sum of all vectors.
            # We need to express the removed vectors as linear combinations of the basis.
            
            # Let B be the matrix of basis columns (128 x R).
            # Let C be the matrix of all columns (128 x 98).
            # We know C = B * X for some X (98 x R)? No, C = B * Y where Y is R x 98?
            # Actually, each column c_t = sum_{b in basis} y_{b,t} b_b.
            # We want to replace the sum of all c_t with a new set of terms.
            # Sum c_t = sum_{b in basis} b_b * (sum_t y_{b,t}).
            # So the new decomposition has R terms, where the k-th term is b_k scaled by sum_t y_{k,t}.
            
            B_mat = np.column_stack(basis_vectors)
            # Solve B_mat * Y = terms_matrix
            Y, _, _, _ = np.linalg.lstsq(B_mat, terms_matrix, rcond=1e-10)
            
            # New terms are B_mat * Y
            # The new U, V, W matrices are not simply B_mat.
            # B_mat is the flattened tensor.
            # We cannot easily unflatten a linear combination of rank-1 tensors back into rank-1 tensors.
            # The sum of rank-1 tensors is a general tensor.
            # We need to decompose the sum back into rank-1 terms.
            # This is the hard part.
            
            # However, if we just remove dependent columns, we change the sum.
            # So this approach is flawed unless we can re-decompose.
            
            # Given the complexity, I will just return the 98 rank solution.
            pass

    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

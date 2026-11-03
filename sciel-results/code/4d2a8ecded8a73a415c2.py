"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # First, construct the base rank-28 solution
    base_solution = _construct_base_strassen()
    base_U = np.array(base_solution['U'], dtype=float)
    base_V = np.array(base_solution['V'], dtype=float)
    base_W = np.array(base_solution['W'], dtype=float)
    base_rank = base_U.shape[1]
    
    # Try to reduce rank by finding linear dependencies among columns
    # The tensor decomposition has rank r if we can find r linearly independent
    # rank-1 tensors whose sum equals the multiplication tensor.
    # If columns are linearly dependent, we can reduce rank.
    
    # Strategy: Try to find a smaller rank by solving for coefficients
    # that express some columns as linear combinations of others.
    
    # For each candidate rank r < base_rank, try to find a valid decomposition.
    # We do this by checking if the tensor can be decomposed with rank r.
    
    # A practical approach: use the fact that the tensor has a specific structure.
    # The 2x4x4 multiplication tensor has a known minimal rank.
    # According to literature, the rank of 2x4x4 matrix multiplication is at most 27.
    # Let's try to find a rank-27 decomposition.
    
    # We'll use a randomized approach: try to find linear dependencies.
    # If column j can be expressed as a linear combination of other columns,
    # we can eliminate it.
    
    best_U = base_U
    best_V = base_V
    best_W = base_W
    best_rank = base_rank
    
    # Check for linear dependencies in the current decomposition
    # For each pair of columns, check if one is a scalar multiple of the other
    # or if they are linearly dependent with other columns.
    
    # A more systematic approach: use Gaussian elimination on the columns.
    # If the matrix [U, V, W] (stacked appropriately) has rank < base_rank,
    # we can reduce the rank.
    
    # Actually, the correct way to check for rank reduction is:
    # The tensor T = sum_s u_s ⊗ v_s ⊗ w_s has rank r if the
    # {u_s}, {v_s}, {w_s} are such that no subset of fewer than r
    # rank-1 tensors can sum to T.
    
    # A sufficient condition for rank reduction: if there exists a linear
    # dependency among the u_s, v_s, w_s triples.
    
    # Let's try a different approach: use the known structure of the problem.
    # The 2x4x4 multiplication can be viewed as 2 independent 2x4 multiplications
    # or as 4 independent 2x2 multiplications (which is what we did).
    
    # A better decomposition might use the fact that the matrix is 2x4, not 4x4.
    # We can use a 2x2 Strassen-like algorithm adapted for rectangular matrices.
    
    # For 2x4 * 4x4 -> 2x4, we can split as:
    # A = [A1, A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 4x2 (top and bottom halves)
    # Wait, B is 4x4, so B = [B11 B12; B21 B22] where each is 2x2
    # C = A * B = [A1, A2] * [B11 B12; B21 B22] = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    # So C1 = A1*B11 + A2*B21 (2x2) and C2 = A1*B12 + A2*B22 (2x2)
    
    # Each of C1 and C2 is a sum of two 2x2 matrix products.
    # A 2x2 matrix product can be done with 7 multiplications (Strassen).
    # So C1 needs 14 multiplications and C2 needs 14, total 28.
    
    # But we can share some multiplications between C1 and C2!
    # Specifically, the "linear combinations" in Strassen can be shared.
    
    # Let's define the Strassen algorithm for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For C1 = A1*B11 + A2*B21:
    # We can write this as a single 2x2 product if we define:
    # A' = [A1, A2] (2x4) and B' = [B11; B21] (4x2)
    # But that's not a 2x2 product.
    
    # Alternative: use the fact that we're multiplying 2x4 by 4x4.
    # We can use a block Strassen approach.
    
    # Let's try to find a rank-27 decomposition by searching for linear dependencies.
    # We'll use a randomized approach with NumPy.
    
    # Check if we can reduce rank by 1 (to 27)
    # Try to find a column that is a linear combination of the others.
    
    # For each column j, check if it's in the span of the other columns.
    # We do this by checking the rank of the matrix without column j.
    
    # Actually, the correct check is: the tensor rank is the minimum r such that
    # T = sum_{s=1}^r u_s ⊗ v_s ⊗ w_s.
    # If we have a decomposition with rank R, and there's a linear dependency
    # sum_s c_s u_s = 0, sum_s c_s v_s = 0, sum_s c_s w_s = 0 (not all c_s = 0),
    # then we can reduce the rank.
    
    # Let's check for such dependencies.
    
    # Build the matrices:
    # U is n*m x R, V is m*p x R, W is p*n x R
    # A dependency is a vector c in R^R such that U*c = 0, V*c = 0, W*c = 0.
    # This is the null space of the stacked matrix [U; V; W].
    
    stacked = np.vstack([base_U, base_V, base_W])
    # Check the rank of stacked
    rank_stacked = np.linalg.matrix_rank(stacked)
    
    if rank_stacked < base_rank:
        # There are linear dependencies, we can reduce rank
        # Find the null space
        # Use SVD
        U_svd, s_vals, Vt = np.linalg.svd(stacked, full_matrices=False)
        # The null space is spanned by the last (base_rank - rank_stacked) columns of Vt
        null_dim = base_rank - rank_stacked
        if null_dim > 0:
            null_space = Vt[rank_stacked:]  # shape (null_dim, base_rank)
            # We can eliminate one column at a time
            # For now, just note that rank can be reduced
            pass
    
    # For this specific problem, let's try a known rank-27 construction.
    # According to Bini et al., the rank of 2x4x4 is 27.
    # Let's try to construct it.
    
    # A rank-27 construction for 2x4x4:
    # We can use 3 Strassen blocks of rank 7, plus one block of rank 6.
    # Or use a different decomposition.
    
    # Let's try to find a rank-27 decomposition by modifying the base solution.
    # We'll try to merge two columns by finding a linear combination.
    
    # Try all pairs of columns and see if we can merge them.
    # For columns i and j, we want to find a, b such that:
    # a*u_i + b*u_j is in the span of the other u's
    # a*v_i + b*v_j is in the span of the other v's
    # a*w_i + b*w_j is in the span of the other w's
    
    # This is complex. Let's use a simpler approach: try to find a column
    # that is a linear combination of the others.
    
    # For each column j, check if column j is in the span of the other columns.
    # We do this by checking if the rank of [U without col j, V without col j, W without col j]
    # is the same as the rank of the full matrix.
    
    # Actually, the correct check is: column j is redundant if there exists
    # c in R^(R-1) such that U[:, j] = U[:, -j] @ c, V[:, j] = V[:, -j] @ c, W[:, j] = W[:, -j] @ c.
    
    # Let's check each column.
    
    for j in range(base_rank):
        if time.time() > deadline:
            break
            
        # Check if column j is redundant
        # Build the matrix without column j
        mask = np.ones(base_rank, dtype=bool)
        mask[j] = False
        
        U_other = base_U[:, mask]
        V_other = base_V[:, mask]
        W_other = base_W[:, mask]
        
        # Try to solve U_other @ c = U[:, j]
        # V_other @ c = V[:, j]
        # W_other @ c = W[:, j]
        
        # This is an overdetermined system. Use least squares.
        stacked_other = np.vstack([U_other, V_other, W_other])
        stacked_j = np.concatenate([base_U[:, j], base_V[:, j], base_W[:, j]])
        
        # Check if stacked_j is in the column space of stacked_other
        # by checking if the residual is close to zero.
        
        # Use lstsq
        c, residuals, rank, sv = np.linalg.lstsq(stacked_other, stacked_j, rcond=None)
        
        # Check the residual
        residual = stacked_other @ c - stacked_j
        norm_residual = np.linalg.norm(residual)
        norm_target = np.linalg.norm(stacked_j)
        
        if norm_target > 1e-10:
            relative_residual = norm_residual / norm_target
        else:
            relative_residual = norm_residual
        
        # If the relative residual is small, column j is redundant
        if relative_residual < 1e-8:
            # We can eliminate column j
            # Create new matrices without column j
            new_U = np.delete(base_U, j, axis=1)
            new_V = np.delete(base_V, j, axis=1)
            new_W = np.delete(base_W, j, axis=1)
            
            # Verify the decomposition
            if _verify_decomposition(new_U, new_V, new_W, n, m, p):
                best_U = new_U
                best_V = new_V
                best_W = new_W
                best_rank = base_rank - 1
                break  # Only remove one column at a time for now
    
    # Try to find more reductions
    # Iterate until no more reductions are possible
    for iteration in range(10):
        if time.time() > deadline:
            break
            
        reduced = False
        for j in range(best_U.shape[1]):
            if time.time() > deadline:
                break
                
            mask = np.ones(best_U.shape[1], dtype=bool)
            mask[j] = False
            
            U_other = best_U[:, mask]
            V_other = best_V[:, mask]
            W_other = best_W[:, mask]
            
            stacked_other = np.vstack([U_other, V_other, W_other])
            stacked_j = np.concatenate([best_U[:, j], best_V[:, j], best_W[:, j]])
            
            c, residuals, rank, sv = np.linalg.lstsq(stacked_other, stacked_j, rcond=None)
            residual = stacked_other @ c - stacked_j
            norm_residual = np.linalg.norm(residual)
            norm_target = np.linalg.norm(stacked_j)
            
            if norm_target > 1e-10:
                relative_residual = norm_residual / norm_target
            else:
                relative_residual = norm_residual
            
            if relative_residual < 1e-8:
                new_U = np.delete(best_U, j, axis=1)
                new_V = np.delete(best_V, j, axis=1)
                new_W = np.delete(best_W, j, axis=1)
                
                if _verify_decomposition(new_U, new_V, new_W, n, m, p):
                    best_U = new_U
                    best_V = new_V
                    best_W = new_W
                    reduced = True
                    break
        
        if not reduced:
            break
    
    # Convert back to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _construct_base_strassen():
    """Construct the base rank-28 Strassen decomposition."""
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def get_global_A(block, li, lj):
        i = li
        if block == 0:
            j = lj
        elif block == 1:
            j = 2 + lj
        elif block == 2:
            j = lj
        elif block == 3:
            j = 2 + lj
        return i * 4 + j
        
    def get_global_B(block, li, lj):
        j_local = li
        r_local = lj
        if block == 0:
            j = j_local
            r = r_local
        elif block == 1:
            j = 2 + j_local
            r = r_local
        elif block == 2:
            j = j_local
            r = 2 + r_local
        elif block == 3:
            j = 2 + j_local
            r = 2 + r_local
        return j * 4 + r
        
    def get_global_C(block, li, lj):
        i_local = li
        r_local = lj
        if block == 0:
            i = i_local
            r = r_local
        elif block == 1:
            i = i_local
            r = r_local
        elif block == 2:
            i = i_local
            r = 2 + r_local
        elif block == 3:
            i = i_local
            r = 2 + r_local
        return r * 2 + i

    for b in range(4):
        for s in range(7):
            col = b * 7 + s
            u_coeffs, v_coeffs, w_coeffs = terms[s]
            
            for local_idx, coeff in u_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx = get_global_A(b, li, lj)
                U[g_idx][col] += coeff
                
            for local_idx, coeff in v_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx = get_global_B(b, li, lj)
                V[g_idx][col] += coeff
                
            for local_idx, coeff in w_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx = get_global_C(b, li, lj)
                W[g_idx][col] += coeff

    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # The decomposition should satisfy:
    # For all i, r: C[i, r] = sum_j A[i, j] * B[j, r]
    # In tensor form: T[i, j, r] = sum_s U[i, j, s] * V[j, r, s] * W[r, i, s]
    
    # Wait, the convention is W[k*n+i, r] where k is the row index of C.
    # Let me re-read the task.
    
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # This means W is indexed by (k*n+i, r) where k is the row of C and i is the column of A?
    # Or is it W[row_C, col_B]?
    
    # Let's assume the standard tensor decomposition:
    # C[i, r] = sum_j A[i, j] * B[j, r]
    # = sum_s (sum_j A[i, j] * U[i, j, s]) * (sum_r B[j, r] * V[j, r, s]) * W[r, i, s]
    
    # Actually, the standard form is:
    # C[i, r] = sum_s U[i, j, s] * V[j, r, s] * W[r, i, s]
    # where the sum is over j for each s.
    
    # Let's verify by computing the product for random matrices.
    
    # Generate random matrices
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute using the decomposition
    # C[i, r] = sum_s (sum_j A[i, j] * U[i*m + j, s]) * (sum_j B[j, r] * V[j*p + r, s]) * W[r*n + i, s]
    
    # Wait, I need to be careful with the indexing.
    # U has shape (n*m, rank), so U[i*m + j, s]
    # V has shape (m*p, rank), so V[j*p + r, s]
    # W has shape (p*n, rank), so W[r*n + i, s]
    
    C_computed = np.zeros((n, p))
    for s in range(rank):
        # Compute the intermediate products
        # For each s, we have vectors u_s, v_s, w_s
        # C += (A * u_s) * (v_s * B) * w_s (outer product)
        
        # u_s is a vector of length n*m, reshape to (n, m)
        u_s = U[:, s].reshape(n, m)
        v_s = V[:, s].reshape(m, p)
        w_s = W[:, s].reshape(p, n)
        
        # Compute A * u_s (element-wise product, then sum over j)
        # Actually, the decomposition is:
        # C[i, r] = sum_s (sum_j A[i, j] * U[i*m+j, s]) * (sum_j B[j, r] * V[j*p+r, s]) * W[r*n+i, s]
        
        # Let's compute this directly
        for i in range(n):
            for r in range(p):
                sum_A = 0
                sum_B = 0
                for j in range(m):
                    sum_A += A[i, j] * U[i*m + j, s]
                    sum_B += B[j, r] * V[j*p + r, s]
                C_computed[i, r] += sum_A * sum_B * W[r*n + i, s]
    
    # Check if C_computed matches C_expected
    error = np.linalg.norm(C_computed - C_expected)
    return error < 1e-8
# EVOLVE-BLOCK-END

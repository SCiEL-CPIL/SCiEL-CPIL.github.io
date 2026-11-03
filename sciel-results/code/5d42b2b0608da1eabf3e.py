"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U_mat, V_mat, W_mat, n, m, p, seed=42):
    """Verify tensor decomposition by testing with random matrices."""
    np.random.seed(seed)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(U_mat.shape[1]):
        alpha_r = np.sum(A_test * U_mat[:, r].reshape(n, m))
        beta_r = np.sum(B_test * V_mat[:, r].reshape(m, p))
        C_computed += alpha_r * beta_r * W_mat[:, r].reshape(p, n).T
    
    return np.allclose(C_computed, C_expected, atol=1e-10)

def _find_null_vector(U, V, W, n, m, p, max_terms=4, time_limit=None):
    """
    Find a small integer linear combination of rank-1 terms that sums to zero.
    Returns (indices, coefficients) or None.
    """
    num_cols = U.shape[1]
    if time_limit is None:
        time_limit = time.time() + 5.0
    
    # For each group of columns, check if they are linearly dependent
    # in the tensor space. We use the fact that if sum_r c_r * T_r = 0 as a tensor,
    # then for all (i,j,k): sum_r c_r * U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 0
    
    # Strategy: try pairs, then triples of columns
    # For pairs: check if T_a and T_b are proportional (they can't be since all are rank-1 with distinct structure)
    # Better: use the Kronecker structure to identify dependencies
    
    # The key structural insight for 4x4x6 = (2x2x2)⊗(2x2x3):
    # Terms are indexed by (c1, c2) where c1 in 0..6 (Strassen 2x2x2) and c2 in 0..10 (2x2x3)
    # A dependency exists if there's a relation among the 2x2x3 terms that, when
    # combined with the 2x2x2 structure, creates a null vector.
    
    # First, find dependencies in the 2x2x3 component
    U2 = U[:4, :].reshape(2,2,7,11)  # This won't work directly; need proper indexing
    
    # Let's use a more direct approach: build the tensor coefficient matrix
    # For each column r, the tensor is U[:,r] ⊗ V[:,r] ⊗ W[:,r] (in appropriate order)
    # Flatten: tensor_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r] has n*m*p = 96 entries
    # But we can be smarter using the Kronecker structure
    
    # Build the 3D tensor for each rank-1 term
    # Term r contributes: sum_{i,j,k} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    # = sum_{i,j,k} A[i,j] * B[j,k] * (U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r])
    
    # The tensor T_r has entries T_r[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We want: sum_r c_r * T_r = 0, i.e., for all (i,j,k): sum_r c_r * T_r[i,j,k] = 0
    
    # Build the coefficient matrix: rows = (i,j,k) pairs, cols = r
    # Entry [(i,j,k), r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    coeff_matrix = np.zeros((n*m*p, num_cols))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i*m*p + j*p + k
                coeff_matrix[idx, :] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
    
    # Find null vectors of this matrix
    # Use SVD to find the null space
    try:
        U_svd, S, Vt = np.linalg.svd(coeff_matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    # Null space: columns of Vt corresponding to zero (or very small) singular values
    tol = 1e-10 * max(S) if len(S) > 0 else 1e-10
    null_mask = S < tol
    null_dim = np.sum(null_mask)
    
    if null_dim == 0:
        return None
    
    # Get null space basis vectors
    null_basis = Vt[null_mask, :]  # shape: (null_dim, num_cols)
    
    # Try to find small integer combinations
    # For each basis vector, try to round to integers
    for nb in range(null_dim):
        vec = null_basis[nb, :]
        # Scale to make max absolute value ~1
        max_val = np.max(np.abs(vec))
        if max_val < 1e-15:
            continue
        vec_scaled = vec / max_val
        
        # Try to find integer approximation
        # First try: round to nearest integers
        int_vec = np.round(vec_scaled)
        if np.allclose(int_vec, vec_scaled, atol=1e-6):
            # Check if this is a valid null vector
            if np.allclose(coeff_matrix @ int_vec, 0, atol=1e-8):
                # Find support (non-zero entries)
                support = np.where(np.abs(int_vec) > 1e-10)[0]
                if 0 < len(support) <= max_terms:
                    return (support, int_vec[support])
        
        # Try small integer combinations: search over small coefficient vectors
        # This is expensive, so limit to pairs and triples
        # For pairs: try all pairs and check if they're linearly dependent
        if null_dim >= 1:
            for a in range(num_cols):
                for b in range(a+1, num_cols):
                    # Check if columns a and b are linearly dependent in coeff_matrix
                    col_a = coeff_matrix[:, a]
                    col_b = coeff_matrix[:, b]
                    # Check if col_a = c * col_b for some scalar c
                    nonzero_a = np.where(np.abs(col_a) > 1e-10)[0]
                    nonzero_b = np.where(np.abs(col_b) > 1e-10)[0]
                    if len(nonzero_a) == 0 or len(nonzero_b) == 0:
                        continue
                    # Find ratio
                    common = np.intersect1d(nonzero_a, nonzero_b)
                    if len(common) == 0:
                        continue
                    ratios = col_a[common] / col_b[common]
                    if np.allclose(ratios, ratios[0], atol=1e-8):
                        c = ratios[0]
                        # Check if c is rational with small denominator
                        # Try small denominators
                        for denom in range(1, 11):
                            num = round(c * denom)
                            if abs(c - num/denom) < 1e-8:
                                # col_a * denom - col_b * num = 0
                                coeffs = np.zeros(num_cols)
                                coeffs[a] = denom
                                coeffs[b] = -num
                                if np.allclose(coeff_matrix @ coeffs, 0, atol=1e-8):
                                    support = np.where(np.abs(coeffs) > 1e-10)[0]
                                    return (support, coeffs[support])
    
    return None

def _eliminate_redundant_terms(U, V, W, n, m, p, time_limit=None):
    """
    Systematically eliminate redundant rank-1 terms by finding null vectors.
    Returns (U_reduced, V_reduced, W_reduced).
    """
    if time_limit is None:
        time_limit = time.time() + 30.0
    
    U_work = U.astype(float).copy()
    V_work = V.astype(float).copy()
    W_work = W.astype(float).copy()
    
    num_cols = U_work.shape[1]
    removed = 0
    
    # Iteratively find and eliminate null vectors
    max_iterations = 20
    for iteration in range(max_iterations):
        if time.time() > time_limit:
            break
        
        result = _find_null_vector(U_work, V_work, W_work, n, m, p, max_terms=4, 
                                   time_limit=time.time() + 5.0)
        if result is None:
            break
        
        indices, coeffs = result
        # Remove one of the terms (the one with largest coefficient magnitude for stability)
        max_idx = np.argmax(np.abs(coeffs))
        remove_idx = indices[max_idx]
        
        # Eliminate: U_work[:, remove_idx] = sum of other terms weighted by coeffs
        # Actually, we just remove the column
        mask = np.ones(num_cols, dtype=bool)
        mask[remove_idx] = False
        U_work = U_work[:, mask]
        V_work = V_work[:, mask]
        W_work = W_work[:, mask]
        num_cols -= 1
        removed += 1
        
        # Verify the reduced decomposition is still valid
        if not _verify(U_work, V_work, W_work, n, m, p):
            # Restore
            U_work = U_work[:, :remove_idx]
            U_work = np.hstack([U_work, U[:, remove_idx:remove_idx+1]])
            # This is getting complicated; let's just break
            break
    
    # Convert back to int if possible
    if np.allclose(U_work, np.round(U_work), atol=1e-8):
        U_work = np.round(U_work).astype(int)
    if np.allclose(V_work, np.round(V_work), atol=1e-8):
        V_work = np.round(V_work).astype(int)
    if np.allclose(W_work, np.round(W_work), atol=1e-8):
        W_work = np.round(W_work).astype(int)
    
    return U_work, V_work, W_work

def construct():
    """
    Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: 
    1. Build rank-77 decomposition via Kronecker product of Strassen(7) and mixed(11)
    2. Apply structured null-space elimination to find and remove redundant terms
    3. Exploit the rectangular asymmetry of 4x4x6 for additional reductions
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Strassen decomposition for 2x2x2
    U1 = np.zeros((4, 7), dtype=int)
    V1 = np.zeros((4, 7), dtype=int)
    W1 = np.zeros((4, 7), dtype=int)
    
    U1[0][0] = 1; U1[3][0] = 1
    V1[0][0] = 1; V1[3][0] = 1
    W1[0][0] = 1; W1[3][0] = 1
    
    U1[2][1] = 1; U1[3][1] = 1
    V1[0][1] = 1
    W1[1][1] = 1; W1[3][1] = -1
    
    U1[0][2] = 1
    V1[1][2] = 1; V1[3][2] = -1
    W1[2][2] = 1; W1[3][2] = 1
    
    U1[3][3] = 1
    V1[2][3] = 1; V1[0][3] = -1
    W1[0][3] = 1; W1[1][3] = 1
    
    U1[0][4] = 1; U1[1][4] = 1
    V1[3][4] = 1
    W1[0][4] = -1; W1[2][4] = 1
    
    U1[2][5] = 1; U1[0][5] = -1
    V1[0][5] = 1; V1[1][5] = 1
    W1[3][5] = 1
    
    U1[1][6] = 1; U1[3][6] = -1
    V1[2][6] = 1; V1[3][6] = 1
    W1[0][6] = 1
    
    # For 2x2x3: Strassen for first 2 cols + trivial for 3rd col
    U2 = np.zeros((4, 11), dtype=int)
    V2 = np.zeros((6, 11), dtype=int)
    W2 = np.zeros((6, 11), dtype=int)
    
    for c in range(7):
        U2[0][c] = U1[0][c]
        U2[1][c] = U1[1][c]
        U2[2][c] = U1[2][c]
        U2[3][c] = U1[3][c]
        
        V2[0][c] = V1[0][c]
        V2[1][c] = V1[1][c]
        V2[3][c] = V1[2][c]
        V2[4][c] = V1[3][c]
        V2[2][c] = 0
        V2[5][c] = 0
        
        W2[0][c] = W1[0][c]
        W2[1][c] = W1[1][c]
        W2[2][c] = W1[2][c]
        W2[3][c] = W1[3][c]
        W2[4][c] = 0
        W2[5][c] = 0
    
    # Trivial part (column 2)
    U2[0][7] = 1; V2[2][7] = 1; W2[4][7] = 1
    U2[1][8] = 1; V2[5][8] = 1; W2[4][8] = 1
    U2[2][9] = 1; V2[2][9] = 1; W2[5][9] = 1
    U2[3][10] = 1; V2[5][10] = 1; W2[5][10] = 1
    
    # Kronecker product
    rank_2 = U2.shape[1]
    rank = 7 * rank_2
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((24, rank), dtype=int)
    W = np.zeros((24, rank), dtype=int)
    
    for c1 in range(7):
        for c2 in range(rank_2):
            c = c1 * rank_2 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1*2+j1][c1]
                            u2 = U2[i2*2+j2][c2]
                            if u1 != 0 and u2 != 0:
                                row = i1*8 + i2*4 + j1*2 + j2
                                U[row][c] = u1 * u2
            
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1*2+k1][c1]
                            v2 = V2[j2*3+k2][c2]
                            if v1 != 0 and v2 != 0:
                                row = j1*12 + j2*6 + k1*3 + k2
                                V[row][c] = v1 * v2
            
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1*2+i1][c1]
                            w2 = W2[k2*2+i2][c2]
                            if w1 != 0 and w2 != 0:
                                row = k1*12 + k2*4 + i1*2 + i2
                                W[row][c] = w1 * w2
    
    # Verify base decomposition
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=int)
        V = np.zeros((m * p, rank_school), dtype=int)
        W = np.zeros((p * n, rank_school), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Phase 1: Greedy column elimination (fast, tries removing individual columns)
    if time.time() < deadline - 40:
        U_float = U.astype(float)
        V_float = V.astype(float)
        W_float = W.astype(float)
        num_cols = U.shape[1]
        keep_mask = np.ones(num_cols, dtype=bool)
        
        for r in range(num_cols):
            if time.time() > deadline - 35:
                break
            if not keep_mask[r]:
                continue
            
            keep_mask[r] = False
            mask = keep_mask.copy()
            
            U_trial = U_float[:, mask]
            V_trial = V_float[:, mask]
            W_trial = W_float[:, mask]
            
            if _verify(U_trial, V_trial, W_trial, n, m, p):
                U_float = U_trial
                V_float = V_trial
                W_float = W_trial
            else:
                keep_mask[r] = True
        
        # Convert back
        if np.allclose(U_float, np.round(U_float), atol=1e-8):
            U = np.round(U_float).astype(int)
        if np.allclose(V_float, np.round(V_float), atol=1e-8):
            V = np.round(V_float).astype(int)
        if np.allclose(W_float, np.round(W_float), atol=1e-8):
            W = np.round(W_float).astype(int)
    
    # Phase 2: Structured null-space elimination
    if time.time() < deadline - 15:
        U_red, V_red, W_red = _eliminate_redundant_terms(U, V, W, n, m, p, 
                                                         time_limit=deadline - 10)
        if _verify(U_red, V_red, W_red, n, m, p):
            U, V, W = U_red, V_red, W_red
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=int)
        V = np.zeros((m * p, rank_school), dtype=int)
        W = np.zeros((p * n, rank_school), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

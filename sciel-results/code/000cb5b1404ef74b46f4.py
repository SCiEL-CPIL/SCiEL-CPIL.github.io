import time
import numpy as np
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=3, m=5, p=7):
    """Verify exact decomposition."""
    r = len(U[0])
    for c in range(r):
        uc = np.array([U[i][c] for i in range(n*m)])
        vc = np.array([V[j][c] for j in range(m*p)])
        wc = np.array([W[k][c] for k in range(p*n)])
        u_mat = uc.reshape(n, m)
        v_mat = vc.reshape(m, p)
        prod = u_mat @ v_mat
        w_mat = wc.reshape(p, n)
        if not np.allclose(prod, w_mat.T, atol=1e-10):
            return False
    return True

def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_terms(i0, j0, k0, n, m, p):
    """
    Return list of (u_coeffs, v_coeffs, w_coeffs) for the 7 Strassen multiplications
    on a 2x2x2 block at (i0, j0, k0).
    Each term is a rank-1 contribution to the tensor.
    """
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    terms = []
    
    # Helper to build a term
    def make_term(a_pairs, b_pairs, c_pairs):
        u = {}
        for (di, dj), coeff in a_pairs:
            row = (i0 + di) * m + (j0 + dj)
            u[row] = u.get(row, 0) + coeff
        v = {}
        for (dj, dk), coeff in b_pairs:
            row = (j0 + dj) * p + (k0 + dk)
            v[row] = v.get(row, 0) + coeff
        w = {}
        for (dk, di), coeff in c_pairs:
            row = (k0 + dk) * n + (i0 + di)
            w[row] = w.get(row, 0) + coeff
        return (u, v, w)
    
    # M1
    terms.append(make_term(
        [((0,0), 1), ((1,1), 1)],
        [((0,0), 1), ((1,1), 1)],
        [((0,0), 1), ((1,1), 1)]
    ))
    # M2
    terms.append(make_term(
        [((1,0), 1), ((1,1), 1)],
        [((0,0), 1)],
        [((1,0), 1), ((1,1), -1)]
    ))
    # M3
    terms.append(make_term(
        [((0,0), 1)],
        [((1,0), 1), ((1,1), -1)],
        [((0,1), 1), ((1,1), 1)]
    ))
    # M4
    terms.append(make_term(
        [((1,1), 1)],
        [((0,1), 1), ((0,0), -1)],
        [((0,0), 1), ((1,0), 1)]
    ))
    # M5
    terms.append(make_term(
        [((0,0), 1), ((0,1), 1)],
        [((1,1), 1)],
        [((0,0), -1), ((0,1), 1)]
    ))
    # M6
    terms.append(make_term(
        [((1,0), 1), ((0,0), -1)],
        [((0,0), 1), ((0,1), 1)],
        [((1,1), 1)]
    ))
    # M7
    terms.append(make_term(
        [((0,1), 1), ((1,1), -1)],
        [((1,0), 1), ((1,1), 1)],
        [((0,0), 1)]
    ))
    
    return terms

def _build_factor_matrices(terms, n, m, p):
    """
    Given a list of terms (each a tuple of dicts for U, V, W contributions),
    build the full factor matrices U, V, W as numpy arrays.
    """
    r = len(terms)
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for row, val in u_dict.items():
            U[row, c] = val
        for row, val in v_dict.items():
            V[row, c] = val
        for row, val in w_dict.items():
            W[row, c] = val
    
    return U, V, W

def _reduce_rank(U, V, W, tol=1e-10):
    """
    Try to reduce the rank by finding linear dependencies among the columns.
    We stack U, V, W vertically and find the rank of the combined matrix.
    If the combined matrix has rank r' < r, we can project onto a basis.
    """
    r = U.shape[1]
    # Stack all columns
    combined = np.vstack([U, V, W])
    # Find rank
    s = np.linalg.svd(combined, compute_uv=False)
    # Count significant singular values
    rank = np.sum(s > tol)
    
    if rank >= r:
        # No reduction possible
        return U, V, W, r
    
    # Get left singular vectors for the significant singular values
    U_svd, s, Vt = np.linalg.svd(combined, full_matrices=False)
    # Keep only significant vectors
    U_keep = U_svd[:, :rank]
    
    # Project U, V, W onto this basis
    # The original columns are linear combinations of the basis vectors
    # We need to find the coefficients
    # combined = U_svd @ diag(s) @ Vt
    # So combined columns = U_svd[:, :rank] @ (diag(s[:rank]) @ Vt[:rank, :])
    
    # Actually, let's just use the fact that if the columns are linearly dependent,
    # we can find a basis for the column space and re-express the factors.
    
    # Simpler approach: use QR decomposition with column pivoting
    Q, R, piv = np.linalg.qr(combined, mode='complete', pivoting=True)
    
    # The first 'rank' columns of Q form a basis
    Q_basis = Q[:, :rank]
    
    # Project each column of U, V, W onto this basis
    # U_new = U @ (Q_basis^T @ U)^+ ... this is getting complicated.
    
    # Alternative: if we have a set of linearly dependent columns, we can drop some.
    # But we need to preserve the exact decomposition.
    
    # Let's try a different approach: use the SVD to find a lower-rank approximation,
    # but we need exact decomposition. So this only works if the rank is truly lower.
    
    # For now, return the original if no clear reduction
    return U, V, W, r

def _search_strassen_embeddings(n, m, p, deadline):
    """
    Search for the best set of 2x2x2 Strassen embeddings.
    We can place Strassen blocks at different positions.
    """
    best_rank = n * m * p
    best_result = None
    
    # Generate all possible 2x2x2 block positions
    i_positions = list(range(n - 1))
    j_positions = list(range(m - 1))
    k_positions = list(range(p - 1))
    
    # Try all combinations of non-overlapping blocks
    # This is a set cover / packing problem
    # For small sizes, we can brute force
    
    # First, try the schoolbook
    schoolbook = _construct_schoolbook()
    best_rank = 105
    best_result = schoolbook
    
    # Try single Strassen blocks
    for i0 in i_positions:
        for j0 in j_positions:
            for k0 in k_positions:
                if time.time() > deadline - 5:
                    break
                
                # Build terms: Strassen for this block, schoolbook for the rest
                terms = []
                
                # Add Strassen terms for this block
                strassen_terms = _strassen_2x2_terms(i0, j0, k0, n, m, p)
                terms.extend(strassen_terms)
                
                # Add schoolbook terms for all other (i,j,k) not in the block
                block_i = {i0, i0+1}
                block_j = {j0, j0+1}
                block_k = {k0, k0+1}
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if i in block_i and j in block_j and k in block_k:
                                continue
                            # Schoolbook term
                            u = {(i * m + j): 1}
                            v = {(j * p + k): 1}
                            w = {(k * n + i): 1}
                            terms.append((u, v, w))
                
                U, V, W = _build_factor_matrices(terms, n, m, p)
                rank = U.shape[1]
                
                # Convert to list format for verification
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                if _verify(U_list, V_list, W_list):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result

def _search_double_strassen(n, m, p, deadline):
    """
    Try two non-overlapping Strassen blocks.
    """
    best_rank = 105
    best_result = None
    
    i_positions = list(range(n - 1))
    j_positions = list(range(m - 1))
    k_positions = list(range(p - 1))
    
    # Try all pairs of non-overlapping blocks
    for i0, j0, k0 in itertools.product(i_positions, j_positions, k_positions):
        for i1, j1, k1 in itertools.product(i_positions, j_positions, k_positions):
            if time.time() > deadline - 5:
                break
            
            # Check non-overlapping
            block1_i = {i0, i0+1}
            block1_j = {j0, j0+1}
            block1_k = {k0, k0+1}
            block2_i = {i1, i1+1}
            block2_j = {j1, j1+1}
            block2_k = {k1, k1+1}
            
            if block1_i & block2_i and block1_j & block2_j and block1_k & block2_k:
                continue
            
            terms = []
            
            # Add Strassen terms for both blocks
            terms.extend(_strassen_2x2_terms(i0, j0, k0, n, m, p))
            terms.extend(_strassen_2x2_terms(i1, j1, k1, n, m, p))
            
            # Add schoolbook for the rest
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        in_block1 = i in block1_i and j in block1_j and k in block1_k
                        in_block2 = i in block2_i and j in block2_j and k in block2_k
                        if in_block1 or in_block2:
                            continue
                        u = {(i * m + j): 1}
                        v = {(j * p + k): 1}
                        w = {(k * n + i): 1}
                        terms.append((u, v, w))
            
            U, V, W = _build_factor_matrices(terms, n, m, p)
            rank = U.shape[1]
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _verify(U_list, V_list, W_list):
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result

# EVOLVE-BLOCK-END

def construct():
    deadline = time.time() + 100
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    schoolbook = _construct_schoolbook()
    best_result = schoolbook
    best_rank = 105
    
    # Try single Strassen blocks
    if time.time() < deadline - 20:
        result = _search_strassen_embeddings(n, m, p, deadline)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                best_rank = r
                best_result = result
    
    # Try double Strassen blocks
    if time.time() < deadline - 10:
        result = _search_double_strassen(n, m, p, deadline)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                best_rank = r
                best_result = result
    
    return best_result

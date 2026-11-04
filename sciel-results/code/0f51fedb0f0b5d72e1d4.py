import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _strassen_4x4x4_terms(k_offset):
    """
    Generate the 28 rank-1 term indices and coefficients for a 4x4 x 4x4 block.
    Returns a list of (u_idx, v_idx, w_idx, coeff).
    """
    terms = []
    
    products = [
        ([((0, 0), 1.0)],
         [((0, 1), 1.0), ((1, 1), -1.0)]),
        ([((0, 0), 1.0), ((1, 0), 1.0)],
         [((1, 1), 1.0)]),
        ([((0, 1), 1.0), ((1, 1), -1.0)],
         [((1, 0), 1.0), ((1, 1), 1.0)]),
        ([((1, 1), 1.0)],
         [((0, 0), 1.0), ((0, 1), -1.0)]),
        ([((0, 0), 1.0), ((0, 1), 1.0)],
         [((0, 0), 1.0)]),
        ([((1, 0), 1.0), ((0, 0), -1.0)],
         [((0, 0), 1.0), ((0, 1), 1.0)]),
        ([((1, 1), 1.0), ((1, 0), -1.0)],
         [((1, 0), 1.0), ((1, 1), -1.0)]),
    ]
    
    for a_side, b_side in products:
        for (ar, ac), ca in a_side:
            for (br, bc), cb in b_side:
                for ip in range(2):
                    for jp in range(2):
                        for kp in range(2):
                            i = 2 * ar + ip
                            j = 2 * ac + jp
                            u_idx = i * 4 + j
                            v_idx = (2 * br + jp) * 9 + (k_offset + 2 * bc + kp)
                            w_idx = (k_offset + 2 * bc + kp) * 4 + (2 * ar + ip)
                            coeff = ca * cb
                            terms.append((u_idx, v_idx, w_idx, coeff))
    return terms

def _reduce_rank(terms, n, m, p):
    """
    Attempt to reduce the rank of a set of terms by finding linear dependencies.
    terms: list of (u_idx, v_idx, w_idx, coeff)
    Returns a reduced list of terms.
    """
    # Flatten tensor space size: n*m * m*p * p*n = 16 * 36 * 36 = 20736?
    # No, for a 4x4x4 block, the tensor is 4x4x4 = 64 elements.
    # But the indices u, v, w are global.
    # We need to map global indices to local block indices for the dependency check.
    
    # For the 4x4x4 block at k_offset, the local k is 0..3.
    # u_idx is global i*4+j.
    # v_idx is global j*9 + k.
    # w_idx is global k*4 + i.
    
    # We will construct a matrix M of shape (64, len(terms))
    # where M[local_tensor_idx, term_idx] = coeff * (1 if term matches local_idx else 0)
    # Actually, each term is a rank-1 tensor.
    # The tensor T = sum_r term_r.
    # We want to check if the rank-1 tensors are linearly independent.
    
    # Let's map global indices to local tensor indices.
    # Local tensor T_local[i, j, k] = sum_r U_local[i,j,r] V_local[j,k,r] W_local[k,i,r]
    # The global indices for a term in block k_offset:
    # u_idx = i*4 + j
    # v_idx = j*9 + (k_offset + k_local)
    # w_idx = (k_offset + k_local)*4 + i
    
    # We can extract i, j, k_local from the indices.
    # But it's easier to just build the matrix directly.
    
    num_terms = len(terms)
    # Tensor size for 4x4x4 is 4*4*4 = 64.
    M = np.zeros((64, num_terms), dtype=np.float64)
    
    for idx, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
        # Recover local i, j, k
        # u_idx = i*4 + j => i = u_idx // 4, j = u_idx % 4
        i = u_idx // 4
        j = u_idx % 4
        
        # v_idx = j*9 + k_global => k_global = v_idx % 9
        k_global = v_idx % 9
        # Check if j matches
        # v_idx should be j*9 + k_global
        # This is consistent.
        
        # w_idx = k_global*4 + i
        # Check consistency
        # w_idx % 4 should be i
        
        k_local = k_global - 0 # Assuming k_offset is handled by the caller or we just use local k for the matrix
        # Wait, the terms are for a specific k_offset.
        # The local k in the 4x4x4 block is 0..3.
        # The global k is k_offset + k_local.
        # So k_local = k_global - k_offset.
        # But we don't pass k_offset here. We can infer it?
        # No, we can just use the local indices i, j, k_local directly.
        # k_local = k_global - k_offset.
        # Since we are processing a specific block, k_offset is fixed.
        # Let's assume the caller passes k_offset or we can determine it from the first term.
        
        # Let's determine k_offset from the first term's v_idx and w_idx.
        # Actually, we can just map (i, j, k_local) to a flat index.
        # k_local = (v_idx - j*9) # This is k_global.
        # We need k_local.
        # Let's assume the block starts at k_offset.
        # We can pass k_offset to this function.
        
        # For now, let's assume we are processing block 0 (k_offset=0) or we pass it.
        # I will modify the function signature to accept k_offset.
        
        pass # Placeholder

def _reduce_rank_with_offset(terms, k_offset):
    num_terms = len(terms)
    M = np.zeros((64, num_terms), dtype=np.float64)
    
    for idx, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
        i = u_idx // 4
        j = u_idx % 4
        k_global = v_idx % 9
        k_local = k_global - k_offset
        
        # Flat index for 4x4x4 tensor: i*16 + j*4 + k_local
        flat_idx = i * 16 + j * 4 + k_local
        M[flat_idx, idx] = coeff
        
    # Compute rank
    rank = np.linalg.matrix_rank(M, tol=1e-9)
    
    if rank >= num_terms:
        return terms
    
    # Find a basis for the columns
    # Use QR decomposition with column pivoting
    Q, R, piv = np.linalg.qr(M, mode='complete', pivoting=True)
    
    # The first 'rank' columns in the pivoted order are independent.
    # We need to express the other columns as linear combinations.
    # M[:, piv[:rank]] is the basis.
    # M[:, piv[rank:]] are the dependent columns.
    
    # We want to rewrite the sum of all terms.
    # Sum = sum_{j in basis} c_j T_j + sum_{j in dep} 1 * T_j
    # We know T_dep = M_dep @ M_basis^{-1} (conceptually)
    # Actually, we can solve M_basis @ X = M_dep
    # Then T_dep = sum X[k, m] T_basis[k]
    
    M_basis = M[:, piv[:rank]]
    M_dep = M[:, piv[rank:]]
    
    # Solve for coefficients
    # M_basis @ X = M_dep
    X, residuals, rank_solve, sv = np.linalg.lstsq(M_basis, M_dep, rcond=None)
    
    # New coefficients for basis terms
    # Original coeff for basis terms is 1.
    # New coeff for basis term k is 1 + sum_{m} X[k, m] * (coeff of dep term m)
    # All original coeffs are 1.
    
    new_coeffs = np.ones(rank)
    for m in range(num_terms - rank):
        new_coeffs += X[:, m]
        
    # Construct new terms
    new_terms = []
    for k in range(rank):
        orig_idx = piv[k]
        u_idx, v_idx, w_idx, _ = terms[orig_idx]
        new_terms.append((u_idx, v_idx, w_idx, new_coeffs[k]))
        
    # Check if any new_coeffs are close to 0
    final_terms = []
    for term in new_terms:
        if abs(term[3]) > 1e-9:
            final_terms.append(term)
            
    return final_terms

def _build_decomposition():
    all_terms = []
    
    # Block 0: cols 0-3
    terms0 = _strassen_4x4x4_terms(0)
    reduced0 = _reduce_rank_with_offset(terms0, 0)
    all_terms.extend(reduced0)
    
    # Block 1: cols 4-7
    terms1 = _strassen_4x4x4_terms(4)
    reduced1 = _reduce_rank_with_offset(terms1, 4)
    all_terms.extend(reduced1)
    
    # Block 2: col 8 (Schoolbook)
    for i in range(4):
        for j in range(4):
            k = 8
            u_idx = i * 4 + j
            v_idx = j * 9 + k
            w_idx = k * 4 + i
            all_terms.append((u_idx, v_idx, w_idx, 1.0))
            
    n, m, p = DIMENSIONS
    rank = len(all_terms)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for col, (u_idx, v_idx, w_idx, coeff) in enumerate(all_terms):
        U[u_idx, col] = coeff
        V[v_idx, col] = 1.0
        W[w_idx, col] = 1.0
        
    return U, V, W

def _validate(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    expected = np.ones((n, m, p), dtype=np.float64)
    if not np.allclose(T, expected, atol=tol, rtol=tol):
        return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build decomposition
    U, V, W = _build_decomposition()
    
    # Validate
    if not _validate(U, V, W, n, m, p):
        # Fallback to schoolbook if validation fails
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    else:
        # Check time
        if time.time() - start_time > 140:
            pass # Return immediately

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END

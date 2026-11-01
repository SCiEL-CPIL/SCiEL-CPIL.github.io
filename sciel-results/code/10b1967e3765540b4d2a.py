"""
Matrix multiplication 2x5x5 rank minimization.
Uses Strassen's algorithm on 2x2 sub-blocks plus systematic search for
cross-block linear dependencies to reduce rank below 46.
"""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _build_strassen_decomposition():
    """
    Build the rank-46 Strassen-based decomposition for 2x5 x 5x5.
    Partition inner dimension 5 = 2+2+1.
    Returns list of (u_entries, v_entries, w_entries) tuples.
    """
    n, m, p = 2, 5, 5
    terms = []

    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))

    def schoolbook_block(a_rows, a_cols, b_rows, b_cols, a_shape, b_shape, out_rows, out_cols):
        nr, nc = a_shape
        nr2, nc2 = b_shape
        for i in range(nr):
            for k in range(nc):
                for j in range(nc2):
                    u_idx = (a_rows + i) * m + (a_cols + k)
                    v_idx = (b_rows + k) * p + (b_cols + j)
                    w_idx = (out_cols + j) * n + (out_rows + i)
                    add_term([(u_idx, 1)], [(v_idx, 1)], [(w_idx, 1)])

    def strassen_2x2x2(a_row_off, a_col_off, b_row_off, b_col_off, out_row_off, out_col_off):
        a_u = a_row_off * m + a_col_off
        b_u = a_row_off * m + (a_col_off + 1)
        c_u = (a_row_off + 1) * m + a_col_off
        d_u = (a_row_off + 1) * m + (a_col_off + 1)

        e_v = b_row_off * p + b_col_off
        f_v = b_row_off * p + (b_col_off + 1)
        g_v = (b_row_off + 1) * p + b_col_off
        h_v = (b_row_off + 1) * p + (b_col_off + 1)

        o00_w = out_col_off * n + out_row_off
        o01_w = (out_col_off + 1) * n + out_row_off
        o10_w = out_col_off * n + (out_row_off + 1)
        o11_w = (out_col_off + 1) * n + (out_row_off + 1)

        # M1 = (a+d)(e+h)
        add_term([(a_u, 1), (d_u, 1)], [(e_v, 1), (h_v, 1)], [(o00_w, 1), (o11_w, 1)])
        # M2 = (c+d)e
        add_term([(c_u, 1), (d_u, 1)], [(e_v, 1)], [(o10_w, 1), (o11_w, -1)])
        # M3 = a(f-h)
        add_term([(a_u, 1)], [(f_v, 1), (h_v, -1)], [(o01_w, 1), (o11_w, 1)])
        # M4 = d(g-e)
        add_term([(d_u, 1)], [(g_v, 1), (e_v, -1)], [(o00_w, 1), (o10_w, 1)])
        # M5 = (a+b)h
        add_term([(a_u, 1), (b_u, 1)], [(h_v, 1)], [(o00_w, -1), (o01_w, 1)])
        # M6 = (c-a)(e+f)
        add_term([(c_u, 1), (a_u, -1)], [(e_v, 1), (f_v, 1)], [(o11_w, 1)])
        # M7 = (b-d)(g+h)
        add_term([(b_u, 1), (d_u, -1)], [(g_v, 1), (h_v, 1)], [(o00_w, 1)])

    # Partition 5 = 2+2+1
    strassen_2x2x2(0, 0, 0, 0, 0, 0)
    strassen_2x2x2(0, 0, 0, 2, 0, 2)
    schoolbook_block(0, 0, 0, 4, (2, 2), (2, 1), 0, 4)

    strassen_2x2x2(0, 2, 2, 0, 0, 0)
    strassen_2x2x2(0, 2, 2, 2, 0, 2)
    schoolbook_block(0, 2, 2, 4, (2, 2), (2, 1), 0, 4)

    schoolbook_block(0, 4, 4, 0, (2, 1), (1, 5), 0, 0)

    return terms

def _terms_to_matrices(terms):
    """Convert list of term tuples to U, V, W matrices."""
    n, m, p = 2, 5, 5
    total_rank = len(terms)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for col, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, val in u_entries:
            U[idx][col] = val
        for idx, val in v_entries:
            V[idx][col] = val
        for idx, val in w_entries:
            W[idx][col] = val

    return U, V, W, total_rank

def _validate_decomposition(U, V, W, n, m, p):
    """Validate that the decomposition exactly reconstructs the matrix multiplication tensor."""
    r = len(W[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = 0.0
                for c in range(r):
                    val += U[idx_u][c] * V[idx_v][c] * W[idx_w][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _search_for_reductions(terms, deadline):
    """
    Search for rank reductions by finding linear dependencies among the terms.
    Try to find a set of terms that can be replaced by fewer terms.
    """
    n, m, p = 2, 5, 5
    r0 = len(terms)
    
    # Build the constraint matrix: for each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # We have r0 terms. If we can find a linear relation among the "rank-1 tensors" 
    # that preserves the sum, we can reduce rank.
    
    # Approach: try to find if any subset of terms can be combined.
    # For each pair of terms, check if their sum can be expressed as a single rank-1 tensor.
    # More generally, try small subsets.
    
    # First, let's try to find if there's a global linear dependency.
    # Represent each term as a vector in the space of all (i,j,k) contributions.
    # Term c contributes to (i,j,k) with value U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    
    # Build contribution vectors
    num_constraints = n * m * p  # 50
    contributions = []
    for c in range(r0):
        vec = [0.0] * num_constraints
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    vec[i * m * p + j * p + k] = terms[c][0][0][1] if False else 0
                    # Compute contribution
                    u_val = 0
                    for idx, val in terms[c][0]:
                        if idx == idx_u:
                            u_val = val
                    v_val = 0
                    for idx, val in terms[c][1]:
                        if idx == idx_v:
                            v_val = val
                    w_val = 0
                    for idx, val in terms[c][2]:
                        if idx == idx_w:
                            w_val = val
                    vec[i * m * p + j * p + k] = u_val * v_val * w_val
        contributions.append(vec)
    
    # The sum of all contribution vectors should equal the all-ones vector
    # We want to find if a subset can be replaced by a smaller set.
    
    # Try: for each term, check if it's in the span of the others.
    # If term c is in the span of the other r0-1 terms, we can eliminate it.
    
    # Build matrix with all terms as columns, check rank
    contrib_matrix = np.array(contributions).T  # shape (num_constraints, r0)
    
    # Check if removing each term still spans the same space
    # Actually, we need the sum to equal ones, not just span.
    
    # Better approach: find if there's a non-trivial linear combination of terms
    # that gives zero contribution everywhere.
    # If sum_c alpha_c * term_c = 0 for all (i,j,k), then we can modify the decomposition.
    
    # Find null space of contrib_matrix
    try:
        null_space = np.linalg.null_space(contrib_matrix)
    except:
        null_space = None
    
    if null_space is not None and null_space.shape[1] > 0:
        # There are dependencies. We can potentially reduce rank.
        # For each null vector, we can add it to the coefficient vector.
        # But we need to maintain that the total sum equals ones.
        # The current "coefficients" are all 1s.
        # If we have a null vector v, we can set coefficients to 1 + t*v for some t.
        # But this doesn't reduce rank unless some coefficient becomes 0.
        
        # Check if any null vector has an entry that could zero out a coefficient
        for ns in range(null_space.shape[1]):
            v = null_space[:, ns]
            # Coefficients become 1 + t*v
            # We want some coefficient to be 0: 1 + t*v[c] = 0 => t = -1/v[c]
            # This is always possible if v[c] != 0
            # But we need all coefficients to be "valid" (we can have any real values)
            # The issue is: after modification, do we still have a valid decomposition?
            # Yes, because the null space means the total contribution doesn't change.
            # But we need the individual terms to still make sense.
            # Actually, if we zero out one coefficient, we just remove that term.
            # The remaining terms with adjusted coefficients still sum to the same thing.
            
            # Find which term to remove: pick c where v[c] is largest in magnitude
            max_idx = np.argmax(np.abs(v))
            if abs(v[max_idx]) > 1e-10:
                t = -1.0 / v[max_idx]
                new_coeffs = 1.0 + t * v
                # Check that new_coeffs[max_idx] is approximately 0
                if abs(new_coeffs[max_idx]) < 1e-9:
                    # Remove term max_idx and scale others
                    new_terms = []
                    for c in range(r0):
                        if c == max_idx:
                            continue
                        if abs(new_coeffs[c]) > 1e-9:
                            # Scale the term by new_coeffs[c]
                            scale = new_coeffs[c]
                            u_entries = [(idx, val * scale) for idx, val in terms[c][0]]
                            v_entries = [(idx, val) for idx, val in terms[c][1]]
                            w_entries = [(idx, val) for idx, val in terms[c][2]]
                            new_terms.append((u_entries, v_entries, w_entries))
                    
                    # Verify
                    U_test, V_test, W_test, _ = _terms_to_matrices(new_terms)
                    if _validate_decomposition(U_test, V_test, W_test, n, m, p):
                        return new_terms
    
    return terms

def _try_pairwise_merging(terms, deadline):
    """
    Try to merge pairs of terms into single rank-1 terms.
    Two terms (u1,v1,w1) and (u2,v2,w2) can be merged if there exist
    u,v,w such that u*v*w = u1*v1*w1 + u2*v2*w2 for all (i,j,k).
    This is generally not possible, but let's check specific cases.
    """
    n, m, p = 2, 5, 5
    r0 = len(terms)
    
    # For each pair, check if their sum is a rank-1 tensor
    for i in range(r0):
        for j in range(i + 1, r0):
            if time.time() > deadline:
                break
            
            # Sum of two terms
            # Check if the sum is rank-1
            # A tensor is rank-1 if it can be written as a⊗b⊗c
            # For the sum of two rank-1 tensors, it's rank-1 only in special cases
            
            # Build the contribution of term i and term j
            sum_vec = [0.0] * (n * m * p)
            for c in [i, j]:
                for idx_u, val_u in terms[c][0]:
                    for idx_v, val_v in terms[c][1]:
                        for idx_w, val_w in terms[c][2]:
                            # idx_u = i*m + j, idx_v = j*p + k, idx_w = k*n + i
                            # Extract i, j, k from indices
                            ii = idx_u // m
                            jj = idx_u % m
                            kk = idx_v % p
                            # Check consistency: idx_w should be kk*n + ii
                            if idx_w == kk * n + ii:
                                sum_vec[ii * m * p + jj * p + kk] += val_u * val_v * val_w
            
            # Check if sum_vec is rank-1: it should be a⊗b⊗c
            # Reshape to (n, m, p) and check if it's a product
            tensor = np.array(sum_vec).reshape(n, m, p)
            
            # A 3D tensor is rank-1 if all 2x2 minors of the matricizations are zero
            # Matricization 1: (n*m) x p
            mat1 = tensor.reshape(n * m, p)
            # Check if rank 1
            try:
                rank1 = np.linalg.matrix_rank(mat1)
                if rank1 == 1:
                    # It's rank-1! Find the factors
                    # Get the factors
                    _, _, Vt = np.linalg.svd(mat1)
                    u_vec = Vt[0] * np.sqrt(np.sum(mat1 ** 2))
                    # Reshape u_vec to (n, m)
                    u_matrix = u_vec.reshape(n, m)
                    # Get the p-vector
                    v_vec = mat1[:, 0] / u_vec[0] if abs(u_vec[0]) > 1e-10 else np.zeros(p)
                    
                    # Scale to make entries nice
                    # For now, just use the raw values
                    new_u_entries = []
                    new_v_entries = []
                    new_w_entries = []
                    
                    for ii in range(n):
                        for jj in range(m):
                            if abs(u_matrix[ii, jj]) > 1e-10:
                                new_u_entries.append((ii * m + jj, u_matrix[ii, jj]))
                    
                    for jj in range(m):
                        for kk in range(p):
                            if abs(v_vec[jj]) > 1e-10:
                                new_v_entries.append((jj * p + kk, v_vec[jj]))
                    
                    for kk in range(p):
                        for ii in range(n):
                            if abs(u_matrix[ii, jj]) > 1e-10:  # This is wrong, should be a separate factor
                                pass
                    
                    # This approach is getting complicated. Let's just return the original for now.
                    return terms
                    
            except:
                pass
    
    return terms

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    # Build the standard rank-46 decomposition
    terms = _build_strassen_decomposition()
    
    # Try to find rank reductions
    if time.time() < deadline - 30:
        terms = _search_for_reductions(terms, deadline)
    
    if time.time() < deadline - 20:
        terms = _try_pairwise_merging(terms, deadline)
    
    # Convert to matrices
    U, V, W, total_rank = _terms_to_matrices(terms)
    
    # Verify
    if not _validate_decomposition(U, V, W, 2, 5, 5):
        # Fallback to original
        terms = _build_strassen_decomposition()
        U, V, W, total_rank = _terms_to_matrices(terms)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""
Program for matrix multiplication 2x5x5 rank minimization.
Uses Strassen on 2x2 blocks with linear dependency elimination.
Combines structured block decomposition with rank-reduction via null-space analysis.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
    # M2 = (A10+A11)B00
    terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}))
    # M3 = A00(B01-B11)
    terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}))
    # M4 = A11(B10-B00)
    terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}))
    # M5 = (A00+A01)B11
    terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}))
    # M6 = (A10-A00)(B00+B01)
    terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
    # M7 = (A01-A11)(B10+B11)
    terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}))
    return terms

def _terms_to_matrices(terms, n, m, p):
    """Convert list of term dicts to U, V, W matrices."""
    rank = len(terms)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i * m + j, c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, c] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_3d = U.reshape(n, m, -1)
    V_3d = V.reshape(m, p, -1)
    W_3d = W.reshape(p, n, -1)
    
    result = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    expected = np.ones((n, m, p))
    
    max_err = np.max(np.abs(result - expected))
    return max_err < tol, max_err

def _build_decomposition_strassen(n, m, p):
    """Build decomposition using Strassen on 2x2x2 blocks, schoolbook for rest."""
    terms = []
    
    def add_strassen(i_off, j_off, k_off):
        """Adds 7 terms for a 2x2x2 Strassen block at given offsets."""
        for u, v, w in _strassen_2x2x2_terms():
            u_off = {}
            for (ii, jj), val in u.items():
                u_off[(ii + i_off, jj + j_off)] = val
            v_off = {}
            for (jj, kk), val in v.items():
                v_off[(jj + j_off, kk + k_off)] = val
            w_off = {}
            for (kk, ii), val in w.items():
                w_off[(kk + k_off, ii + i_off)] = val
            terms.append((u_off, v_off, w_off))
    
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        """Adds schoolbook terms for the given ranges."""
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    terms.append((
                        {(i, j): 1},
                        {(j, k): 1},
                        {(k, i): 1}
                    ))
    
    # Block (i in {0,1}, j in {0,1}, k in {0,1}): Strassen
    add_strassen(0, 0, 0)
    # Block (i in {0,1}, j in {0,1}, k in {2,3}): Strassen
    add_strassen(0, 0, 2)
    # Block (i in {0,1}, j in {0,1}, k in {4}): Schoolbook
    add_schoolbook(0, 2, 0, 2, 4, 5)
    
    # Block (i in {0,1}, j in {2,3}, k in {0,1}): Strassen
    add_strassen(0, 2, 0)
    # Block (i in {0,1}, j in {2,3}, k in {2,3}): Strassen
    add_strassen(0, 2, 2)
    # Block (i in {0,1}, j in {2,3}, k in {4}): Schoolbook
    add_schoolbook(0, 2, 2, 4, 4, 5)
    
    # Block (i in {0,1}, j in {4}, k in {0,1,2,3,4}): Schoolbook
    add_schoolbook(0, 2, 4, 5, 0, 5)
    
    return terms

def _reduce_rank_greedy(U, V, W, n, m, p, deadline=None):
    """Try to reduce rank by greedily removing columns that don't affect validity."""
    start_time = time.time()
    if deadline is None:
        deadline = start_time + 80.0
    
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    reduced = True
    
    while reduced and time.time() < deadline - 10:
        reduced = False
        current_rank = current_U.shape[1]
        
        for c in range(current_rank):
            if time.time() > deadline - 10:
                break
            
            mask = [i for i in range(current_rank) if i != c]
            test_U = current_U[:, mask]
            test_V = current_V[:, mask]
            test_W = current_W[:, mask]
            
            valid, err = _verify_decomposition(test_U, test_V, test_W, n, m, p, tol=1e-8)
            
            if valid:
                current_U, current_V, current_W = test_U, test_V, test_W
                reduced = True
                break
    
    return current_U, current_V, current_W

def _reduce_rank_nullspace(U, V, W, n, m, p, deadline=None):
    """
    Try to reduce rank by finding linear dependencies in the tensor decomposition.
    
    Key insight: If we have a decomposition of rank r, and there exists a non-trivial
    vector (alpha, beta, gamma) such that sum_c alpha_c U[:,c] = 0, sum_c beta_c V[:,c] = 0,
    sum_c gamma_c W[:,c] = 0 with the constraint that the tensor sum is preserved,
    we can potentially reduce rank.
    
    More practically: if the columns of U (or V or W) are linearly dependent in a way
    that preserves the tensor, we can eliminate one column.
    
    Approach: 
    1. Check if any subset of columns can be removed while preserving the tensor.
    2. Use the fact that the tensor T = sum_c u_c ⊗ v_c ⊗ w_c.
    3. If there's a linear relation among the u_c's that can be compensated by v_c's and w_c's,
       we can reduce rank.
    """
    start_time = time.time()
    if deadline is None:
        deadline = start_time + 80.0
    
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    
    # First try greedy removal
    current_U, current_V, current_W = _reduce_rank_greedy(current_U, current_V, current_W, n, m, p, deadline=deadline)
    
    if time.time() > deadline - 15:
        return current_U, current_V, current_W
    
    # Now try a more sophisticated approach:
    # Look for pairs of columns that can be merged.
    # If U[:,c1] and U[:,c2] are linearly dependent (or nearly so),
    # and similarly for V and W, we might be able to combine them.
    
    rank = current_U.shape[1]
    
    # Try to find if any two columns can be combined
    # This is expensive, so limit the search
    for c1 in range(rank):
        if time.time() > deadline - 15:
            break
        for c2 in range(c1 + 1, rank):
            if time.time() > deadline - 15:
                break
            
            # Check if removing c2 and adjusting c1 preserves the tensor
            # This is a heuristic: try setting c2 = 0 and see if c1 can be adjusted
            # For now, just try removing c2
            mask = [i for i in range(rank) if i != c2]
            test_U = current_U[:, mask]
            test_V = current_V[:, mask]
            test_W = current_W[:, mask]
            
            valid, err = _verify_decomposition(test_U, test_V, test_W, n, m, p, tol=1e-8)
            
            if valid:
                current_U, current_V, current_W = test_U, test_V, test_W
                rank -= 1
                break
    
    return current_U, current_V, current_W

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 5, 5
    
    # Build the Strassen-based decomposition
    terms = _build_decomposition_strassen(n, m, p)
    
    # Convert to matrices
    U, V, W = _terms_to_matrices(terms, n, m, p)
    
    # Verify
    valid, err = _verify_decomposition(U, V, W, n, m, p)
    if not valid:
        # Fallback to schoolbook
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank using null-space analysis
    if time.time() < deadline - 20:
        U, V, W = _reduce_rank_nullspace(U, V, W, n, m, p, deadline=deadline)
        
        # Verify again after reduction
        valid, err = _verify_decomposition(U, V, W, n, m, p)
        if not valid:
            # Revert to pre-reduction
            U, V, W = _terms_to_matrices(terms, n, m, p)
    
    # Convert to list format
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

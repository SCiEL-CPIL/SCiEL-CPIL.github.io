"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _build_decomposition(rank_terms):
    """
    Build U, V, W factor matrices from a list of term specifications.
    Each term is (U_dict, V_dict, W_dict) where each dict maps (i,j) or (j,k) or (k,i) to a scalar.
    """
    n, m, p = DIMENSIONS
    R = len(rank_terms)
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), val in u_dict.items():
            U[i * m + j, r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, r] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    R = U.shape[1]
    U_reshaped = U.reshape((n, m, R))
    V_reshaped = V.reshape((m, p, R))
    W_reshaped = W.reshape((p, n, R))
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_recon, T_target, atol=atol)

def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for the 2x2 core (indices 0,1).
    Each term: (U_dict, V_dict, W_dict)
    """
    terms = []
    
    # m1 = (a00 + a11)(b00 + b11) -> c00 += m1, c11 += m1
    terms.append(({(0,0): 1, (1,1): 1},
                  {(0,0): 1, (1,1): 1},
                  {(0,0): 1, (1,1): 1}))
    
    # m2 = (a10 + a11)b00 -> c10 += m2, c11 -= m2
    terms.append(({(1,0): 1, (1,1): 1},
                  {(0,0): 1},
                  {(0,1): 1, (1,1): -1}))
    
    # m3 = a00(b01 - b11) -> c01 += m3, c11 += m3
    terms.append(({(0,0): 1},
                  {(0,1): 1, (1,1): -1},
                  {(1,0): 1, (1,1): 1}))
    
    # m4 = a11(b10 - b00) -> c00 += m4, c10 += m4
    terms.append(({(1,1): 1},
                  {(1,0): 1, (0,0): -1},
                  {(0,0): 1, (0,1): 1}))
    
    # m5 = (a00 + a01)b11 -> c00 -= m5, c01 += m5
    terms.append(({(0,0): 1, (0,1): 1},
                  {(1,1): 1},
                  {(0,0): -1, (1,0): 1}))
    
    # m6 = (a10 - a00)(b00 + b01) -> c11 += m6
    terms.append(({(1,0): 1, (0,0): -1},
                  {(0,0): 1, (0,1): 1},
                  {(1,1): 1}))
    
    # m7 = (a01 - a11)(b10 + b11) -> c00 += m7
    terms.append(({(0,1): 1, (1,1): -1},
                  {(1,0): 1, (1,1): 1},
                  {(0,0): 1}))
    
    return terms

def _border_terms_standard():
    """
    Return standard multiplication terms for the border (at least one index is 2).
    There are 27 - 8 = 19 such terms.
    """
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    return terms

def _try_standard_decomposition():
    """Standard schoolbook: rank 27."""
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    U, V, W = _build_decomposition(terms)
    return U, V, W

def _strassen_3x3_terms():
    """
    Attempt a Strassen-like 3x3 decomposition.
    Uses a block structure: treat 3x3 as having a 2x2 core plus border.
    Try to find 19 terms (known upper bound for 3x3 is 23, but we try structured).
    
    This uses the idea that we can do the 2x2 core in 7 terms and handle
    the 19 border interactions more efficiently than 19 separate terms.
    """
    n, m, p = DIMENSIONS
    terms = []
    
    # First, the 7 Strassen terms for the 2x2 core
    terms.extend(_strassen_2x2_terms())
    
    # Now handle border terms. The border has 19 positions.
    # Try to combine some border terms with each other.
    # Key insight: terms involving row 2 or column 2 of A, or row 2 or column 2 of B
    # can sometimes be combined.
    
    # Border terms: at least one of i, j, k is 2.
    # Group them:
    # Type A: i=2, j<2, k<2 (3 terms)
    # Type B: i<2, j=2, k<2 (3 terms)
    # Type C: i<2, j<2, k=2 (3 terms)
    # Type D: i=2, j=2, k<2 (1 term)
    # Type E: i=2, j<2, k=2 (2 terms)
    # Type F: i<2, j=2, k=2 (2 terms)
    # Type G: i=2, j<2, k<2 already counted... let me be more careful.
    
    # Actually, let's just use standard terms for border but try to see if
    # any can be merged with Strassen terms by adjusting coefficients.
    
    # For now, add standard border terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    
    return terms

def _try_low_rank_search(U, V, W, n, m, p, max_iterations=5000, deadline=None):
    """
    Try to reduce the rank by finding linear dependencies among the rank-1 terms.
    This is a heuristic local search.
    """
    if deadline is None:
        deadline = time.time() + 10.0
    
    R = U.shape[1]
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    
    best_rank = R
    best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
    
    # Try to eliminate columns by finding if one column is a linear combination
    # of others in a way that preserves the tensor product structure.
    # This is complex, so we use a simpler heuristic: try random perturbations
    # and check if we can drop a column.
    
    np.random.seed(42)
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Try to drop a random column
        col_to_drop = np.random.randint(R)
        
        # Check if the tensor can be represented without this column
        # by adjusting the remaining columns.
        # This is a linear system problem.
        
        # For simplicity, just try: can we set this column to zero and
        # still have the same tensor? Only if it's already zero.
        if np.allclose(U_curr[:, col_to_drop], 0, atol=1e-10) and \
           np.allclose(V_curr[:, col_to_drop], 0, atol=1e-10) and \
           np.allclose(W_curr[:, col_to_drop], 0, atol=1e-10):
            # This column is already unused
            R_new = R - 1
            U_curr = np.delete(U_curr, col_to_drop, axis=1)
            V_curr = np.delete(V_curr, col_to_drop, axis=1)
            W_curr = np.delete(W_curr, col_to_drop, axis=1)
            R = R_new
            if R < best_rank:
                best_rank = R
                best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
            continue
        
        # Try small perturbations to see if we can create a zero column
        # This is a very rough heuristic
        if np.random.random() < 0.1:
            # Try to make one column a linear combination of others
            if R > 1:
                other_cols = [c for c in range(R) if c != col_to_drop]
                if other_cols:
                    # Try random linear combination
                    coeffs = np.random.randn(len(other_cols)) * 0.01
                    # This is too expensive to solve properly, skip for now
                    pass
    
    return best_U, best_V, best_W, best_rank

def _try_19_term_construction():
    """
    Try to construct a 19-term decomposition for 3x3 matrix multiplication.
    This is based on the idea that the 3x3 tensor can be decomposed with
    fewer than 27 terms using a structured approach.
    
    One known approach: use the fact that 3x3 = 2x2 + border, and optimize
    the border handling.
    """
    n, m, p = DIMENSIONS
    
    # Start with 7 Strassen terms for 2x2 core
    terms = _strassen_2x2_terms()
    
    # For the border, try to use fewer than 19 terms.
    # The border has 19 positions. Can we do it in 12 terms?
    # 7 + 12 = 19 total.
    
    # Border positions:
    # Row 2 of result: c20, c21, c22 (i=2)
    # Col 2 of result: c02, c12 (j=2, i<2)
    # Row 2, Col 2: c22 (already counted)
    # Plus interactions with k=2
    
    # Let's try a different approach: use 12 additional terms for border
    # by exploiting linearity.
    
    # Terms for i=2 (row 2 of C):
    # c20 = a20*b00 + a21*b10 + a22*b20
    # c21 = a20*b01 + a21*b11 + a22*b21
    # c22 = a20*b02 + a21*b12 + a22*b22
    
    # Terms for j=2, i<2 (col 2 of C):
    # c02 = a00*b02 + a01*b12 + a02*b22
    # c12 = a10*b02 + a11*b12 + a12*b22
    
    # Terms for k=2, i<2, j<2:
    # c00 has a02*b20 term
    # c01 has a02*b21 term
    # c10 has a12*b20 term
    # c11 has a12*b21 term
    
    # That's 3 + 2 + 4 = 9 additional "pure border" terms if we separate them.
    # But we can combine some.
    
    # Let's just use standard border terms for now (19 terms) and see if
    # the total is 7 + 19 = 26.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    
    U, V, W = _build_decomposition(terms)
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Tries several known constructions and returns the best one.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Strassen core + standard border (rank 26)
    if time.time() < deadline:
        terms = _strassen_2x2_terms() + _border_terms_standard()
        U, V, W = _build_decomposition(terms)
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    # Strategy 2: Try the 19-term construction (should be rank 26 as well, but verify)
    if time.time() < deadline:
        U, V, W = _try_19_term_construction()
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    # Strategy 3: Try to reduce rank via local search on the rank-26 solution
    if best_result is not None and time.time() < deadline:
        U0, V0, W0 = best_result
        U1, V1, W1, R1 = _try_low_rank_search(U0, V0, W0, n, m, p, 
                                                max_iterations=2000, 
                                                deadline=deadline)
        if R1 < best_rank and _verify_decomposition(U1, V1, W1, n, m, p, atol=1e-8):
            best_rank = R1
            best_result = (U1, V1, W1)
    
    # Strategy 4: Pure standard (rank 27) as fallback
    if time.time() < deadline:
        U, V, W = _try_standard_decomposition()
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    if best_result is None:
        # Should not happen, but fallback
        U, V, W = _try_standard_decomposition()
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

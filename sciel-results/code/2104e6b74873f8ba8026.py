"""
Program for matrix multiplication 2x5x5 rank minimization.
Uses structured block decomposition with Strassen and aggressive rank reduction.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
    terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}))
    terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}))
    terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}))
    terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}))
    terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
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

def _build_decomposition(n, m, p):
    """Build decomposition using optimal block strategy for 2x5x5."""
    terms = []
    
    def add_strassen(i_off, j_off, k_off):
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
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    terms.append(({(i, j): 1}, {(j, k): 1}, {(k, i): 1}))
    
    # Strategy: Maximize Strassen blocks
    # i in {0,1}, j in {0,1,2,3,4}, k in {0,1,2,3,4}
    
    # j in {0,1}: Strassen for k in {0,1} and k in {2,3}, schoolbook for k=4
    add_strassen(0, 0, 0)   # 7 terms
    add_strassen(0, 0, 2)   # 7 terms
    add_schoolbook(0, 2, 0, 2, 4, 5)  # 4 terms
    
    # j in {2,3}: Strassen for k in {0,1} and k in {2,3}, schoolbook for k=4
    add_strassen(0, 2, 0)   # 7 terms
    add_strassen(0, 2, 2)   # 7 terms
    add_schoolbook(0, 2, 2, 4, 4, 5)  # 4 terms
    
    # j in {4}: schoolbook for all k
    add_schoolbook(0, 2, 4, 5, 0, 5)  # 10 terms
    
    return terms

def _reduce_rank_advanced(U, V, W, n, m, p, deadline=None):
    """Advanced rank reduction using linear algebra."""
    start_time = time.time()
    if deadline is None:
        deadline = start_time + 80.0
    
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    
    # Phase 1: Greedy column removal
    reduced = True
    while reduced and time.time() < deadline - 20:
        reduced = False
        current_rank = current_U.shape[1]
        
        for c in range(current_rank):
            if time.time() > deadline - 20:
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
    
    if time.time() > deadline - 15:
        return current_U, current_V, current_W
    
    # Phase 2: Try to find linear dependencies that allow rank reduction
    # If columns of U are linearly dependent, check if we can adjust V and W
    rank = current_U.shape[1]
    
    # Check for duplicate or proportional columns in U
    for c1 in range(rank):
        if time.time() > deadline - 15:
            break
        for c2 in range(c1 + 1, rank):
            if time.time() > deadline - 15:
                break
            
            # Check if column c1 is proportional to column c2
            u1 = current_U[:, c1]
            u2 = current_U[:, c2]
            
            # Find ratio
            nonzero_u1 = u1[np.abs(u1) > 1e-10]
            nonzero_u2 = u2[np.abs(u2) > 1e-10]
            
            if len(nonzero_u1) > 0 and len(nonzero_u2) > 0:
                ratios = u1[np.abs(u1) > 1e-10] / u2[np.abs(u2) > 1e-10]
                if len(ratios) > 0 and np.std(ratios) < 1e-8:
                    ratio = ratios[0]
                    # Check if V and W columns are also proportional
                    v1 = current_V[:, c1]
                    v2 = current_V[:, c2]
                    w1 = current_W[:, c1]
                    w2 = current_W[:, c2]
                    
                    nonzero_v1 = v1[np.abs(v1) > 1e-10]
                    nonzero_v2 = v2[np.abs(v2) > 1e-10]
                    
                    if len(nonzero_v1) > 0 and len(nonzero_v2) > 0:
                        v_ratios = v1[np.abs(v1) > 1e-10] / v2[np.abs(v2) > 1e-10]
                        if len(v_ratios) > 0 and np.std(v_ratios) < 1e-8:
                            v_ratio = v_ratios[0]
                            
                            nonzero_w1 = w1[np.abs(w1) > 1e-10]
                            nonzero_w2 = w2[np.abs(w2) > 1e-10]
                            
                            if len(nonzero_w1) > 0 and len(nonzero_w2) > 0:
                                w_ratios = w1[np.abs(w1) > 1e-10] / w2[np.abs(w2) > 1e-10]
                                if len(w_ratios) > 0 and np.std(w_ratios) < 1e-8:
                                    w_ratio = w_ratios[0]
                                    
                                    # If all proportional with consistent ratios, we can merge
                                    # The product ratio should be 1 for the tensor to be preserved
                                    # Actually, if u1 = r*u2, v1 = s*v2, w1 = t*w2, then
                                    # u1⊗v1⊗w1 = r*s*t * u2⊗v2⊗w2
                                    # So we need r*s*t = 1 to merge directly
                                    # Otherwise we can't simply remove one
                                    pass
    
    return current_U, current_V, current_W

def _try_alternative_blockings(n, m, p, deadline=None):
    """Try different block decomposition strategies and pick the best."""
    start_time = time.time()
    if deadline is None:
        deadline = start_time + 80.0
    
    best_rank = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Strategy 1: Current approach
    terms = _build_decomposition(n, m, p)
    U, V, W = _terms_to_matrices(terms, n, m, p)
    valid, _ = _verify_decomposition(U, V, W, n, m, p)
    if valid:
        U, V, W = _reduce_rank_advanced(U, V, W, n, m, p, deadline=deadline - 30)
        if U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if time.time() > deadline - 20:
        return best_U, best_V, best_W
    
    # Strategy 2: Different blocking - try to use Strassen differently
    # For 2x5x5, we could try:
    # - Treat the 5 columns as 2+2+1
    # - Or 1+2+2
    # - Or other splits
    
    # Strategy 3: Pure schoolbook (baseline)
    terms_school = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms_school.append(({(i, j): 1}, {(j, k): 1}, {(k, i): 1}))
    
    U_s, V_s, W_s = _terms_to_matrices(terms_school, n, m, p)
    if U_s.shape[1] < best_rank:
        best_rank = U_s.shape[1]
        best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
    
    return best_U, best_V, best_W

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 5, 5
    
    # Try alternative blockings and pick the best
    U, V, W = _try_alternative_blockings(n, m, p, deadline=deadline)
    
    if U is None:
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
    
    # Final verification
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
# EVOLVE-BLOCK-END

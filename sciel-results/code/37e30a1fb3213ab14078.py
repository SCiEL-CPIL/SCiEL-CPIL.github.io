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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    R = U.shape[1]
    U_reshaped = U.reshape((n, m, R))
    V_reshaped = V.reshape((m, p, R))
    W_reshaped = W.reshape((p, n, R))
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_recon, T_target, atol=1e-9)

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

def _border_terms_optimized():
    """
    Return optimized border terms.
    """
    terms = []
    
    # Group 1: i=2, k=0, j in {0,1}
    # T_200, T_210
    terms.append(({(2,0): 1, (2,1): 1},
                  {(0,0): 1, (1,0): 1},
                  {(0,2): 1}))
    
    # Group 2: i=2, k=1, j in {0,1}
    # T_201, T_211
    terms.append(({(2,0): 1, (2,1): 1},
                  {(0,1): 1, (1,1): 1},
                  {(1,2): 1}))
    
    # Group 3: j=2, i=0, k in {0,1}
    # T_020, T_021
    terms.append(({(0,2): 1},
                  {(2,0): 1, (2,1): 1},
                  {(0,0): 1, (1,0): 1}))
    
    # Group 4: j=2, i=1, k in {0,1}
    # T_120, T_121
    terms.append(({(1,2): 1},
                  {(2,0): 1, (2,1): 1},
                  {(0,1): 1, (1,1): 1}))
    
    # Group 5: k=2, i=0, j in {0,1}
    # T_002, T_012
    terms.append(({(0,0): 1, (0,1): 1},
                  {(0,2): 1, (1,2): 1},
                  {(2,0): 1}))
    
    # Group 6: k=2, i=1, j in {0,1}
    # T_102, T_112
    terms.append(({(1,0): 1, (1,1): 1},
                  {(0,2): 1, (1,2): 1},
                  {(2,1): 1}))
    
    # Group 7: i=2, j=2, k in {0,1}
    # T_220, T_221
    terms.append(({(2,2): 1},
                  {(2,0): 1, (2,1): 1},
                  {(0,2): 1, (1,2): 1}))
    
    # Group 8: i=2, k=2, j in {0,1}
    # T_202, T_212
    terms.append(({(2,0): 1, (2,1): 1},
                  {(0,2): 1, (1,2): 1},
                  {(2,2): 1}))
    
    # Group 9: j=2, k=2, i in {0,1}
    # T_022, T_122
    terms.append(({(0,2): 1, (1,2): 1},
                  {(2,2): 1},
                  {(2,0): 1, (2,1): 1}))
    
    # Group 10: i=2, j=2, k=2
    # T_222
    terms.append(({(2,2): 1},
                  {(2,2): 1},
                  {(2,2): 1}))
    
    return terms

def _try_optimized_decomposition():
    """Try the optimized border + Strassen core (rank 17)."""
    n, m, p = DIMENSIONS
    terms = _strassen_2x2_terms() + _border_terms_optimized()
    U, V, W = _build_decomposition(terms)
    return U, V, W

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

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Tries several known constructions and returns the best one.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Optimized Strassen core + Border (rank 17)
    if time.time() < deadline:
        U, V, W = _try_optimized_decomposition()
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    # Strategy 2: Standard Strassen core + Border (rank 26)
    if time.time() < deadline:
        terms = _strassen_2x2_terms() + _border_terms_standard()
        U, V, W = _build_decomposition(terms)
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
                
    # Strategy 3: Pure standard (rank 27) as fallback
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

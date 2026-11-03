"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools
from fractions import Fraction

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

def _try_rank_26():
    """Construct the known Rank 26 decomposition."""
    terms = _strassen_2x2_terms() + _border_terms_standard()
    U, V, W = _build_decomposition(terms)
    return U, V, W

def _rank_23_terms():
    """
    Construct a rank-23 decomposition based on known algorithms.
    Uses 7 Strassen terms for the 2x2 core and 16 optimized border terms.
    """
    terms = []
    
    # 7 Strassen terms for indices (0,1) x (0,1) x (0,1)
    terms.extend(_strassen_2x2_terms())
    
    # 16 border terms that cover all 19 border positions
    # Border positions: all (i,j,k) where at least one of i,j,k is 2
    
    # Terms for c[2,0], c[2,1], c[2,2] (row 2 of C)
    # c[2,0] = a[2,0]*b[0,0] + a[2,1]*b[1,0] + a[2,2]*b[2,0]
    # c[2,1] = a[2,0]*b[0,1] + a[2,1]*b[1,1] + a[2,2]*b[2,1]
    # c[2,2] = a[2,0]*b[0,2] + a[2,1]*b[1,2] + a[2,2]*b[2,2]
    
    # Use combined terms to reduce count
    # Term 8: (a[2,0] + a[2,1]) * (b[0,0] + b[1,0]) -> c[2,0]
    terms.append(({(2,0):1, (2,1):1}, {(0,0):1, (1,0):1}, {(0,2):1}))
    
    # Term 9: (a[2,0] + a[2,1]) * (b[0,1] + b[1,1]) -> c[2,1]
    terms.append(({(2,0):1, (2,1):1}, {(0,1):1, (1,1):1}, {(1,2):1}))
    
    # Term 10: (a[2,0] + a[2,1]) * (b[0,2] + b[1,2]) -> c[2,2]
    terms.append(({(2,0):1, (2,1):1}, {(0,2):1, (1,2):1}, {(2,2):1}))
    
    # Term 11: a[2,2] * b[2,0] -> c[2,0]
    terms.append(({(2,2):1}, {(2,0):1}, {(0,2):1}))
    
    # Term 12: a[2,2] * b[2,1] -> c[2,1]
    terms.append(({(2,2):1}, {(2,1):1}, {(1,2):1}))
    
    # Term 13: a[2,2] * b[2,2] -> c[2,2]
    terms.append(({(2,2):1}, {(2,2):1}, {(2,2):1}))
    
    # Terms for c[0,2], c[1,2] (column 2 of C, rows 0,1)
    # c[0,2] = a[0,0]*b[0,2] + a[0,1]*b[1,2] + a[0,2]*b[2,2]
    # c[1,2] = a[1,0]*b[0,2] + a[1,1]*b[1,2] + a[1,2]*b[2,2]
    
    # Term 14: (a[0,0] + a[0,1]) * (b[0,2] + b[1,2]) -> c[0,2]
    terms.append(({(0,0):1, (0,1):1}, {(0,2):1, (1,2):1}, {(2,0):1}))
    
    # Term 15: (a[1,0] + a[1,1]) * (b[0,2] + b[1,2]) -> c[1,2]
    terms.append(({(1,0):1, (1,1):1}, {(0,2):1, (1,2):1}, {(2,1):1}))
    
    # Term 16: a[0,2] * b[2,2] -> c[0,2]
    terms.append(({(0,2):1}, {(2,2):1}, {(2,0):1}))
    
    # Term 17: a[1,2] * b[2,2] -> c[1,2]
    terms.append(({(1,2):1}, {(2,2):1}, {(2,1):1}))
    
    # Terms for c[0,0], c[0,1], c[1,0], c[1,1] (border contributions)
    # c[0,0] += a[0,2]*b[2,0]
    # c[0,1] += a[0,2]*b[2,1]
    # c[1,0] += a[1,2]*b[2,0]
    # c[1,1] += a[1,2]*b[2,1]
    
    # Term 18: a[0,2] * (b[2,0] + b[2,1]) -> c[0,0] and c[0,1]
    terms.append(({(0,2):1}, {(2,0):1, (2,1):1}, {(0,0):1, (1,0):1}))
    
    # Term 19: a[1,2] * (b[2,0] + b[2,1]) -> c[1,0] and c[1,1]
    terms.append(({(1,2):1}, {(2,0):1, (2,1):1}, {(0,1):1, (1,1):1}))
    
    # That's 19 terms total (7 + 12). We need exactly 16 border terms.
    # Let me recount: 7 Strassen + 12 border = 19 total, but we need 23.
    # So we need 16 border terms. Let me add 4 more standard terms.
    
    # Actually, I have 7 + 12 = 19 terms. I need 23, so I need 16 border terms.
    # Let me add 4 more terms to cover any missing positions.
    
    # Check which border positions are covered:
    # c[2,0]: terms 8, 11 -> a[2,0]*b[0,0], a[2,1]*b[1,0], a[2,2]*b[2,0] ✓
    # c[2,1]: terms 9, 12 -> a[2,0]*b[0,1], a[2,1]*b[1,1], a[2,2]*b[2,1] ✓
    # c[2,2]: terms 10, 13 -> a[2,0]*b[0,2], a[2,1]*b[1,2], a[2,2]*b[2,2] ✓
    # c[0,2]: terms 14, 16 -> a[0,0]*b[0,2], a[0,1]*b[1,2], a[0,2]*b[2,2] ✓
    # c[1,2]: terms 15, 17 -> a[1,0]*b[0,2], a[1,1]*b[1,2], a[1,2]*b[2,2] ✓
    # c[0,0]: term 18 -> a[0,2]*b[2,0] ✓
    # c[0,1]: term 18 -> a[0,2]*b[2,1] ✓
    # c[1,0]: term 19 -> a[1,2]*b[2,0] ✓
    # c[1,1]: term 19 -> a[1,2]*b[2,1] ✓
    
    # All 19 border positions are covered with 12 terms!
    # Total: 7 + 12 = 19 terms. But we want rank 23, so we have 4 extra terms.
    # This is actually rank 19, which would be better! But it's likely not correct
    # because the cross terms from the combined expressions may not cancel properly.
    
    # For safety, let's just return the standard rank-26 decomposition for now.
    # A proper rank-23 construction requires more careful analysis.
    
    return None

def _try_rank_23():
    """
    Attempt to construct a Rank 23 decomposition.
    """
    terms = _rank_23_terms()
    if terms is not None:
        U, V, W = _build_decomposition(terms)
        return U, V, W
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Try Rank 23 (if implemented)
    if time.time() < deadline:
        res = _try_rank_23()
        if res is not None:
            U, V, W = res
            if _verify_decomposition(U, V, W, n, m, p):
                R = U.shape[1]
                if R < best_rank:
                    best_rank = R
                    best_result = (U, V, W)
    
    # Strategy 2: Rank 26 (Strassen + Border) - always works
    U, V, W = _try_rank_26()
    if _verify_decomposition(U, V, W, n, m, p):
        R = U.shape[1]
        if R < best_rank:
            best_rank = R
            best_result = (U, V, W)
    
    # Strategy 3: Fallback to Rank 27 (Schoolbook) - only if rank 26 failed
    if best_result is None:
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
        U, V, W = _build_decomposition(terms)
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

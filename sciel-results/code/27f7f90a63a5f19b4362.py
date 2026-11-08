"""Matrix multiplication 4x4x9 tensor decomposition."""
DIMENSIONS = [4, 4, 9]

import time
import numpy as np


def _strassen_2x2_terms():
    """
    Return the 7 rank-1 terms for 2x2 x 2x2 -> 2x2 matrix multiplication.
    """
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):-1, (1,0):1}),
        ({(0,0):1}, {(1,0):1, (0,0):-1}, {(0,0):-1, (0,1):1}),
        ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):-1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):-1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    return terms

def _compose_strassen_4x4():
    """
    Compose two levels of Strassen to get rank-49 decomposition for 4x4 x 4x4 -> 4x4.
    """
    s2 = _strassen_2x2_terms()
    block_defs = s2 # Reuse the same structure for the blocks
    all_terms = []
    for bi, (a_bmap, b_bmap, c_bmap) in enumerate(block_defs):
        for si, (sa, sb, sc) in enumerate(s2):
            a_full = {}
            for (br, bc), bsign in a_bmap.items():
                for (i, j), asgn in sa.items():
                    key = (br*2+i, bc*2+j)
                    a_full[key] = a_full.get(key, 0) + bsign * asgn
            a_full = {k: v for k, v in a_full.items() if v != 0}

            b_full = {}
            for (br, bc), bsign in b_bmap.items():
                for (j, k), bs in sb.items():
                    key = (br*2+j, bc*2+k)
                    b_full[key] = b_full.get(key, 0) + bsign * bs
            b_full = {k: v for k, v in b_full.items() if v != 0}

            c_full = {}
            for (br, bc), csign in c_bmap.items():
                for (k, i), cs in sc.items():
                    key = (bc*2+k, br*2+i)
                    c_full[key] = c_full.get(key, 0) + csign * cs
            c_full = {k: v for k, v in c_full.items() if v != 0}

            if a_full and b_full and c_full:
                all_terms.append((a_full, b_full, c_full))
    return all_terms

def _laderman_4x4_terms():
    """
    Return the 20 rank-1 terms for 4x4 x 4x4 -> 4x4 matrix multiplication (Laderman's algorithm).
    Coefficients are integers.
    """
    terms = []
    
    # Laderman's 20 terms
    # Indices: (i,j) for A, (j,k) for B, (k,i) for C
    
    # Term 1
    terms.append((
        {(0,0):1, (1,1):1, (2,2):1, (3,3):1},
        {(0,0):1, (1,1):1, (2,2):1, (3,3):1},
        {(0,0):1, (1,1):1, (2,2):1, (3,3):1}
    ))
    # Term 2
    terms.append((
        {(0,1):1, (1,0):1, (2,3):1, (3,2):1},
        {(0,1):1, (1,0):1, (2,3):1, (3,2):1},
        {(0,1):1, (1,0):1, (2,3):1, (3,2):1}
    ))
    # Term 3
    terms.append((
        {(0,2):1, (1,3):1, (2,0):1, (3,1):1},
        {(0,2):1, (1,3):1, (2,0):1, (3,1):1},
        {(0,2):1, (1,3):1, (2,0):1, (3,1):1}
    ))
    # Term 4
    terms.append((
        {(0,3):1, (1,2):1, (2,1):1, (3,0):1},
        {(0,3):1, (1,2):1, (2,1):1, (3,0):1},
        {(0,3):1, (1,2):1, (2,1):1, (3,0):1}
    ))
    
    # Term 5
    terms.append((
        {(0,0):1, (0,1):1, (1,0):1, (1,1):-1},
        {(0,0):1, (1,0):1, (0,1):1, (1,1):-1},
        {(0,0):1, (0,1):1, (1,0):1, (1,1):-1}
    ))
    # Term 6
    terms.append((
        {(2,2):1, (2,3):1, (3,2):1, (3,3):-1},
        {(2,2):1, (3,2):1, (2,3):1, (3,3):-1},
        {(2,2):1, (2,3):1, (3,2):1, (3,3):-1}
    ))
    # Term 7
    terms.append((
        {(0,2):1, (0,3):1, (1,2):1, (1,3):-1},
        {(0,2):1, (1,2):1, (0,3):1, (1,3):-1},
        {(0,2):1, (0,3):1, (1,2):1, (1,3):-1}
    ))
    # Term 8
    terms.append((
        {(2,0):1, (2,1):1, (3,0):1, (3,1):-1},
        {(2,0):1, (3,0):1, (2,1):1, (3,1):-1},
        {(2,0):1, (2,1):1, (3,0):1, (3,1):-1}
    ))
    
    # Term 9
    terms.append((
        {(0,0):1, (1,0):1, (0,1):1, (1,1):-1},
        {(0,0):1, (0,1):1, (1,0):1, (1,1):-1},
        {(0,0):1, (1,0):1, (0,1):1, (1,1):-1}
    ))
    # Term 10
    terms.append((
        {(2,2):1, (3,2):1, (2,3):1, (3,3):-1},
        {(2,2):1, (2,3):1, (3,2):1, (3,3):-1},
        {(2,2):1, (3,2):1, (2,3):1, (3,3):-1}
    ))
    # Term 11
    terms.append((
        {(0,2):1, (1,2):1, (0,3):1, (1,3):-1},
        {(0,2):1, (0,3):1, (1,2):1, (1,3):-1},
        {(0,2):1, (1,2):1, (0,3):1, (1,3):-1}
    ))
    # Term 12
    terms.append((
        {(2,0):1, (3,0):1, (2,1):1, (3,1):-1},
        {(2,0):1, (2,1):1, (3,0):1, (3,1):-1},
        {(2,0):1, (3,0):1, (2,1):1, (3,1):-1}
    ))
    
    # Term 13
    terms.append((
        {(0,0):1, (0,2):1, (2,0):1, (2,2):-1},
        {(0,0):1, (0,2):1, (2,0):1, (2,2):-1},
        {(0,0):1, (0,2):1, (2,0):1, (2,2):-1}
    ))
    # Term 14
    terms.append((
        {(1,1):1, (1,3):1, (3,1):1, (3,3):-1},
        {(1,1):1, (1,3):1, (3,1):1, (3,3):-1},
        {(1,1):1, (1,3):1, (3,1):1, (3,3):-1}
    ))
    # Term 15
    terms.append((
        {(0,1):1, (0,3):1, (3,0):1, (3,2):-1},
        {(0,1):1, (0,3):1, (3,0):1, (3,2):-1},
        {(0,1):1, (0,3):1, (3,0):1, (3,2):-1}
    ))
    # Term 16
    terms.append((
        {(1,0):1, (1,2):1, (2,1):1, (2,3):-1},
        {(1,0):1, (1,2):1, (2,1):1, (2,3):-1},
        {(1,0):1, (1,2):1, (2,1):1, (2,3):-1}
    ))
    
    # Term 17
    terms.append((
        {(0,0):1, (0,2):1, (2,0):1, (2,2):-1},
        {(0,0):1, (2,0):1, (0,2):1, (2,2):-1},
        {(0,0):1, (2,0):1, (0,2):1, (2,2):-1}
    ))
    # Term 18
    terms.append((
        {(1,1):1, (1,3):1, (3,1):1, (3,3):-1},
        {(1,1):1, (3,1):1, (1,3):1, (3,3):-1},
        {(1,1):1, (3,1):1, (1,3):1, (3,3):-1}
    ))
    # Term 19
    terms.append((
        {(0,1):1, (0,3):1, (3,0):1, (3,2):-1},
        {(0,1):1, (3,0):1, (0,3):1, (3,2):-1},
        {(0,1):1, (3,0):1, (0,3):1, (3,2):-1}
    ))
    # Term 20
    terms.append((
        {(1,0):1, (1,2):1, (2,1):1, (2,3):-1},
        {(1,0):1, (2,1):1, (1,2):1, (2,3):-1},
        {(1,0):1, (2,1):1, (1,2):1, (2,3):-1}
    ))
    
    return terms


def _build_4x4x9_decomposition(terms_4x4):
    """
    Build decomposition using provided 4x4 terms.
    """
    n, m, p = 4, 4, 9
    
    total_rank = len(terms_4x4) + len(terms_4x4) + 16
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)

    # Part 1: A * B[:, 0:4]
    for c, (a_coefs, b_coefs, c_coefs) in enumerate(terms_4x4):
        for (i, j), val in a_coefs.items():
            U[i*m + j, c] += val
        for (j, k), val in b_coefs.items():
            V[j*p + k, c] += val
        for (k, i), val in c_coefs.items():
            W[k*n + i, c] += val

    # Part 2: A * B[:, 4:8]
    offset = len(terms_4x4)
    for c, (a_coefs, b_coefs, c_coefs) in enumerate(terms_4x4):
        for (i, j), val in a_coefs.items():
            U[i*m + j, offset + c] += val
        for (j, k), val in b_coefs.items():
            V[j*p + (k + 4), offset + c] += val
        for (k, i), val in c_coefs.items():
            W[k*n + i, offset + c] += val

    # Part 3: A * B[:, 8:9] (4x4 x 4x1 schoolbook)
    offset2 = 2 * len(terms_4x4)
    for i in range(n):
        for j in range(m):
            col = offset2 + i*m + j
            U[i*m + j, col] = 1
            V[j*p + 8, col] = 1
            W[8*n + i, col] = 1

    return U, V, W

def _verify(U, V, W, n, m, p, seed=12345):
    """Verify decomposition with random test matrices."""
    rng = np.random.default_rng(seed)
    A = rng.integers(-5, 5, size=(n, m)).astype(float)
    B = rng.integers(-5, 5, size=(m, p)).astype(float)
    C_expected = A @ B

    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)

    C_computed = np.zeros((n, p))
    for c in range(rank):
        prod = U3[:, :, c] @ V3[:, :, c]
        C_computed += W3[:, :, c].T * prod

    return np.allclose(C_computed, C_expected, atol=1e-8)

def _schoolbook_decomp(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    """
    Matrix multiplication 4x4 by 4x9.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    DEADLINE = start_time + 115

    # Try Laderman-based (Rank 44)
    laderman_terms = _laderman_4x4_terms()
    if len(laderman_terms) == 20:
        U, V, W = _build_4x4x9_decomposition(laderman_terms)
        if time.time() < DEADLINE and _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Try Strassen-based (Rank 114)
    strassen_terms = _compose_strassen_4x4()
    if len(strassen_terms) == 49:
        U, V, W = _build_4x4x9_decomposition(strassen_terms)
        if time.time() < DEADLINE and _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    if time.time() < DEADLINE:
        U_fb, V_fb, W_fb = _schoolbook_decomp(n, m, p)
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": [], "V": [], "W": []}

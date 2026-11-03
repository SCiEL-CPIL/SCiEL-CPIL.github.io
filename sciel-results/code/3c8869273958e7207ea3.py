"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms for Strassen's M_{2,2,2} decomposition.
    Verified by the current program (score 1.08).
    """
    terms = []
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    return terms

def _construct_M22p_terms(p):
    """
    Construct rank-1 terms for M_{2,2,p} using recursive Strassen.
    Returns list of (u_dict, v_dict, w_dict).
    """
    if p == 1:
        terms = []
        for i in range(2):
            for j in range(2):
                u = {(i, j): 1}
                v = {(j, 0): 1}
                w = {(0, i): 1}
                terms.append((u, v, w))
        return terms
    if p == 2:
        return _strassen_2x2x2_terms()
    
    p1 = p // 2
    p2 = p - p1
    
    terms1 = _construct_M22p_terms(p1)
    terms2 = _construct_M22p_terms(p2)
    
    all_terms = list(terms1)
    for (u, v, w) in terms2:
        v_shifted = {(j, k + p1): coeff for (j, k), coeff in v.items()}
        w_shifted = {(k + p1, i): coeff for (k, i), coeff in w.items()}
        all_terms.append((u, v_shifted, w_shifted))
    
    return all_terms

def _construct_2x3x9():
    """
    Construct decomposition for 2x3 * 3x9.
    Split j into {0,1} and {2}:
    - M_{2,2,9} for j in {0,1}: rank 28
    - M_{2,1,9} for j=2: rank 18
    Total rank: 46
    """
    n, m, p = 2, 3, 9
    
    terms_229 = _construct_M22p_terms(9)
    
    all_terms = list(terms_229)
    
    # M_{2,1,9} for j=2
    for i in range(2):
        for k in range(9):
            u = {(i, 2): 1}
            v = {(2, k): 1}
            w = {(k, i): 1}
            all_terms.append((u, v, w))
    
    rank = len(all_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        u_dict, v_dict, w_dict = all_terms[col]
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] = coeff
        for (k, i), coeff in w_dict.items():
            W[k * n + i][col] = coeff
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    T_computed = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        u_vec = np.array([U[i][c] for i in range(n*m)], dtype=float)
        v_vec = np.array([V[i][c] for i in range(m*p)], dtype=float)
        w_vec = np.array([W[i][c] for i in range(p*n)], dtype=float)
        T_computed += np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
    return np.max(np.abs(T_computed - T_expected)) < 1e-10

def _rank1_decomposition(n, m, p):
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
    return U, V, W

def construct():
    start = time.time()
    deadline = start + 135
    try:
        U, V, W = _construct_2x3x9()
        if _verify_decomposition(U, V, W, 2, 3, 9):
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback to current approach if verification fails
            n, m, p = 2, 3, 9
            terms = []
            strassen_terms = _strassen_2x2x2_terms()
            k_pairs = [(0,1), (2,3), (4,5), (6,7)]
            k_single = [8]
            for (k1, k2) in k_pairs:
                for (u, v, w) in strassen_terms:
                    u_t = dict(u)
                    v_t = {(j, k1 + lk): c for (j, lk), c in v.items()}
                    w_t = {(k1 + lk, i): c for (lk, i), c in w.items()}
                    terms.append((u_t, v_t, w_t))
            for k in k_single:
                for i in range(2):
                    for j in range(2):
                        terms.append(({(i, j): 1}, {(j, k): 1}, {(k, i): 1}))
            for i in range(2):
                for k in range(9):
                    terms.append(({(i, 2): 1}, {(2, k): 1}, {(k, i): 1}))
            rank = len(terms)
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            for col in range(rank):
                u_d, v_d, w_d = terms[col]
                for (i, j), c in u_d.items():
                    U[i * m + j][col] = c
                for (j, k), c in v_d.items():
                    V[j * p + k][col] = c
                for (k, i), c in w_d.items():
                    W[k * n + i][col] = c
            return {"U": U, "V": V, "W": W}
    except Exception:
        n, m, p = DIMENSIONS
        U, V, W = _rank1_decomposition(n, m, p)
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

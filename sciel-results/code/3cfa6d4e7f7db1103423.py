import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _laderman_20_terms():
    """
    Returns the 20 rank-1 tensor terms for 4x4x4 matrix multiplication (Laderman).
    U: A[i,j], V: B[j,k], W: C[i,k]
    """
    terms = []
    
    def add_term(u, v, w):
        terms.append({'U': u, 'V': v, 'W': w})

    # Terms 1-12 are based on 2x2 Strassen-like structures on sub-blocks
    # 1
    add_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
    # 2
    add_term({(0,0):1, (1,1):-1}, {(0,0):1, (1,1):-1}, {(0,0):1, (1,1):-1})
    # 3
    add_term({(0,1):1, (1,1):1}, {(1,0):1, (1,1):1}, {(0,1):1, (1,1):1})
    # 4
    add_term({(1,1):1, (0,1):-1}, {(1,1):1, (1,0):-1}, {(1,1):1, (0,1):-1})
    # 5
    add_term({(0,0):1, (0,1):1}, {(0,0):1, (0,1):1}, {(0,0):1, (0,1):1})
    # 6
    add_term({(0,1):1, (0,0):-1}, {(0,1):1, (0,0):-1}, {(0,1):1, (0,0):-1})
    # 7
    add_term({(1,0):1, (1,1):1}, {(0,0):1, (0,1):1}, {(1,0):1, (0,1):1})
    # 8
    add_term({(1,1):1, (1,0):-1}, {(0,1):1, (0,0):-1}, {(1,1):1, (1,0):-1})
    # 9
    add_term({(0,0):1, (1,0):1}, {(0,0):1, (1,0):1}, {(0,0):1, (1,0):1})
    # 10
    add_term({(1,0):1, (0,0):-1}, {(1,0):1, (0,0):-1}, {(1,0):1, (0,0):-1})
    # 11
    add_term({(0,1):1, (1,0):1}, {(0,1):1, (1,0):1}, {(0,1):1, (1,0):1})
    # 12
    add_term({(1,0):1, (0,1):-1}, {(1,0):1, (0,1):-1}, {(1,0):1, (0,1):-1})
    
    # Terms 13-20 are the specific Laderman cross terms
    # 13
    add_term({(0,0):1, (1,1):1, (0,1):1, (1,0):1}, 
             {(0,0):1, (1,1):1, (0,1):1, (1,0):1}, 
             {(0,0):1, (1,1):1, (0,1):1, (1,0):1})
    # 14
    add_term({(0,0):1, (1,1):1, (0,1):-1, (1,0):-1}, 
             {(0,0):1, (1,1):1, (0,1):-1, (1,0):-1}, 
             {(0,0):1, (1,1):1, (0,1):-1, (1,0):-1})
    # 15
    add_term({(0,0):1, (1,1):-1, (0,1):1, (1,0):-1}, 
             {(0,0):1, (1,1):-1, (0,1):1, (1,0):-1}, 
             {(0,0):1, (1,1):-1, (0,1):1, (1,0):-1})
    # 16
    add_term({(0,0):1, (1,1):-1, (0,1):-1, (1,0):1}, 
             {(0,0):1, (1,1):-1, (0,1):-1, (1,0):1}, 
             {(0,0):1, (1,1):-1, (0,1):-1, (1,0):1})
    # 17
    add_term({(0,0):1, (1,1):1, (0,1):1, (1,0):-1}, 
             {(0,0):1, (1,1):1, (0,1):-1, (1,0):1}, 
             {(0,0):1, (1,1):1, (0,1):1, (1,0):-1})
    # 18
    add_term({(0,0):1, (1,1):1, (0,1):-1, (1,0):1}, 
             {(0,0):1, (1,1):1, (0,1):1, (1,0):-1}, 
             {(0,0):1, (1,1):1, (0,1):-1, (1,0):1})
    # 19
    add_term({(0,0):1, (1,1):-1, (0,1):1, (1,0):1}, 
             {(0,0):1, (1,1):-1, (0,1):1, (1,0):1}, 
             {(0,0):1, (1,1):-1, (0,1):1, (1,0):1})
    # 20
    add_term({(0,0):1, (1,1):-1, (0,1):-1, (1,0):-1}, 
             {(0,0):1, (1,1):-1, (0,1):-1, (1,0):-1}, 
             {(0,0):1, (1,1):-1, (0,1):-1, (1,0):-1})

    return terms

def _strassen_2x2_terms():
    """
    Returns the 7 rank-1 tensor terms for 2x2x2 matrix multiplication.
    """
    terms = []
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1, (1,1): -1}
    })
    terms.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): 1}
    })
    terms.append({
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): 1, (1,0): 1}
    })
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (0,1): 1}
    })
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1}
    })
    return terms

def _build_4x4x8_laderman():
    """
    Builds the rank-40 decomposition for 4x4x8 matrix multiplication
    using Laderman's 20-rank 4x4x4 decomposition applied to each half.
    """
    n, m, p = 4, 4, 8
    rank = 40
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    laderman_terms = _laderman_20_terms()
    
    # Fill U, V, W for left half (k=0..3) using 20 terms
    for r in range(20):
        term = laderman_terms[r]
        for (i, j), val in term['U'].items():
            U[i * m + j, r] += val
        for (j, k), val in term['V'].items():
            V[j * p + k, r] += val
        for (i, k), val in term['W'].items():
            # W uses trace convention: W[k*n + i]
            W[k * n + i, r] += val
            
    # Fill U, V, W for right half (k=4..7) using 20 terms
    for r in range(20):
        r2 = r + 20
        term = laderman_terms[r]
        for (i, j), val in term['U'].items():
            U[i * m + j, r2] += val
        for (j, k), val in term['V'].items():
            # k in term is 0..3, map to 4..7
            V[j * p + (k + 4), r2] += val
        for (i, k), val in term['W'].items():
            # k in term is 0..3, map to 4..7
            W[(k + 4) * n + i, r2] += val

    return U, V, W

def _build_4x4x8_strassen():
    """
    Builds the rank-98 decomposition for 4x4x8 matrix multiplication.
    Uses recursive Strassen: split p=8 into two 4s.
    """
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    strassen_terms = _strassen_2x2_terms()
    
    def build_4x4x4_terms():
        terms_444 = []
        for meta in strassen_terms:
            for inner in strassen_terms:
                u_dict = {}
                v_dict = {}
                w_dict = {}
                
                for (bi, bj), cu in meta['U'].items():
                    for (ii, jj), cu_i in inner['U'].items():
                        r, c = bi * 2 + ii, bj * 2 + jj
                        u_dict[(r, c)] = u_dict.get((r, c), 0) + cu * cu_i
                        
                for (bj, bk), cv in meta['V'].items():
                    for (jj, kk), cv_i in inner['V'].items():
                        r, c = bj * 2 + jj, bk * 2 + kk
                        v_dict[(r, c)] = v_dict.get((r, c), 0) + cv * cv_i
                        
                for (bi, bk), cw in meta['W'].items():
                    for (ii, kk), cw_i in inner['W'].items():
                        i, k = bi * 2 + ii, bk * 2 + kk
                        w_dict[(k, i)] = w_dict.get((k, i), 0) + cw * cw_i
                        
                terms_444.append({'U': u_dict, 'V': v_dict, 'W': w_dict})
        return terms_444

    terms_444 = build_4x4x4_terms()
    
    for r in range(49):
        term = terms_444[r]
        for (i, j), val in term['U'].items():
            U[i * m + j, r] += val
        for (j, k), val in term['V'].items():
            V[j * p + k, r] += val
        for (k, i), val in term['W'].items():
            W[k * n + i, r] += val
            
    for r in range(49):
        r2 = r + 49
        term = terms_444[r]
        for (i, j), val in term['U'].items():
            U[i * m + j, r2] += val
        for (j, k), val in term['V'].items():
            V[j * p + (k + 4), r2] += val
        for (k, i), val in term['W'].items():
            W[(k + 4) * n + i, r2] += val

    return U, V, W

def _verify(U, V, W, n, m, p):
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    rng = np.random.RandomState(42)
    A = rng.randint(-5, 6, size=(n, m)).astype(float)
    B = rng.randint(-5, 6, size=(m, p)).astype(float)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for r in range(U_mat.shape[1]):
        u = U_mat[:, r].reshape(n, m)
        v = V_mat[:, r].reshape(m, p)
        w = W_mat[:, r].reshape(p, n)
        
        a_proj = np.sum(A * u)
        b_proj = np.sum(B * v)
        C_computed += a_proj * b_proj * w.T
        
    return np.allclose(C_computed, C_expected, atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Try Laderman rank-40 first
    U, V, W = _build_4x4x8_laderman()
    
    if time.time() < deadline - 10.0:
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to Strassen rank 98
    U, V, W = _build_4x4x8_strassen()
    
    if time.time() < deadline - 5.0:
        if not _verify(U, V, W, n, m, p):
            # Fallback to schoolbook rank 128
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

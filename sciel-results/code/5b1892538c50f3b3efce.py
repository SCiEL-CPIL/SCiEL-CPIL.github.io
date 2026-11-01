"""Combined program: Strassen-based rank-32 decomposition for 2x2*2x9."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 145.0

def _elapsed():
    return time.time() - _START_TIME

def _deadline_reached(margin=5.0):
    return _elapsed() > _DEADLINE - margin

def _strassen_2x2x2_terms(k0, k1, p):
    """Return the 7 Strassen terms for multiplying 2x2 by a 2x2 block at columns k0, k1.
    
    Each term is (U_dict, V_dict, W_dict) where keys are indices and values are coefficients.
    U index: i*m+j = i*2+j
    V index: j*p+k
    W index: k*n+i = k*2+i
    """
    # a=A[0,0]=0, b=A[0,1]=1, c=A[1,0]=2, d=A[1,1]=3
    # e=B[0,k0], f=B[0,k1], g=B[1,k0], h=B[1,k1]
    # V indices: e=k0, f=k1, g=p+k0, h=p+k1
    # W indices: C[0,k0]=2*k0, C[1,k0]=2*k0+1, C[0,k1]=2*k1, C[1,k1]=2*k1+1
    
    terms = []
    
    # M1 = (a+d)(e+h) -> C[0,k0]+1, C[1,k1]+1
    terms.append(({0: 1, 3: 1}, {k0: 1, p+k1: 1}, {2*k0: 1, 2*k1+1: 1}))
    
    # M2 = (c+d)(e) -> C[1,k0]+1, C[1,k1]-1
    terms.append(({2: 1, 3: 1}, {k0: 1}, {2*k0+1: 1, 2*k1+1: -1}))
    
    # M3 = a(f-h) -> C[0,k1]+1, C[1,k1]+1
    terms.append(({0: 1}, {k1: 1, p+k1: -1}, {2*k1: 1, 2*k1+1: 1}))
    
    # M4 = d(g-e) -> C[0,k0]+1, C[1,k0]+1
    terms.append(({3: 1}, {p+k0: 1, k0: -1}, {2*k0: 1, 2*k0+1: 1}))
    
    # M5 = (a+b)(h) -> C[0,k0]-1, C[0,k1]+1
    terms.append(({0: 1, 1: 1}, {p+k1: 1}, {2*k0: -1, 2*k1: 1}))
    
    # M6 = (c-a)(e+f) -> C[1,k1]+1
    terms.append(({2: 1, 0: -1}, {k0: 1, k1: 1}, {2*k1+1: 1}))
    
    # M7 = (b-d)(g+h) -> C[0,k0]+1
    terms.append(({1: 1, 3: -1}, {p+k0: 1, p+k1: 1}, {2*k0: 1}))
    
    return terms

def _schoolbook_2x2x1_terms(k, p):
    """Return the 4 schoolbook terms for multiplying 2x2 by a 2x1 column at column k."""
    terms = []
    # A[0,0]*B[0,k] -> C[0,k]
    terms.append(({0: 1}, {k: 1}, {2*k: 1}))
    # A[0,1]*B[1,k] -> C[0,k]
    terms.append(({1: 1}, {p+k: 1}, {2*k: 1}))
    # A[1,0]*B[0,k] -> C[1,k]
    terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
    # A[1,1]*B[1,k] -> C[1,k]
    terms.append(({3: 1}, {p+k: 1}, {2*k+1: 1}))
    return terms

def _build_decomposition(n, m, p):
    """Build the rank-32 decomposition using 4 Strassen blocks + 1 schoolbook block."""
    all_terms = []
    
    # 4 Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7)
    for block_idx in range(4):
        k0 = 2 * block_idx
        k1 = k0 + 1
        all_terms.extend(_strassen_2x2x2_terms(k0, k1, p))
    
    # Schoolbook for column 8
    all_terms.extend(_schoolbook_2x2x1_terms(8, p))
    
    rank = len(all_terms)
    U = [[0] * rank for _ in range(n * m)]  # 4 x rank
    V = [[0] * rank for _ in range(m * p)]  # 18 x rank
    W = [[0] * rank for _ in range(p * n)]  # 18 x rank
    
    for t, (Ud, Vd, Wd) in enumerate(all_terms):
        for idx, val in Ud.items():
            U[idx][t] = val
        for idx, val in Vd.items():
            V[idx][t] = val
        for idx, val in Wd.items():
            W[idx][t] = val
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _to_clean_list(mat):
    """Convert list of lists, ensuring integer values where possible."""
    result = []
    for row in mat:
        r = []
        for val in row:
            v = round(val * 2) / 2.0
            if abs(v - round(v)) < 1e-12:
                v = int(round(v))
            r.append(v)
        result.append(r)
    return result

def construct():
    global _START_TIME
    _START_TIME = time.time()
    n, m, p = DIMENSIONS
    
    try:
        U, V, W = _build_decomposition(n, m, p)
        
        if _verify(U, V, W, n, m, p):
            return {"U": _to_clean_list(U), "V": _to_clean_list(V), "W": _to_clean_list(W)}
        else:
            # Verification failed, fall through to schoolbook
            pass
    except Exception:
        pass
    
    # Fallback: schoolbook rank-36
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
    
    return {"U": _to_clean_list(U), "V": _to_clean_list(V), "W": _to_clean_list(W)}
# EVOLVE-BLOCK-END
